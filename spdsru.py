import torch
import math
import torch.nn.init


def FM(A, B, a, n):
    '''
    Compute the Weighted Fréchet mean
    '''
    return torch.add((1.0 - a) * A, a * B)


def NUS(W_root, A, a_num, tot, n=1):
    '''
    Compute the weighted average on the M -> Y
    '''
    W = torch.pow(W_root, 2)
    if a_num == 1:
        return (W[0] / tot) * A
    else:
        result = torch.squeeze(A[:, :1, :, :]) * (W[0] / tot)
        for i in range(1, A.shape[1]):
            result = result + torch.squeeze(A[:, i : i + 1, :, :]) * (W[i] / tot)
        return result


def MatrixExp(B, l, n):
    '''
    input a matrix B, and the total length to be calculated, n is the size of B
    output the somehow exp(B) = I + B + B^2 / 2! + B^3 / 3! + ... + B^l / l!
    '''

    Result = torch.eye(n, device=B.device)
    return torch.matmul(torch.inverse(torch.subtract(Result, B)), torch.add(Result, B))


def Translation(A, B, n, batch_size):

    '''
    input the matrix A and vector B
    change B to be SO
    like [[0 ,  1, 2]
          [-1,  0, 3]
          [-2, -3, 0]]
    return B * A * B.T
    '''
    power_matrix = 5
    B = torch.reshape(B, [1, -1])

    # lower_triangel = fill_triangular(B)
    line_B = [torch.zeros([1, n], device=A.device)]
    for i in range(n - 1):
        temp_line = torch.cat(
            [B[:1, i : 2 * i + 1], torch.zeros([1, n - i - 1], device=A.device)], axis=1
        )
        line_B.append(temp_line)

    lower_triangel = torch.cat(line_B, axis=0)

    B_matrix = torch.subtract(lower_triangel, lower_triangel.T)

    B_matrix = MatrixExp(B_matrix, power_matrix, n)

    B_matrix = torch.unsqueeze(B_matrix, 0).repeat([batch_size, 1, 1])

    Tresult = torch.matmul(B_matrix, A)  # B * A

    Tresult = torch.matmul(Tresult, B_matrix.permute([0, 2, 1]))  # B * A * B.T
    return Tresult


def Chol_de(A, n):
    '''
    input matrix A and it's size n
    decomponent by Cholesky
    return a vector with size n*(n+1)/2
    '''
    # A = tf.add (A , 1e-10 * tf.diag(tf.random_uniform([n])) )
    # A = tf.cond(
    #     tf.greater( tf.matrix_determinant(A),tf.constant(0.0) ) ,
    #     lambda: A,
    #     lambda: tf.add (A , 1e-10 * tf.eye(n) ) )
    # L = tf.cholesky(A)

    L = A
    result = L[:, :1, :1]
    for i in range(1, n):
        j = i
        result = torch.cat([result, L[:, i : i + 1, : j + 1]], axis=2)

    result = torch.reshape(result, [-1, n * (n + 1) // 2])
    return result


def Chol_com(l, n, batch_size):
    '''
    input vector l and target shape n and eps to be the smallest value
    return lower trangle matrix
    '''
    lower_triangle_ = torch.unsqueeze(
        torch.cat([l[:, :1], torch.zeros((batch_size, n - 1))], axis=1),
        1,
    )
    for i in range(1, n):
        lower_triangle_ = torch.cat(
            [
                lower_triangle_,
                torch.unsqueeze(
                    torch.cat(
                        [
                            l[:, i * (i + 1) // 2 : i * (i + 1) // 2 + i + 1],
                            torch.zeros((batch_size, n - i - 1)),
                        ],
                        axis=1,
                    ),
                    1,
                ),
            ],
            axis=1,
        )

    lower_triangle_ = torch.add(
        lower_triangle_,
        torch.unsqueeze(torch.eye(n) * 1e-2, axis=0).repeat([batch_size, 1, 1]),
    )
    result = torch.matmul(lower_triangle_, lower_triangle_.transpose())
    return result


class SPDSRU(torch.nn.Module):
    """
    Implements a simple distribution based recurrent unit that keeps moving
    averages of the mean map embeddings of features of inputs on manifold.
    """

    def __init__(self, alpha, batch_size, matrix_size, eps=1e-10):
        super(SPDSRU, self).__init__()
        self._alpha = alpha
        self._a_num = len(alpha)
        self._batch_size = batch_size
        self._matrix_size = n = matrix_size
        self._eps = eps

        self.WR_root = torch.nn.Parameter(torch.Tensor(self._a_num))
        self.Wt_root = torch.nn.Parameter(torch.Tensor(1))
        self.Wphi_root = torch.nn.Parameter(torch.Tensor(1))
        self.Ws_root = torch.nn.Parameter(torch.Tensor(self._a_num))

        self.Br = torch.nn.Parameter(torch.Tensor(n * (n - 1) // 2, 1))
        self.Bt = torch.nn.Parameter(torch.Tensor(n * (n - 1) // 2, 1))
        self.By = torch.nn.Parameter(torch.Tensor(n * (n - 1) // 2, 1))

        self.reset_parameters()

    @property
    def state_size(self):
        return int(self._a_num * self._matrix_size * self._matrix_size)

    @property
    def output_size(self):
        return int(self._matrix_size * self._matrix_size)

    def forward(self, inputs, state, scope=None):
        a_num = self._a_num
        batch_size = self._batch_size
        eps = self._eps
        n = self._matrix_size
        a = self._alpha

        Xt = inputs
        Mt_1 = torch.reshape(
            state, [-1, self._a_num, self._matrix_size, self._matrix_size]
        )

        n_current_X = torch.reshape(Xt, [batch_size, n, n])
        Yt = NUS(
            self.WR_root,
            Mt_1,
            a_num,
            torch.sum(torch.pow(self.WR_root, 2)) + eps,
            n,
        )
        Rt = Translation(Yt, self.Br, n, batch_size)
        Tt = FM(
            n_current_X,
            Rt,
            torch.pow(self.Wt_root, 2)
            / (
                torch.sum(
                    torch.pow(self.Wt_root, 2)
                    + torch.pow(self.Wphi_root, 2)
                )
                + eps
            ),
            n,
        )
        Phit = Translation(Tt, self.Bt, n, batch_size)

        next_state = []
        for j in range(a_num):
            next_state.append(
                torch.unsqueeze(
                    FM(
                        torch.reshape(Mt_1[:, j : j + 1, :n, :n], [batch_size, n, n]),
                        Phit,
                        a[j],
                        n,
                    ),
                    1,
                )
            )
        Mt = torch.cat(next_state, axis=1)
        St = NUS(
            self.Ws_root,
            Mt,
            a_num,
            torch.sum(torch.pow(self.Ws_root, 2)) + eps,
            n,
        )
        Ot = Translation(St, self.By, n, batch_size)

        out_state = torch.reshape(
            Mt, [-1, self._a_num, self._matrix_size, self._matrix_size]
        )

        output = torch.reshape(Ot, [-1, self._matrix_size, self._matrix_size])

        return output, out_state

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self._matrix_size) if self._matrix_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)


class SPDLSTM(torch.nn.Module):
    def __init__(self, alpha, batch_size, matrix_size, eps=1e-10) -> None:
        super().__init__()
        self.alpha = alpha
        self.cell = SPDSRU(alpha, batch_size, matrix_size, eps)

    def forward(self, input, state=None):
        if state is None:
            state = torch.diag_embed(
                torch.ones(
                    input.shape[1],
                    len(self.alpha),
                    input.shape[-1],
                    requires_grad=True,
                    device=input.device,
                )
            )
        output = []
        for input_mat in input:
            output_mat, state = self.cell(input_mat, state)
            output.append(output_mat)
        return torch.stack(output), state


if __name__ == '__main__':
    alpha = [0.01, 0.25, 0.5, 0.9, 0.99]
    spdsru = SPDLSTM(alpha, 1, 10)
    optimizer = torch.optim.Adam(spdsru.parameters())
    spdsru.train()
    state = torch.diag_embed(1e-1 * torch.ones(1, len(alpha), 10))

    for _ in range(3):
        input = torch.rand(1, 1, 10, 10, requires_grad=True)
        input, state = spdsru(input)
        loss = torch.sum(input)
        loss.backward(retain_graph=True)
        optimizer.step()
        # state = state.detach()
        optimizer.zero_grad()
