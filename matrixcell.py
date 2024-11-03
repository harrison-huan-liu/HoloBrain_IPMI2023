# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()
import tensorflow as tf
import numpy as np


def FM(A, B, a, n):
    '''
    Compute the Weighted Fréchet mean
    '''
    return tf.add((1.0 - a) * A, a * B)


def NUS(W_root, A, a_num, tot, n=1):
    '''
    Compute the weighted average on the M -> Y
    '''
    W = tf.pow(W_root, 2)
    if a_num == 1:
        return (W[0] / tot) * A
    else:
        result = tf.squeeze(tf.slice(A, [0, 0, 0, 0], [-1, 1, -1, -1])) * (W[0] / tot)
        for i in range(1, A.shape[1]):
            result = result + tf.squeeze(tf.slice(A, [0, i, 0, 0], [-1, 1, -1, -1])) * (
                W[i] / tot
            )
        return result


def MatrixExp(B, l, n):
    '''
    input a matrix B, and the total length to be calculated, n is the size of B
    output the somehow exp(B) = I + B + B^2 / 2! + B^3 / 3! + ... + B^l / l!
    '''

    Result = tf.eye(n)
    return tf.matmul(tf.linalg.inv(tf.subtract(Result, B)), tf.add(Result, B))


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
    B = tf.reshape(B, [1, -1])

    # lower_triangel = fill_triangular(B)
    line_B = [tf.zeros([1, n])]
    for i in range(n - 1):
        temp_line = tf.concat(
            [tf.slice(B, [0, i], [1, i + 1]), tf.zeros([1, n - i - 1])], axis=1
        )
        line_B.append(temp_line)

    lower_triangel = tf.concat(line_B, axis=0)

    B_matrix = tf.subtract(lower_triangel, tf.transpose(lower_triangel))

    B_matrix = MatrixExp(B_matrix, power_matrix, n)

    B_matrix = tf.tile(tf.expand_dims(B_matrix, 0), [batch_size, 1, 1])

    Tresult = tf.matmul(B_matrix, A)  # B * A

    Tresult = tf.matmul(Tresult, tf.transpose(B_matrix, [0, 2, 1]))  # B * A * B.T
    return Tresult


def Chol_de(A, n, batch_size):
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
    result = tf.slice(L, [0, 0, 0], [-1, 1, 1])
    for i in range(1, n):
        j = i
        result = tf.concat([result, tf.slice(L, [0, i, 0], [-1, 1, j + 1])], axis=2)

    result = tf.reshape(result, [-1, n * (n + 1) // 2])
    return result


def Chol_com(l, n, batch_size):
    '''
    input vector l and target shape n and eps to be the smallest value
    return lower trangle matrix
    '''
    lower_triangle_ = tf.expand_dims(
        tf.concat(
            [tf.slice(l, [0, 0], [-1, 1]), tf.zeros((batch_size, n - 1))], axis=1
        ),
        1,
    )
    for i in range(1, n):
        lower_triangle_ = tf.concat(
            [
                lower_triangle_,
                tf.expand_dims(
                    tf.concat(
                        [
                            tf.slice(l, [0, i * (i + 1) // 2], [-1, i + 1]),
                            tf.zeros((batch_size, n - i - 1)),
                        ],
                        axis=1,
                    ),
                    1,
                ),
            ],
            axis=1,
        )

    lower_triangle_ = tf.add(
        lower_triangle_,
        tf.tile(tf.expand_dims(tf.eye(n) * 1e-2, axis=0), [batch_size, 1, 1]),
    )
    result = tf.matmul(lower_triangle_, lower_triangle_, transpose_b=True)
    return result


class SPDSRU(tf.compat.v1.nn.rnn_cell.RNNCell):
    """
    Implements a simple distribution based recurrent unit that keeps moving
    averages of the mean map embeddings of features of inputs on manifold.
    """

    def __init__(self, alpha, batch_size, matrix_size, eps=1e-10):
        self._alpha = alpha
        self._a_num = len(alpha)
        self._batch_size = batch_size
        self._matrix_size = matrix_size
        self._eps = eps

        a_num = self._a_num
        n = self._matrix_size

        self.Weights_rnn = {
            'WR_root': tf.Variable(tf.random.uniform([a_num]), name='WR_root'),
            'Wt_root': tf.Variable(tf.random.uniform([1]), name='Wt_root'),
            'Wphi_root': tf.Variable(tf.random.uniform([1]), name='Wphi_root'),
            'Ws_root': tf.Variable(tf.random.uniform([a_num]), name='Ws_root'),
        }  # To make every weights to be positive.

        self.Bias_rnn = {
            'Br': tf.Variable(tf.random.uniform([n * (n - 1) // 2, 1]), name='Br'),
            'Bt': tf.Variable(tf.random.uniform([n * (n - 1) // 2, 1]), name='Bt'),
            'By': tf.Variable(tf.random.uniform([n * (n - 1) // 2, 1]), name='By'),
        }

    @property
    def state_size(self):
        return int(self._a_num * self._matrix_size * self._matrix_size)

    @property
    def output_size(self):
        return int(self._matrix_size * self._matrix_size)

    def __call__(self, inputs, state, scope=None):
        a_num = self._a_num
        batch_size = self._batch_size
        eps = self._eps
        n = self._matrix_size
        a = self._alpha

        Xt = inputs
        Mt_1 = tf.reshape(
            state, [-1, self._a_num, self._matrix_size, self._matrix_size]
        )

        n_current_X = tf.reshape(Xt, [batch_size, n, n])
        Yt = NUS(
            self.Weights_rnn['WR_root'],
            Mt_1,
            a_num,
            tf.reduce_sum(tf.pow(self.Weights_rnn['WR_root'], 2)) + eps,
            n,
        )
        Rt = Translation(Yt, self.Bias_rnn['Br'], n, batch_size)
        Tt = FM(
            n_current_X,
            Rt,
            tf.pow(self.Weights_rnn['Wt_root'], 2)
            / (
                tf.reduce_sum(
                    [
                        tf.pow(self.Weights_rnn['Wt_root'], 2),
                        tf.pow(self.Weights_rnn['Wphi_root'], 2),
                    ]
                )
                + eps
            ),
            n,
        )
        Phit = Translation(Tt, self.Bias_rnn['Bt'], n, batch_size)

        next_state = []
        for j in range(a_num):
            next_state.append(
                tf.expand_dims(
                    FM(
                        tf.reshape(
                            tf.slice(Mt_1, [0, j, 0, 0], [-1, 1, n, n]),
                            [batch_size, n, n],
                        ),
                        Phit,
                        a[j],
                        n,
                    ),
                    1,
                )
            )
        Mt = tf.concat(next_state, axis=1)
        St = NUS(
            self.Weights_rnn['Ws_root'],
            Mt,
            a_num,
            tf.reduce_sum(tf.pow(self.Weights_rnn['Ws_root'], 2)) + eps,
            n,
        )
        Ot = Translation(St, self.Bias_rnn['By'], n, batch_size)

        out_state = tf.reshape(
            Mt, [-1, int(self._a_num * self._matrix_size * self._matrix_size)]
        )

        output = tf.reshape(Ot, [-1, int(self._matrix_size * self._matrix_size)])

        return (output, out_state)


if __name__ == '__main__':
    alpha = tf.constant([0, 0.25, 0.5, 0.9, 0.99])
    batch_size = 4
    matrix_size = 10
    inputs = tf.random.normal([batch_size, matrix_size, matrix_size])
    inputs = inputs @ tf.transpose(inputs, [0, 2, 1])
    state = tf.random.normal([batch_size, alpha.shape[0], matrix_size, matrix_size])
    state = state @ tf.transpose(state, [0, 1, 3, 2])
    cell = SPDSRU([0, 0.25, 0.5, 0.9, 0.99], 4, 10)
    output, out_state = cell(inputs, state)
    print(output, out_state)

