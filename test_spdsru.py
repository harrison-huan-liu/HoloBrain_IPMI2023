import matrixcell
import spdsru
import torch
import unittest
import tensorflow as tf
import numpy as np


class Case(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_FM(self):
        a1 = torch.tensor([[2, 0], [0, 2]], dtype=torch.float32)
        b1 = torch.tensor([[3, 0], [0, 3]], dtype=torch.float32)
        c1 = torch.tensor([0.5], dtype=torch.float32)
        a2 = tf.constant([[2, 0], [0, 2]], tf.float32)
        b2 = tf.constant([[3, 0], [0, 3]], tf.float32)
        c2 = tf.constant([0.5], tf.float32)
        res1 = spdsru.FM(a1, b1, c1, None)
        res2 = matrixcell.FM(a2, b2, c2, None)
        self.assertTrue(np.all(np.isclose(res1, res2)))

    def test_spdsru(self):
        alpha = np.array([0, 0.25, 0.5, 0.9, 0.99], np.float32)
        batch_size = 4
        matrix_size = 10
        inputs = np.asarray(
            np.random.randn(batch_size, matrix_size, matrix_size), np.float32
        )
        inputs = inputs @ np.transpose(inputs, [0, 2, 1])
        state = np.asarray(
            np.random.randn(batch_size, alpha.shape[0], matrix_size, matrix_size),
            np.float32,
        )
        state = state @ np.transpose(state, [0, 1, 3, 2])

        cell1 = spdsru.SPDSRU(torch.from_numpy(alpha), batch_size, matrix_size)
        cell2 = matrixcell.SPDSRU(tf.convert_to_tensor(alpha), batch_size, matrix_size)

        for key, val in cell2.Weights_rnn.items():
            cell1.Weights_rnn[key] = torch.nn.Parameter(torch.from_numpy(val.numpy()))
        for key, val in cell2.Bias_rnn.items():
            cell1.Bias_rnn[key] = torch.nn.Parameter(torch.from_numpy(val.numpy()))

        output1, out_state1 = cell1(torch.from_numpy(inputs), torch.from_numpy(state))
        output2, out_state2 = cell2(
            tf.convert_to_tensor(inputs), tf.convert_to_tensor(state)
        )
        self.assertTrue(
            np.all(np.isclose(output1.detach().numpy(), output2.numpy(), 1e-2))
        )
        self.assertTrue(
            np.all(np.isclose(out_state1.detach().numpy(), out_state2.numpy(), 1e-2))
        )


if __name__ == '__main__':
    unittest.main()
