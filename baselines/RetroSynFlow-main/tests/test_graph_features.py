import unittest
from unittest.mock import patch

import torch

from retflow.utils.graph_features import eigh_real_nodes


class EigenFeaturesTest(unittest.TestCase):
    def test_padding_and_isolated_nodes_have_stable_spectrum(self):
        laplacian = torch.zeros((2, 8, 8), dtype=torch.float32)
        laplacian[0, :2, :2] = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        laplacian[1, :4, :4] = torch.tensor(
            [
                [1.0, -1.0, 0.0, 0.0],
                [-1.0, 2.0, -1.0, 0.0],
                [0.0, -1.0, 2.0, -1.0],
                [0.0, 0.0, -1.0, 1.0],
            ]
        )
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )

        values, vectors = eigh_real_nodes(laplacian, mask)

        self.assertEqual(values.shape, (2, 8))
        self.assertEqual(vectors.shape, (2, 8, 8))
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.isfinite(vectors).all())
        self.assertEqual(int((values[0] < 1e-5).sum()), 4)
        self.assertTrue(torch.equal(vectors[0, 5:], torch.zeros_like(vectors[0, 5:])))

    def test_failed_batch_retries_graphs_in_float64(self):
        laplacian = torch.tensor(
            [
                [[1.0, -1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        )
        mask = torch.ones((2, 2), dtype=torch.bool)
        original_eigh = torch.linalg.eigh
        calls = []

        def fail_batch_once(matrix):
            calls.append((tuple(matrix.shape), matrix.dtype))
            if matrix.ndim == 3:
                raise RuntimeError("synthetic batched solver failure")
            return original_eigh(matrix)

        with patch("torch.linalg.eigh", side_effect=fail_batch_once):
            values, vectors = eigh_real_nodes(laplacian, mask)

        self.assertEqual(calls[0], ((2, 2, 2), torch.float32))
        self.assertEqual(
            calls[1:],
            [((2, 2), torch.float64), ((2, 2), torch.float64)],
        )
        self.assertEqual(values.dtype, torch.float32)
        self.assertEqual(vectors.dtype, torch.float32)

    def test_rejects_non_contiguous_masks(self):
        laplacian = torch.zeros((1, 3, 3))
        mask = torch.tensor([[True, False, True]])

        with self.assertRaisesRegex(ValueError, "right-padded contiguous"):
            eigh_real_nodes(laplacian, mask)


if __name__ == "__main__":
    unittest.main()
