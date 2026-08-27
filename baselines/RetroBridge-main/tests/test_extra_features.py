import unittest
from unittest.mock import patch

import torch

from src.features.extra_features import EigenFeatures


class EigenFeaturesTest(unittest.TestCase):
    @staticmethod
    def noisy_graph(dtype=torch.float32):
        n = 8
        edges = torch.zeros((1, n, n, 2), dtype=dtype)
        edges[..., 0] = 1
        edges[:, 0, 1, 0] = 0
        edges[:, 1, 0, 0] = 0
        edges[:, 0, 1, 1] = 1
        edges[:, 1, 0, 1] = 1
        return {
            'E_t': edges,
            'node_mask': torch.ones((1, n), dtype=torch.bool),
        }

    def test_all_features_use_float64_solver_and_restore_dtype(self):
        original_eigh = torch.linalg.eigh
        solver_dtypes = []

        def recording_eigh(matrix):
            solver_dtypes.append(matrix.dtype)
            return original_eigh(matrix)

        with patch('torch.linalg.eigh', side_effect=recording_eigh):
            _, eigenvalues, _, eigenvectors = EigenFeatures('all')(self.noisy_graph())

        self.assertEqual(solver_dtypes, [torch.float64])
        self.assertEqual(eigenvalues.dtype, torch.float32)
        self.assertEqual(eigenvectors.dtype, torch.float32)
        self.assertTrue(torch.isfinite(eigenvalues).all())
        self.assertTrue(torch.isfinite(eigenvectors).all())

    def test_eigenvalue_features_use_float64_solver_and_restore_dtype(self):
        original_eigvalsh = torch.linalg.eigvalsh
        solver_dtypes = []

        def recording_eigvalsh(matrix):
            solver_dtypes.append(matrix.dtype)
            return original_eigvalsh(matrix)

        with patch('torch.linalg.eigvalsh', side_effect=recording_eigvalsh):
            _, eigenvalues = EigenFeatures('eigenvalues')(self.noisy_graph())

        self.assertEqual(solver_dtypes, [torch.float64])
        self.assertEqual(eigenvalues.dtype, torch.float32)
        self.assertTrue(torch.isfinite(eigenvalues).all())


if __name__ == '__main__':
    unittest.main()
