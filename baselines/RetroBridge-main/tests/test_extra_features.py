import unittest
from unittest.mock import patch

import torch

from src.features.extra_features import EigenFeatures, eigh_real_nodes


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

    def test_all_features_use_batched_float32_fast_path(self):
        original_eigh = torch.linalg.eigh
        solver_dtypes = []

        def recording_eigh(matrix):
            solver_dtypes.append(matrix.dtype)
            return original_eigh(matrix)

        with patch('torch.linalg.eigh', side_effect=recording_eigh):
            _, eigenvalues, _, eigenvectors = EigenFeatures('all')(self.noisy_graph())

        self.assertEqual(solver_dtypes, [torch.float32])
        self.assertEqual(eigenvalues.dtype, torch.float32)
        self.assertEqual(eigenvectors.dtype, torch.float32)
        self.assertTrue(torch.isfinite(eigenvalues).all())
        self.assertTrue(torch.isfinite(eigenvectors).all())

    def test_eigenvalue_features_use_batched_float32_fast_path(self):
        original_eigvalsh = torch.linalg.eigvalsh
        solver_dtypes = []

        def recording_eigvalsh(matrix):
            solver_dtypes.append(matrix.dtype)
            return original_eigvalsh(matrix)

        with patch('torch.linalg.eigvalsh', side_effect=recording_eigvalsh):
            _, eigenvalues = EigenFeatures('eigenvalues')(self.noisy_graph())

        self.assertEqual(solver_dtypes, [torch.float32])
        self.assertEqual(eigenvalues.dtype, torch.float32)
        self.assertTrue(torch.isfinite(eigenvalues).all())

    def test_groups_graphs_by_valid_node_count(self):
        laplacian = torch.zeros((4, 6, 6), dtype=torch.float32)
        masks = torch.tensor([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
        ], dtype=torch.bool)
        for index, count in enumerate((3, 3, 5, 5)):
            adjacency = torch.zeros((count, count))
            adjacency[torch.arange(count - 1), torch.arange(1, count)] = 1
            adjacency = adjacency + adjacency.T
            laplacian[index, :count, :count] = torch.diag(adjacency.sum(1)) - adjacency

        original_eigh = torch.linalg.eigh
        batch_shapes = []

        def recording_eigh(matrix):
            batch_shapes.append(tuple(matrix.shape))
            return original_eigh(matrix)

        with patch('torch.linalg.eigh', side_effect=recording_eigh):
            values, vectors = eigh_real_nodes(laplacian, masks, compute_eigenvectors=True)

        # Both sizes fit in one capped bucket: one solve replaces four calls.
        self.assertEqual(batch_shapes, [(4, 6, 6)])
        self.assertEqual(values.shape, (4, 6))
        self.assertEqual(vectors.shape, (4, 6, 6))
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.isfinite(vectors).all())
        self.assertTrue(torch.equal(vectors[0, 3:], torch.zeros_like(vectors[0, 3:])))

    def test_failed_group_retries_only_affected_graphs_in_float64(self):
        laplacian = torch.tensor([
            [[1.0, -1.0], [-1.0, 1.0]],
            [[1.0, -1.0], [-1.0, 1.0]],
        ])
        mask = torch.ones((2, 2), dtype=torch.bool)
        original_eigh = torch.linalg.eigh
        calls = []

        def fail_group_once(matrix):
            calls.append((tuple(matrix.shape), matrix.dtype))
            if matrix.ndim == 3:
                raise RuntimeError("synthetic grouped solver failure")
            return original_eigh(matrix)

        with patch('torch.linalg.eigh', side_effect=fail_group_once):
            values, vectors = eigh_real_nodes(laplacian, mask, compute_eigenvectors=True)

        self.assertEqual(calls[0], ((2, 2, 2), torch.float32))
        self.assertEqual(calls[1:], [
            ((2, 2), torch.float64),
            ((2, 2), torch.float64),
        ])
        self.assertEqual(values.dtype, torch.float32)
        self.assertEqual(vectors.dtype, torch.float32)

    def test_rejects_non_contiguous_masks(self):
        laplacian = torch.zeros((1, 3, 3))
        mask = torch.tensor([[True, False, True]])
        with self.assertRaisesRegex(ValueError, "right-padded contiguous"):
            eigh_real_nodes(laplacian, mask, compute_eigenvectors=True)

    def test_bucket_padding_preserves_molecular_eigenvalues(self):
        adjacency = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        molecule_laplacian = torch.diag(adjacency.sum(1)) - adjacency
        padded_laplacian = torch.zeros((1, 8, 8))
        padded_laplacian[0, :4, :4] = molecule_laplacian
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)

        values, _ = eigh_real_nodes(
            padded_laplacian, mask, compute_eigenvectors=False
        )
        reference = torch.linalg.eigvalsh(molecule_laplacian)
        self.assertTrue(torch.allclose(values[0, :4], reference, atol=1e-5))
        self.assertTrue((values[0, 4:] > reference.max()).all())

    def test_isolated_node_tie_break_preserves_component_count(self):
        # One two-node component plus three isolated nodes => four components.
        laplacian = torch.zeros((1, 5, 5))
        laplacian[0, :2, :2] = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        mask = torch.ones((1, 5), dtype=torch.bool)
        values, _ = eigh_real_nodes(laplacian, mask, compute_eigenvectors=False)
        self.assertEqual(int((values[0] < 1e-5).sum()), 4)


if __name__ == '__main__':
    unittest.main()
