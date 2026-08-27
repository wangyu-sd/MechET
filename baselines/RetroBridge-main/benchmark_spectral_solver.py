#!/usr/bin/env python3
"""Benchmark the robust bucketed spectral solver on CPU or GPU."""

import argparse
import time

import torch

from src.features.extra_features import eigh_real_nodes


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_laplacians(batch_size, min_nodes, max_nodes, edge_probability, device):
    counts = torch.linspace(min_nodes, max_nodes, batch_size, device=device).long()
    node_index = torch.arange(max_nodes, device=device)
    mask = node_index.unsqueeze(0) < counts.unsqueeze(1)
    random_edges = torch.rand(batch_size, max_nodes, max_nodes, device=device)
    adjacency = (random_edges < edge_probability).float().triu(diagonal=1)
    adjacency = adjacency + adjacency.transpose(-1, -2)
    adjacency = adjacency * mask.unsqueeze(1) * mask.unsqueeze(2)
    laplacian = torch.diag_embed(adjacency.sum(dim=-1)) - adjacency
    return laplacian, mask, counts


def legacy_per_graph_fp64(laplacian, counts):
    results = []
    for graph_index, count in enumerate(counts.tolist()):
        results.append(torch.linalg.eigh(laplacian[graph_index, :count, :count].double()))
    return results


def benchmark(name, operation, repeats, device):
    operation()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        operation()
    synchronize(device)
    seconds = (time.perf_counter() - started) / repeats
    print(f"{name}: {seconds:.6f} seconds/batch", flush=True)
    return seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-nodes", type=int, default=64)
    parser.add_argument("--max-nodes", type=int, default=112)
    parser.add_argument("--edge-probability", type=float, default=0.03)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)
    laplacian, mask, counts = make_laplacians(
        args.batch_size,
        args.min_nodes,
        args.max_nodes,
        args.edge_probability,
        device,
    )
    print(
        f"device={device} batch={args.batch_size} "
        f"nodes={args.min_nodes}..{args.max_nodes}",
        flush=True,
    )

    legacy_seconds = None
    if not args.skip_legacy:
        legacy_seconds = benchmark(
            "legacy_per_graph_fp64",
            lambda: legacy_per_graph_fp64(laplacian, counts),
            args.repeats,
            device,
        )
    bucketed_seconds = benchmark(
        "bucketed_fp32_with_fallback",
        lambda: eigh_real_nodes(laplacian, mask, compute_eigenvectors=True),
        args.repeats,
        device,
    )
    if legacy_seconds is not None:
        print(f"speedup={legacy_seconds / bucketed_seconds:.3f}x", flush=True)


if __name__ == "__main__":
    main()
