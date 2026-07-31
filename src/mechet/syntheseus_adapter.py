"""Adapters from MechET proof hypotheses to Syntheseus search models.

The adapter intentionally consumes an offline candidate pool first. This keeps
search benchmarking deterministic and separates planner comparisons from online
LLM serving. An online actor can later implement the same candidate-provider
contract without changing Syntheseus search code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from rdkit import Chem

from .proof_program import parse_proof_program


def canonical_unmapped(smiles: str) -> str:
    """Canonicalize a dot-separated SMILES string after removing atom maps."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@dataclass(frozen=True)
class PoolCandidate:
    target: str
    precursor: str
    score: float
    proof: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class MechETCandidatePool:
    """Canonical target-to-precursor index built from hypothesis JSONL files."""

    def __init__(self, candidates: Iterable[PoolCandidate]) -> None:
        self._index: dict[str, list[PoolCandidate]] = {}
        for item in candidates:
            key = canonical_unmapped(item.target)
            self._index.setdefault(key, []).append(item)
        for key in self._index:
            self._index[key].sort(key=lambda item: item.score, reverse=True)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "MechETCandidatePool":
        candidates: list[PoolCandidate] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for item in row.get("hypotheses") or row.get("candidates") or []:
                if not item.get("execute_ok", True):
                    continue
                precursor = str(
                    item.get("derived_core_precursor")
                    or item.get("derived_precursor")
                    or item.get("precursor")
                    or ""
                ).strip()
                proof = str(item.get("proof") or item.get("prediction") or "")
                target = str(row.get("target") or row.get("product") or "").strip()
                if not target and proof:
                    try:
                        target = parse_proof_program(proof).target_smiles
                    except Exception:
                        target = ""
                if not target or not precursor:
                    continue
                score = item.get("forward_rank_score")
                if score is None:
                    evidence = item.get("forward_evidence") or {}
                    score = (
                        float(item.get("model_logprob") or item.get("score") or 0.0)
                        + float(evidence.get("target_score") or 0.0)
                        + 0.5 * float(evidence.get("selectivity_margin") or 0.0)
                        - 0.25 * float(evidence.get("uncertainty") or 0.0)
                    )
                if not math.isfinite(float(score)):
                    continue
                candidates.append(
                    PoolCandidate(
                        target=target,
                        precursor=precursor,
                        score=float(score),
                        proof=proof,
                        metadata={
                            key: value
                            for key, value in item.items()
                            if key not in {"proof", "prediction"}
                        },
                    )
                )
        return cls(candidates)

    def query(self, target_smiles: str, num_results: int = 20) -> list[PoolCandidate]:
        """Return ranked candidates for a mapped or unmapped target SMILES."""
        key = canonical_unmapped(target_smiles)
        return list(self._index.get(key, ()))[: max(int(num_results), 0)]

    @property
    def n_targets(self) -> int:
        return len(self._index)

    @property
    def n_candidates(self) -> int:
        return sum(len(values) for values in self._index.values())


def normalized_probabilities(values: Sequence[PoolCandidate]) -> list[float]:
    if not values:
        return []
    maximum = max(item.score for item in values)
    weights = [math.exp(max(-50.0, min(50.0, item.score - maximum))) for item in values]
    total = sum(weights)
    return [weight / total for weight in weights]


try:  # Optional dependency: imported only for planning runs.
    from syntheseus import BackwardReactionModel, Bag, Molecule, SingleProductReaction
except ImportError:  # pragma: no cover - exercised when the optional extra is absent.
    BackwardReactionModel = object  # type: ignore[assignment,misc]
    Bag = Molecule = SingleProductReaction = None  # type: ignore[assignment]


class MechETBackwardReactionModel(BackwardReactionModel):  # type: ignore[misc]
    """Syntheseus backward model backed by MechET hypothesis candidates."""

    def __init__(
        self,
        candidate_pool: MechETCandidatePool | str | Path,
        *,
        default_num_results: int = 20,
        **kwargs: Any,
    ) -> None:
        if Molecule is None:
            raise ImportError("install mechet[planning] to use Syntheseus")
        super().__init__(**kwargs)
        self.pool = (
            candidate_pool
            if isinstance(candidate_pool, MechETCandidatePool)
            else MechETCandidatePool.from_jsonl(candidate_pool)
        )
        self.default_num_results = int(default_num_results)

    @staticmethod
    def _reaction(
        product: Any,
        candidate: PoolCandidate,
        probability: float,
    ) -> Any:
        fragments = [
            Molecule(fragment)
            for fragment in canonical_unmapped(candidate.precursor).split(".")
            if fragment
        ]
        return SingleProductReaction(
            reactants=Bag(fragments),
            product=product,
            metadata={
                "probability": float(probability),
                "mechet_score": candidate.score,
                "proof": candidate.proof,
                "mechet": candidate.metadata,
            },
        )

    def _get_reactions(
        self,
        inputs: list[Any],
        num_results: int,
    ) -> list[Sequence[Any]]:
        limit = int(num_results or self.default_num_results)
        output: list[Sequence[Any]] = []
        for product in inputs:
            candidates = self.pool.query(product.smiles, num_results=limit)
            probabilities = normalized_probabilities(candidates)
            output.append(
                [
                    self._reaction(product, candidate, probability)
                    for candidate, probability in zip(candidates, probabilities)
                ]
            )
        return output
