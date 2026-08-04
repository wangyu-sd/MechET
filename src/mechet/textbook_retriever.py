"""Deterministic BM25 and molecular-state retrieval for textbook passages."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Sequence

from rdkit import Chem

from .textbook_store import TextbookPassage, TextbookStore

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+\-]{1,}")
_FUNCTIONAL_GROUP_SMARTS = {
    "carbonyl": "[CX3]=[OX1]",
    "aldehyde": "[CX3H1](=O)[#6,H]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[CX3](=O)[NX3]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "alkyl_halide": "[CX4]-[F,Cl,Br,I]",
    "aryl_halide": "[c]-[F,Cl,Br,I]",
    "alcohol": "[OX2H][#6]",
    "alkoxide": "[O-][#6]",
    "amine": "[NX3;H0,H1,H2;!$(N-C=O)]",
    "nitrile": "[CX2]#[NX1]",
    "nitro": "[$([NX3+](=O)[O-]),$([NX3](=O)=O)]",
    "epoxide": "[OX2r3]1[CX4r3][CX4r3]1",
    "aromatic": "[a]",
}
_COMPILED_FUNCTIONAL_GROUPS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in _FUNCTIONAL_GROUP_SMARTS.items()
}


@dataclass(frozen=True)
class RetrievalResult:
    passage: TextbookPassage
    score: float
    lexical_score: float
    state_score: float
    matched_terms: tuple[str, ...]
    state_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passage": self.passage.to_dict(),
            "score": self.score,
            "lexical_score": self.lexical_score,
            "state_score": self.state_score,
            "matched_terms": list(self.matched_terms),
            "state_terms": list(self.state_terms),
        }


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def molecular_state_terms(smiles: str) -> tuple[str, ...]:
    if not smiles:
        return ()
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        return ()
    terms: set[str] = set()
    for name, query in _COMPILED_FUNCTIONAL_GROUPS.items():
        if query is not None and mol.HasSubstructMatch(query):
            terms.add(name)
    if any(atom.GetFormalCharge() < 0 for atom in mol.GetAtoms()):
        terms.update(("anionic", "nucleophile"))
    if any(atom.GetFormalCharge() > 0 for atom in mol.GetAtoms()):
        terms.update(("cationic", "electrophile"))
    if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
        terms.add("aromatic")
    if any(atom.IsInRing() for atom in mol.GetAtoms()):
        terms.add("ring")
    if any(atom.GetAtomicNum() in {9, 17, 35, 53} for atom in mol.GetAtoms()):
        terms.update(("halide", "leaving_group"))
    return tuple(sorted(terms))


class TextbookRetriever:
    """Deterministic indexed retriever suitable for frozen matched experiments."""

    def __init__(
        self,
        store: TextbookStore,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        state_weight: float = 0.35,
    ) -> None:
        self.store = store
        self.k1 = float(k1)
        self.b = float(b)
        self.state_weight = float(state_weight)
        self.documents = [self._document_tokens(item) for item in store.passages]
        self.lengths = [len(tokens) for tokens in self.documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.term_frequencies: list[dict[str, int]] = []
        self.passage_term_sets: list[set[str]] = []
        self.document_frequency: dict[str, int] = {}
        self.inverted_index: dict[str, list[int]] = {}
        for index, tokens in enumerate(self.documents):
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            term_set = set(frequencies)
            self.term_frequencies.append(frequencies)
            self.passage_term_sets.append(term_set)
            for token in term_set:
                self.document_frequency[token] = (
                    self.document_frequency.get(token, 0) + 1
                )
                self.inverted_index.setdefault(token, []).append(index)
        self._manifest = {
            "strategy": "indexed_bm25_plus_molecular_state_terms",
            "k1": self.k1,
            "b": self.b,
            "state_weight": self.state_weight,
            "n_documents": len(self.documents),
            "vocabulary_size": len(self.document_frequency),
            "corpus": self.store.manifest(),
        }

    @staticmethod
    def _document_tokens(item: TextbookPassage) -> list[str]:
        metadata = " ".join(
            (
                item.title,
                *item.topics,
                *item.reaction_families,
                *item.functional_groups,
            )
        )
        return tokenize(metadata + " " + item.text)

    def _idf(self, term: str) -> float:
        n_docs = len(self.documents)
        df = self.document_frequency.get(term, 0)
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def _bm25(self, query_terms: Sequence[str], index: int) -> float:
        frequencies = self.term_frequencies[index]
        if not frequencies or not query_terms:
            return 0.0
        length = self.lengths[index]
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            numerator = frequency * (self.k1 + 1.0)
            denominator = frequency + self.k1 * (
                1.0
                - self.b
                + self.b * length / max(self.average_length, 1.0)
            )
            score += self._idf(term) * numerator / denominator
        return score

    def _state_match(
        self, index: int, state_terms: set[str]
    ) -> tuple[float, set[str]]:
        matched = state_terms & self.passage_term_sets[index]
        return len(matched) / max(len(state_terms), 1), matched

    def retrieve(
        self,
        query: str = "",
        *,
        state_smiles: str = "",
        top_k: int = 6,
        source_ids: Iterable[str] = (),
        max_per_source: int = 3,
    ) -> list[RetrievalResult]:
        top_k = max(int(top_k), 0)
        if top_k == 0:
            return []
        state_terms = set(molecular_state_terms(state_smiles))
        query_terms = tokenize(query)
        if not query_terms:
            query_terms = sorted(state_terms)
        scoring_terms = set(query_terms) | state_terms
        candidate_indexes: set[int] = set()
        for term in scoring_terms:
            candidate_indexes.update(self.inverted_index.get(term, ()))
        if not candidate_indexes:
            return []

        allowed_sources = set(map(str, source_ids))
        results: list[RetrievalResult] = []
        for index in candidate_indexes:
            item = self.store.passages[index]
            if allowed_sources and item.source_id not in allowed_sources:
                continue
            lexical = self._bm25(query_terms, index)
            state_score, state_matches = self._state_match(index, state_terms)
            matched_query = set(query_terms) & self.passage_term_sets[index]
            score = lexical + self.state_weight * state_score
            if score <= 0:
                continue
            results.append(
                RetrievalResult(
                    passage=item,
                    score=float(score),
                    lexical_score=float(lexical),
                    state_score=float(state_score),
                    matched_terms=tuple(sorted(matched_query)),
                    state_terms=tuple(sorted(state_matches)),
                )
            )
        results.sort(
            key=lambda result: (
                result.score,
                result.passage.source_id,
                result.passage.passage_id,
            ),
            reverse=True,
        )
        output: list[RetrievalResult] = []
        per_source: dict[str, int] = {}
        for result in results:
            source_id = result.passage.source_id
            if (
                max_per_source > 0
                and per_source.get(source_id, 0) >= max_per_source
            ):
                continue
            output.append(result)
            per_source[source_id] = per_source.get(source_id, 0) + 1
            if len(output) >= top_k:
                break
        return output

    def manifest(self) -> dict[str, Any]:
        return {
            **self._manifest,
            "corpus": {
                **self._manifest["corpus"],
                "source_counts": dict(
                    self._manifest["corpus"]["source_counts"]
                ),
                "license_counts": dict(
                    self._manifest["corpus"]["license_counts"]
                ),
            },
        }
