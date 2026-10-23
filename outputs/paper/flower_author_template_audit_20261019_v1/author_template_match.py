"""Joint-state SMARTS compatibility, NOT chemical feasibility certification.

Author rules run FORWARD; saved events run INVERSE. Ordinary mapped explicit
hydrogens are collapsed to H counts, without neutralization/isotope removal.
Both states must satisfy the full rule under ONE private-atom correspondence,
and the rule must explain ALL bond/charge/H-count changes, not just a motif.
Reagent screening and the stage sequence are intentionally not certified.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
import re

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.warning')


class UnsupportedTemplate(ValueError):
    pass


class MatchWorkLimit(RuntimeError):
    pass


def read_literal(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(name)


def molecule(s):
    p = Chem.SmilesParserParams()
    p.removeHs = False
    m = Chem.MolFromSmiles(s, p)
    if m is None:
        raise ValueError('unparseable saved state')
    maps = [a.GetAtomMapNum() for a in m.GetAtoms()]
    if not all(maps) or len(set(maps)) != len(maps):
        raise ValueError('state needs unique positive atom maps')
    p = Chem.RemoveHsParameters()
    p.removeMapped = True
    p.removeIsotopes = False
    p.removeHydrides = False
    m = Chem.RemoveHs(m, p)
    return m


def fixed_h(a):
    desc = a.DescribeQuery()
    nums = re.findall(r'AtomHCount (\d+) = val', desc)
    if len(nums) == 1 and 'AtomOr' not in desc and '!=' not in desc:
        return int(nums[0])
    return None


def query_charge(a):
    # QueryAtom.GetFormalCharge() is zero even for many explicit SMARTS
    # charge predicates. Read the query, not that concrete-atom property.
    nums = re.findall(r'AtomFormalCharge (-?\d+) = val', a.DescribeQuery())
    if len(set(nums)) > 1:
        raise UnsupportedTemplate('ambiguous charge disjunction')
    return int(nums[0]) if nums else 0


def bond_key(b):
    return tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())))


def bonds(m):
    return {bond_key(b): b for b in m.GetBonds()}


@dataclass
class Rule:
    smarts: str
    left: object
    right: object
    a0: dict
    a1: dict
    b0: dict
    b1: dict
    db: dict
    dq: dict
    dh: dict
    centre: set
    signature: tuple
    refs: list


def signature(db, dq):
    return tuple(sorted(db.values())), tuple(sorted(q for q in dq.values() if q))


def typed_charge_key(rule):
    return tuple(sorted(((rule.a0[i].GetAtomicNum(), q, rule.dh[i])
                         for i, q in rule.dq.items() if q),
                        key=lambda x: (x[0], x[1], -999 if x[2] is None else x[2])))


@lru_cache(maxsize=65536)
def compile_rule(smarts):
    left, right = [Chem.MolFromSmarts(s) for s in smarts.split('>>')]
    if left is None or right is None:
        raise UnsupportedTemplate('SMARTS parse failure')
    a0 = {a.GetAtomMapNum(): a for a in left.GetAtoms()}
    a1 = {a.GetAtomMapNum(): a for a in right.GetAtoms()}
    if 0 in a0 or 0 in a1 or len(a0) != left.GetNumAtoms() or len(a1) != right.GetNumAtoms():
        raise UnsupportedTemplate('unmapped or repeated template atom')
    if a0.keys() != a1.keys():
        raise UnsupportedTemplate('template adds/deletes mapped atoms')
    # Explicit H-only/hydride rules are retained in the source catalog but not
    # claimed covered by this H-count-normalized matcher.
    if any(a.GetAtomicNum() == 1 for a in (*a0.values(), *a1.values())):
        raise UnsupportedTemplate('explicit hydrogen template requires separate matcher')
    b0, b1 = bonds(left), bonds(right)
    db = {k: (b0[k].GetBondTypeAsDouble() if k in b0 else 0,
              b1[k].GetBondTypeAsDouble() if k in b1 else 0)
          for k in b0.keys() | b1.keys()
          if (b0[k].GetBondTypeAsDouble() if k in b0 else 0) !=
             (b1[k].GetBondTypeAsDouble() if k in b1 else 0)}
    dq = {i: query_charge(a1[i]) - query_charge(a0[i]) for i in a0}
    dh = {}
    for i in a0:
        h0, h1 = fixed_h(a0[i]), fixed_h(a1[i])
        dh[i] = h1-h0 if h0 is not None and h1 is not None else (
            0 if a0[i].GetSmarts() == a1[i].GetSmarts() or h0 == h1 else None)
    centre = {x for k in db for x in k} | {i for i, q in dq.items() if q} | {
        i for i, h in dh.items() if h != 0}
    if not centre:
        raise UnsupportedTemplate('no bond/charge/H-count change encoded')
    return Rule(smarts, left, right, a0, a1, b0, b1, db, dq, dh, centre,
                signature(db, dq), [])


class Event:
    def __init__(self, before, after):
        # A saved inverse event's AFTER state is the forward reactant state.
        self.m0, self.m1 = molecule(after), molecule(before)
        self.a0 = {a.GetAtomMapNum(): a for a in self.m0.GetAtoms()}
        self.a1 = {a.GetAtomMapNum(): a for a in self.m1.GetAtoms()}
        if self.a0.keys() != self.a1.keys():
            raise ValueError('normalized atom inventory changed')
        if any((self.a0[i].GetAtomicNum(), self.a0[i].GetIsotope()) !=
               (self.a1[i].GetAtomicNum(), self.a1[i].GetIsotope()) for i in self.a0):
            raise ValueError('element/isotope inventory changed')
        self.b0, self.b1 = bonds(self.m0), bonds(self.m1)
        self.db = {k: (self.b0[k].GetBondTypeAsDouble() if k in self.b0 else 0,
                       self.b1[k].GetBondTypeAsDouble() if k in self.b1 else 0)
                   for k in self.b0.keys() | self.b1.keys()
                   if (self.b0[k].GetBondTypeAsDouble() if k in self.b0 else 0) !=
                      (self.b1[k].GetBondTypeAsDouble() if k in self.b1 else 0)}
        self.dq = {i: self.a1[i].GetFormalCharge()-self.a0[i].GetFormalCharge() for i in self.a0}
        self.dh = {i: self.a1[i].GetTotalNumHs()-self.a0[i].GetTotalNumHs() for i in self.a0}
        self.centre = {x for k in self.db for x in k} | {
            i for i in self.a0 if self.dq[i] or self.dh[i] or
            self.a1[i].GetNumRadicalElectrons() != self.a0[i].GetNumRadicalElectrons()}
        self.signature = signature(self.db, self.dq)
        self.atom_cache = {}
        self.radical_changed = any(self.a0[i].GetNumRadicalElectrons() !=
                                   self.a1[i].GetNumRadicalElectrons() for i in self.a0)

    def candidate_rules(self, groups):
        typed = [(self.a0[i].GetAtomicNum(), q, self.dh[i]) for i, q in self.dq.items() if q]
        keys = {tuple(sorted(v, key=lambda x: (x[0], x[1], -999 if x[2] is None else x[2])))
                for v in product(*[[(z, q, h), (0, q, h), (z, q, None), (0, q, None)]
                                   for z, q, h in typed])}
        for key in keys:
            yield from groups.get((self.signature, key), ())

    def atom_candidates(self, q0, q1):
        key = (q0.GetSmarts(), q1.GetSmarts())
        if key not in self.atom_cache:
            out = []
            for m, s in zip((self.m0, self.m1), key):
                q = Chem.MolFromSmarts(s)
                # Query initialization here also evaluates recursive SMARTS
                # in their actual molecular context, unlike QueryAtom.Match.
                out.append({m.GetAtomWithIdx(t[0]).GetAtomMapNum()
                            for t in m.GetSubstructMatches(q, uniquify=False, maxMatches=0)})
            self.atom_cache[key] = out[0] & out[1]
        return self.atom_cache[key]

    def match(self, rule, max_work=100000):
        if self.radical_changed or self.signature != rule.signature or len(self.centre) > len(rule.centre):
            return None
        candidates = {}
        for i in rule.a0:
            c = self.atom_candidates(rule.a0[i], rule.a1[i])
            c = {j for j in c if self.dq[j] == rule.dq[i] and
                 (rule.dh[i] is None or self.dh[j] == rule.dh[i])}
            c = c & self.centre if i in rule.centre else c - self.centre
            if not c:
                return None
            candidates[i] = c
        assigned, used = {}, set()
        union_bonds = rule.b0.keys() | rule.b1.keys()
        work = 0

        def pair_ok(i, j, x, y):
            k, a = tuple(sorted((i, j))), tuple(sorted((x, y)))
            for tb, eb in ((rule.b0, self.b0), (rule.b1, self.b1)):
                if k in tb:
                    if a not in eb or not tb[k].Match(eb[a]):
                        return False
                elif k in union_bonds and a in eb:
                    return False
            return True

        def search():
            nonlocal work
            work += 1
            if work > max_work:
                raise MatchWorkLimit('CSP safety guard reached; inconclusive, not a negative')
            if len(assigned) == len(candidates):
                mapped_db = {tuple(sorted((assigned[a], assigned[b]))): v
                             for (a, b), v in rule.db.items()}
                if mapped_db != self.db or not self.centre <= used:
                    return None
                return dict(assigned)
            rest = [i for i in candidates if i not in assigned]
            i = min(rest, key=lambda x: (x not in rule.centre,
                    -sum(tuple(sorted((x, j))) in union_bonds for j in assigned), len(candidates[x])))
            for x in sorted(candidates[i]-used):
                if all(pair_ok(i, j, x, y) for j, y in assigned.items()):
                    assigned[i] = x
                    used.add(x)
                    result = search()
                    if result:
                        return result
                    used.remove(x)
                    del assigned[i]
            return None
        return search()


def match_event(before, after, templates):
    event = Event(before, after)
    out = []
    for s in templates:
        rule = compile_rule(s)
        binding = event.match(rule)
        if binding:
            out.append({'reaction_smarts': s, 'template_to_state_atom_map': binding})
    return out


def load_catalog(index_paths, acid_base_path):
    """Expand only author-allowed acid/base pairs; no learned/generated rules.

    Reproduce author's truthy A/B threshold selection (including zero being
    inactive). No selection by gold reaction class. Original required/excluded
    reagent screens are retained as provenance but NOT certified by this local
    transition audit, since saved inverse states may omit original spectators.
    """
    import json
    acid_base = read_literal(acid_base_path, 'Acid_base')
    catalog, errors, counts = {}, [], Counter()
    for path in index_paths:
        version = Path(path).name.replace('_index.jsonl', '')
        for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
            row = json.loads(line)
            pka = row['pKa_constraint']
            variants = [(row['reaction_smarts'], None)]
            if pka:
                variants = []
                A, B = pka.get('A'), pka.get('B')
                for ab_index, ab in enumerate(acid_base):
                    if B and 'B' in ab['role'] and ab['pKa'] >= B:
                        pair = (ab['Base'], ab['Acid'])
                    elif A and not B and 'A' in ab['role'] and ab['pKa'] <= A:
                        pair = (ab['Acid'], ab['Base'])
                    else:
                        continue
                    left, right = row['reaction_smarts'].split('>>')
                    variants.append((left+'.'+pair[0]+'>>'+right+'.'+pair[1], ab_index))
            for s, ab_index in variants:
                counts[version+':expanded_occurrences'] += 1
                ref = dict(row, catalog_version=version, catalog_line=lineno,
                           acid_base_index=ab_index, expanded_smarts=s)
                try:
                    rule = compile_rule(s)
                    if s not in catalog:
                        catalog[s] = rule
                    catalog[s].refs.append(ref)
                except UnsupportedTemplate as exc:
                    errors.append(dict(ref, unsupported_reason=str(exc)))
    groups = defaultdict(list)
    for rule in catalog.values():
        groups[(rule.signature, typed_charge_key(rule))].append(rule)
    return groups, errors, dict(counts)
