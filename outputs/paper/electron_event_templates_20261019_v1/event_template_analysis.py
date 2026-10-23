"""Offline, deterministic motif annotation of EXISTING saved inverse programs.

No model calls, executor changes, chemical feasibility certification or new
experimental outcomes. All motifs are analyst-authored graph rules; references
support the general mechanism, not these particular generated trajectories.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.warning')
ROOT = Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
WORK = Path('/aaa/fionafyang/buddy1/whaleywang/MechET-pr83-greedy')
SOURCE = ROOT / 'outputs/paper/nmi_literature_evidence/source_checked_20261019_v18'
REFS = {
    'arrows': 'https://openstax.org/books/organic-chemistry/pages/6-5-using-curved-arrows-in-polar-reaction-mechanisms',
    'carbonyl': 'https://openstax.org/books/organic-chemistry/pages/21-2-nucleophilic-acyl-substitution-reactions',
    'sn2': 'https://openstax.org/books/organic-chemistry/pages/11-3-characteristics-of-the-sn2-reaction',
    'e2': 'https://openstax.org/books/organic-chemistry/pages/11-8-the-e2-reaction-and-the-deuterium-isotope-effect',
}


def dumps(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


@lru_cache(maxsize=1024)
def molecule(smiles):
    p = Chem.SmilesParserParams()
    p.removeHs = False
    m = Chem.MolFromSmiles(smiles, p)
    if m is None:
        raise ValueError('saved state cannot be parsed')
    ids = [a.GetAtomMapNum() for a in m.GetAtoms()]
    if min(ids, default=0) <= 0 or len(set(ids)) != len(ids):
        raise ValueError('saved state requires unique positive private maps')
    return m


def graphs(smiles):
    m = molecule(smiles)
    atoms = {a.GetAtomMapNum(): a for a in m.GetAtoms()}
    bonds = {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()))):
             b.GetBondTypeAsDouble() for b in m.GetBonds()}
    return atoms, bonds


def inverse_moves(moves):
    """Invert a WHOLE coupled event, not the textual arrow order alone."""
    out = []
    for m in moves:
        s, t = m['source'], m['sink']
        sk, tk = s['kind'], t['kind']
        sa, ta = list(s['atoms']), list(t['atoms'])
        if sk == 'LP':
            pair = ta if tk == 'BOND' else sorted([sa[0], ta[0]])
            ns, nt = {'kind':'BOND', 'atoms':pair}, {'kind':'ATOM', 'atoms':sa}
        elif sk == 'BOND' and tk in {'ATOM', 'LP'}:
            ns, nt = {'kind':'LP', 'atoms':ta}, {'kind':'BOND', 'atoms':sa}
        elif sk == tk == 'BOND' or {sk, tk} == {'BOND', 'RADICAL_PAIR'}:
            ns, nt = dict(t), dict(s)
        else:
            raise ValueError('unsupported inverse arrow')
        out.append({'source':ns, 'sink':nt, 'electrons':2})
    return out


def operations(moves):
    adds, cuts, shifts, other = [], [], [], []
    for m in moves:
        s, t = m['source'], m['sink']
        sk, tk = s['kind'], t['kind']
        sa, ta = tuple(s['atoms']), tuple(t['atoms'])
        if sk == 'LP' and tk in {'ATOM', 'BOND'}:
            pair = ta if tk == 'BOND' else (sa[0], ta[0])
            if sa[0] not in pair or len(set(pair)) != 2:
                raise ValueError('invalid lone-pair bond formation')
            adds.append((sa[0], next(x for x in pair if x != sa[0]), tuple(sorted(pair))))
        elif sk == 'BOND' and tk in {'ATOM', 'LP'}:
            if ta[0] not in sa:
                raise ValueError('nonlocal bond-to-atom arrow')
            cuts.append((tuple(sorted(sa)), ta[0], next(x for x in sa if x != ta[0])))
        elif sk == tk == 'BOND':
            shifts.append((tuple(sorted(sa)), tuple(sorted(ta))))
        else:
            other.append(m)
    return adds, cuts, shifts, other


def classify_forward(state, moves):
    atoms, bonds = graphs(state)
    adds, cuts, shifts, other = operations(moves)
    def result(template, status='conditions_insufficient', flags=(), refs=('arrows',)):
        return {'template':template, 'status':status, 'flags':list(flags),
                'references':[REFS[k] for k in refs],
                'unassessed':['solvent_temperature_concentration', 'barriers_and_competition',
                              'availability_and_speciation_of_imports'],
                'interpretation':'forward counterpart of the complete saved inverse event'}
    def z(i):
        return atoms[i].GetAtomicNum()
    def order(pair):
        return bonds.get(tuple(sorted(pair)), 0)
    def acceptable_leaver(i):
        a = atoms[i]
        if z(i) in {17, 35, 53} or a.GetFormalCharge() > 0:
            return True
        # O bound to an oxidized sulfur (sulfonate-like leaving group).
        if z(i) == 8:
            for n in a.GetNeighbors():
                if n.GetAtomicNum() == 16 and sum(
                    b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(n).GetAtomicNum() == 8
                    for b in n.GetBonds()) >= 2:
                    return True
        return False
    if other:
        return result('radical_or_other_container_event', 'uncovered', ['outside polar motif rules'])
    if len(adds) == len(cuts) == 1 and not shifts and len(moves) == 2:
        donor, centre, formed = adds[0]
        broken, leaver, cut_centre = cuts[0]
        if centre == cut_centre and formed != broken:
            if z(centre) == 1 and order(broken) == 1 and order(formed) == 0:
                return result('proton_transfer', 'template_compatible',
                              ['acid-base pKa and site selectivity not determined'])
            # Nucleophilic addition to a carbonyl: pi electrons go to O.
            if z(centre) == 6 and z(leaver) == 8 and order(broken) == 2 and order(formed) == 0:
                return result('carbonyl_addition', 'template_compatible', refs=('carbonyl',))
            # O lone pair restores a carbonyl while a different sigma bond breaks.
            if z(centre) == 6 and z(donor) == 8 and order(formed) == order(broken) == 1:
                status = 'template_compatible' if acceptable_leaver(leaver) else 'conditions_insufficient'
                return result('carbonyl_elimination', status,
                              [] if status == 'template_compatible' else
                              ['leaving-group activation/stabilization requires context'], ('carbonyl',))
            if z(centre) == 6 and order(formed) == 0 and order(broken) == 1:
                a = atoms[centre]
                if a.GetIsAromatic() or any(b.GetBondTypeAsDouble() > 1 for b in a.GetBonds()):
                    return result('concerted_carbon_substitution', 'template_conflict',
                                  ['SN2 template inapplicable at sp2/sp carbon; alternative mechanisms not ruled out'], ('sn2',))
                nc = sum(n.GetAtomicNum() == 6 for n in a.GetNeighbors())
                if nc >= 3:
                    return result('concerted_carbon_substitution', 'template_conflict',
                                  ['tertiary carbon incompatible with ordinary SN2; not a proof of impossible net reaction'], ('sn2',))
                status = 'template_compatible' if acceptable_leaver(leaver) else 'conditions_insufficient'
                return result('concerted_carbon_substitution', status,
                              [] if status == 'template_compatible' else
                              ['poor or unclassified leaving group; activation is not established'], ('sn2',))
            if z(centre) in {15, 16}:
                return result('heteroatom_addition_elimination_or_substitution', flags=
                              ['P/S oxidation state and ligand context not covered by current chemical rules'])
            if order(broken) >= 2 or order(formed) >= 1:
                return result('other_multiple_bond_addition_elimination', flags=
                              ['not a carbonyl-specific template; atom roles require further classification'])
    if len(adds) == len(cuts) == len(shifts) == 1 and len(moves) == 3:
        donor, hydrogen, formed = adds[0]
        broken, leaving, carbon = cuts[0]
        source, sink = shifts[0]
        if z(hydrogen) == 1 and hydrogen in source:
            beta = next(i for i in source if i != hydrogen)
            if z(beta) == z(carbon) == 6 and set(sink) == {beta, carbon} and order(sink) == 1:
                return result('E2_like_elimination', flags=
                              ['anti-periplanar geometry and base/leaving-group compatibility not established'], refs=('e2',))
    if len(moves) == 1:
        if adds:
            donor, acceptor, pair = adds[0]
            if order(pair) == 0 and atoms[acceptor].GetFormalCharge() > 0:
                return result('lone_pair_capture_of_cation', 'template_compatible')
            return result('isolated_lone_pair_bond_formation', flags=
                          ['electron accounting alone does not establish a standalone elementary event'])
        if cuts:
            return result('isolated_bond_heterolysis_or_polarization', flags=
                          ['ion stabilization or coupling to another arrow not established'])
        if shifts:
            return result('bond_migration_or_pi_shift', flags=
                          ['migration acceptor activation/conjugation not established'])
    return result('unclassified_coupled_event', 'uncovered',
                  ['no full-event motif matched; not labelled chemically incorrect'])


def declared_deltas(moves):
    b, q = Counter(), Counter()
    adds, cuts, shifts, other = operations(moves)
    if other:
        raise ValueError('delta check does not implement radical containers')
    for donor, acceptor, pair in adds:
        b[pair] += 1
        q[donor] += 1
        q[acceptor] -= 1
    for pair, target, remainder in cuts:
        b[pair] -= 1
        q[target] -= 1
        q[remainder] += 1
    for source, sink in shifts:
        shared = set(source) & set(sink)
        if len(shared) != 1:
            raise ValueError('bond shift does not have one shared atom')
        b[source] -= 1
        b[sink] += 1
        q[next(x for x in source if x not in shared)] += 1
        q[next(x for x in sink if x not in shared)] -= 1
    return {k:v for k,v in b.items() if v}, {k:v for k,v in q.items() if v}


def analyze_moves(before, moves, after):
    a0, b0 = graphs(before)
    a1, b1 = graphs(after)
    if a0.keys() != a1.keys():
        raise ValueError('atom inventory changed during FLOW')
    db = {p:b1.get(p, 0)-b0.get(p, 0) for p in b0.keys() | b1.keys()
          if b1.get(p, 0) != b0.get(p, 0)}
    dq = {i:a1[i].GetFormalCharge()-a0[i].GetFormalCharge() for i in a0
          if a1[i].GetFormalCharge() != a0[i].GetFormalCharge()}
    pred_b, pred_q = declared_deltas(moves)
    involved = {x for m in moves for c in ('source', 'sink') for x in m[c]['atoms']}
    aromatic = any(a0[x].GetIsAromatic() or a1[x].GetIsAromatic() for x in involved)
    residual = {p for p in db.keys() | pred_b.keys() if db.get(p,0) != pred_b.get(p,0)}
    # Ring closure can change aromaticity outside the arrow atoms themselves.
    # Defer ONLY these extra 1.5<->integer representation differences; this is
    # not a chemical failure and is not promoted to a successful replay.
    remote_aromatic_only = bool(residual) and all(
        pred_b.get(p,0) == 0 and 1.5 in (b0.get(p,0),b1.get(p,0))
        and abs(db.get(p,0)) == .5 for p in residual)
    aromatic = aromatic or remote_aromatic_only
    matched = db == pred_b and dq == pred_q
    check = 'matched' if matched else ('aromatic_representation_deferred' if aromatic and dq == pred_q else 'mismatch')
    forward = inverse_moves(moves)
    r = classify_forward(after, forward)
    r.update(transition_check=check, forward_moves=forward,
             net_bond_deltas=[{'atoms':list(k), 'delta':v} for k,v in sorted(db.items())],
             net_charge_deltas=dq, centre_atom_maps=sorted(involved),
             atom_inventory_preserved=True, charge_conserved=sum(dq.values()) == 0)
    if not db and not dq and not pred_b and not pred_q:
        r.update(template='cancelled_arrows_no_transformation', status='template_conflict',
                 flags=['declared coupled arrows cancel without changing any bond or formal charge'])
    elif check == 'mismatch':
        r.update(status='conditions_insufficient', flags=r['flags'] +
                 ['recorded successor disagrees with declared arrow deltas; annotation not certified'])
    elif check == 'aromatic_representation_deferred':
        r.update(status='conditions_insufficient', flags=r['flags'] +
                 ['aromatic/Kekule representation prevents a simple delta confirmation'])
    # Exact state/arrow identity is retained; SMARTS are descriptive fragments,
    # NOT a canonical learned chemical-template identifier.
    local = set(involved)
    for i in involved:
        local.update(n.GetAtomMapNum() for n in a0[i].GetNeighbors())
    r['reaction_centre_radius1_smarts'] = Chem.MolFragmentToSmarts(
        molecule(before), atomsToUse=[a0[i].GetIdx() for i in sorted(local)], isomericSmarts=True)
    return r


def at_pointer(tree, pointer):
    x = tree
    for part in pointer.strip('/').split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        x = x[int(part)] if isinstance(x, list) else x[part]
    return x


def process_tree(task):
    path, programs = task
    from mechet.natural_language_electron_flow import compile_event_arguments
    raw = Path(path).read_bytes()
    tree = json.loads(raw)
    actual_sha = hashlib.sha256(raw).hexdigest()
    if any(r['source']['tree_sha256'] != actual_sha for r in programs):
        raise ValueError('native tree hash differs from frozen source: ' + path)
    rows, links = {}, []
    for r in programs:
        actions = at_pointer(tree, r['program_source_pointer'])
        if digest(actions) != r['program_payload_sha256']:
            raise ValueError('native action hash differs from frozen source')
        event_ids = []
        for index, action in enumerate(actions):
            if action['name'] != 'apply_electron_flow':
                continue
            before, args = action['state_before'], action['arguments']
            eid = digest([r['run_id'], before, args])
            event_ids.append(eid)
            if eid in rows:
                rows[eid]['program_occurrences'] += 1
                rows[eid]['reaction_keys'].add(r['reaction_key'])
                rows[eid]['program_record_ids'].add(r['program_record_id'])
                continue
            after = actions[index+1]['state_before'] if index+1 < len(actions) else None
            base = {'event_id':eid, 'run_id':r['run_id'], 'target_id':r['target_id'],
                    'program_occurrences':1, 'reaction_keys':{r['reaction_key']},
                    'program_record_ids':{r['program_record_id']},
                    'native_source_path':path, 'native_source_sha256':actual_sha,
                    'native_action_pointer':r['program_source_pointer']+'/'+str(index),
                    'state_before':before, 'state_after':after, 'arguments':args,
                    'action_index':index, 'template':'unavailable', 'status':'uncovered'}
            try:
                if args.get('bond_order_changes') or args.get('charge_changes'):
                    base.update(template='BE_DELTA_without_explicit_arrows', flags=['not inferred as a mechanistic template'])
                else:
                    moves = compile_event_arguments(before, args)
                    base['inverse_moves'] = moves
                    base.update(analyze_moves(before, moves, after))
            except Exception as e:
                base.update(annotation_error=type(e).__name__+': '+str(e))
            rows[eid] = base
        links.append({'program_record_id':r['program_record_id'], 'run_id':r['run_id'],
                      'target_id':r['target_id'], 'reaction_key':r['reaction_key'], 'event_ids':event_ids})
    for row in rows.values():
        row['reaction_keys'] = sorted(row['reaction_keys'])
        row['program_record_ids'] = sorted(row['program_record_ids'])
    return list(rows.values()), links, len(raw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    by_tree = defaultdict(list)
    source_path = SOURCE / 'program_reaction_links.jsonl.gz'
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with gzip.open(source_path, 'rt') as f:
        for line in f:
            r = json.loads(line)
            by_tree[r['source']['tree_path']].append(r)
    print(dumps({'phase':'reading_native_trees', 'trees':len(by_tree),
                 'programs':sum(map(len, by_tree.values())), 'workers':a.workers}), flush=True)
    stats = defaultdict(lambda: {'events':0, 'occurrences':0, 'templates':Counter(),
                                'statuses':Counter(), 'transition_checks':Counter()})
    examples, seen_events, programs_n, unique_n, bytes_n = {}, {}, 0, 0, 0
    with gzip.open(a.output/'event_annotations.jsonl.gz', 'wt', encoding='utf-8') as ef, \
         gzip.open(a.output/'program_event_links.jsonl.gz', 'wt', encoding='utf-8') as pf, \
         ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(process_tree, t):t[0] for t in by_tree.items()}
        for n, future in enumerate(as_completed(futures), 1):
            events, links, nbytes = future.result()
            bytes_n += nbytes
            for event in events:
                # A saved event may be shared across trees or terminal contexts.
                # Emit one row per (native tree,event); global summaries use exact
                # event IDs and separately sum program occurrences.
                ef.write(dumps(event)+'\n')
                stat = stats[event['run_id']]
                stat['occurrences'] += event['program_occurrences']
                eid = event['event_id']
                if eid not in seen_events:
                    seen_events[eid] = (event['run_id'], event['status'])
                    unique_n += 1
                    stat['events'] += 1
                    stat['templates'][event['template']] += 1
                    stat['statuses'][event['status']] += 1
                    stat['transition_checks'][event.get('transition_check','unavailable')] += 1
                    exkey = event['run_id']+'|'+event['template']+'|'+event['status']
                    if exkey not in examples or 'Cabozantinib' in event['target_id']:
                        examples[exkey] = event
            for link in links:
                pf.write(dumps(link)+'\n')
            programs_n += len(links)
            if n % 100 == 0 or n == len(futures):
                progress = {'trees_done':n, 'trees_total':len(futures),
                            'percent':round(100*n/len(futures),2), 'programs_done':programs_n,
                            'unique_events':unique_n, 'elapsed_s':round(time.monotonic()-start,1)}
                (a.output/'progress.json').write_text(json.dumps(progress, indent=2))
                print(dumps(progress), flush=True)
    with gzip.open(a.output/'representative_events.jsonl.gz', 'wt', encoding='utf-8') as f:
        for k,r in sorted(examples.items()):
            f.write(dumps(r)+'\n')
    summary = {'schema':'offline_inverse_event_template_analysis_v1',
               'scope':'all existing saved complete-program records in frozen PR89 v18 pool; not a new model experiment',
               'program_records':programs_n, 'native_trees':len(by_tree), 'native_bytes_read':bytes_n,
               'unique_exact_state_arrow_events':unique_n, 'by_run':dict(stats),
               'elapsed_s':round(time.monotonic()-start,2), 'rdkit_version':Chem.rdBase.rdkitVersion,
               'source':str(source_path), 'source_sha256':source_sha, 'references':REFS,
               'deduplication':'exact run_id + private mapped state_before + original arguments; NOT symmetry-invariant chemistry deduplication',
               'physical_mechanism_verified':False,
               'limitations':['authored graph motifs, not experimentally validated microscopic pathways',
                 'compatible means local motif conditions only, not full chemical feasibility',
                 'conflict applies to the assigned template, not all possible mechanisms',
                 'no energy, pKa, solvent, 3D geometry or independent forward-model validation',
                 'general motifs outside this limited library remain uncovered, not chemically false',
                 'current repository alias resolver; recorded-successor delta check flags disagreements',
                 'rejected and incomplete search paths are outside this frozen complete-program population']}
    (a.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(dumps({'complete':True, 'programs':programs_n, 'events':unique_n,
                 'elapsed_s':summary['elapsed_s']}), flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(WORK/'src'))
    main()
