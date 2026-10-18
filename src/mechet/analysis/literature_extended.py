"""Descriptive paired and chemistry-complexity panels, not accuracy estimates."""
from functools import lru_cache
import copy
import math

from .literature_population_figures import prepare_panels, RUNS, LABELS
from .literature_route_metrics import paired_observed_targets, source_supported


@lru_cache(maxsize=50000)
def _molecule_counts(smiles):
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('invalid archived molecule in complexity analysis')
    return molecule.GetNumHeavyAtoms(), molecule.GetRingInfo().NumRings()


def prepare_extended(queries, evidence, metrics, target_cohorts):
    queries = list(queries)
    keys = {q['reaction_key'] for q in queries}
    if len(keys) != len(queries) or keys != set(evidence):
        raise ValueError('extended reaction pool differs from complete evidence pool')
    population = prepare_panels(metrics, target_cohorts)
    axes = {'changed_bonds': ('0', '1-3', '4-6', '7+', 'Unknown'),
            'product_heavy_atoms': ('1-15', '16-30', '31-45', '46+'),
            'ring_count_change': ('decrease', 'unchanged', 'increase')}
    complexity = {axis: {value: {'reactions': 0, 'indexed_candidates': 0,
                                'source_supported': 0} for value in bins}
                  for axis, bins in axes.items()}
    for q in queries:
        heavy, rings = _molecule_counts('.'.join(q['product_components']))
        precursor_rings = sum(_molecule_counts(s)[1] for s in q['structural_components'])
        edits = q.get('changed_bond_count')
        if edits is not None and (type(edits) is not int or edits < 0):
            raise ValueError('invalid changed-bond count')
        bins = {'changed_bonds': 'Unknown' if edits is None else
                ('0' if edits == 0 else '1-3' if edits <= 3 else '4-6' if edits <= 6 else '7+'),
                'product_heavy_atoms': '1-15' if heavy <= 15 else '16-30' if heavy <= 30 else
                    '31-45' if heavy <= 45 else '46+',
                'ring_count_change': 'increase' if rings > precursor_rings else
                    'decrease' if rings < precursor_rings else 'unchanged'}
        record = evidence[q['reaction_key']]
        for axis, value in bins.items():
            cell = complexity[axis][value]
            cell['reactions'] += 1
            cell['indexed_candidates'] += bool(record.get('source_candidates'))
            cell['source_supported'] += source_supported(record)
    method = metrics['runs'][RUNS[0]]['targets']
    paired = {}
    for run in RUNS[1:]:
        baseline = metrics['runs'][run]['targets']; comparisons = {}
        for name in ('StockSolved@B', 'PrecedentCoveredSolved@B'):
            comparisons[name] = paired_observed_targets(target_cohorts,
                [r['target_id'] for r in method if r[name]],
                [r['target_id'] for r in baseline if r[name]],
                method_observed=[r['target_id'] for r in method if r['report_status'] == 'observed'],
                baseline_observed=[r['target_id'] for r in baseline if r['report_status'] == 'observed'],
                bootstrap_draws=2000, seed=17)
        paired[run] = comparisons
    return {'unique_reactions': len(queries), 'complexity': complexity, 'paired': paired,
            'population': population['route_totals'], 'work': None,
            'complexity_unit': 'distinct complete chemical reactions, not source hits or program occurrences',
            'ring_count_definition': 'RDKit ring-basis count, forward product minus structural precursors; not physical ring strain',
            'matched_compute_claim': False, 'chemical_success_rate_established': False}


def attach_work(data, work, *, supported_disconnections):
    """Use an independently authenticated full-run work receipt, not report sums."""
    population = data['population'][RUNS[0]]
    if (any(work.get(k) != population[v] for k, v in (
            ('original_target_denominator', 'denominator'), ('observed_reports', 'observed_reports'),
            ('pending_reports', 'pending_reports'), ('nonzero_stock_connected_targets', 'StockSolved@B'))) or
            work.get('matched_compute_claim') is not False or
            work.get('worker_elapsed_is_global_wall_time') is not False or
            work.get('actual_gpu_active_seconds') is not None or
            work.get('fixed_observed_reports_only_cost_claim') is not False):
        raise ValueError('actual work population or accounting scope differs')
    totals = work['totals']
    fields = ('actual_generated_output_tokens', 'reused_output_tokens_not_charged',
              'unique_scheduled_model_forward_calls', 'cpu_executor_thread_seconds')
    if any(type(totals.get(k)) not in (int, float) or not math.isfinite(totals[k]) or totals[k] < 0 for k in fields):
        raise ValueError('invalid actual work counter')
    if type(supported_disconnections) is not int or supported_disconnections < 0:
        raise ValueError('invalid supported-disconnection count')
    tokens = totals['actual_generated_output_tokens']
    ratio = lambda n: n*1000000/tokens if tokens > 0 else None
    result = copy.deepcopy(data)
    result['work'] = {'actual_generated_output_tokens': tokens,
        'cache_return_tokens_excluded': totals['reused_output_tokens_not_charged'],
        'unique_scheduled_model_forward_calls': totals['unique_scheduled_model_forward_calls'],
        'cpu_executor_thread_seconds': totals['cpu_executor_thread_seconds'],
        'supported_disconnections': supported_disconnections,
        'covered_targets': population['PrecedentCoveredSolved@B'],
        'supported_disconnections_per_million_generated_tokens': ratio(supported_disconnections),
        'covered_targets_per_million_generated_tokens': ratio(population['PrecedentCoveredSolved@B']),
        'actual_gpu_active_seconds': None, 'scope': 'all frozen shard work, including pending targets'}
    return result


def render_extended(data, style):
    from scripts.render_nmi_tree_figures import label, svg_document
    figures = {}
    width = style['format']['full_width_mm']*72/25.4
    def rule(y):
        return f'<path d="M7 {y} H{width-7}" stroke="{style["palette"]["neutral_mid"]}" stroke-width="0.5"/>'
    body = label(7, 14, 'Paired recorded discoveries on the original target population', size=8, weight='bold')
    body += label(7, 29, 'Descriptive comparison; unequal work. Pending targets are not failed chemistry.', size=6)
    y = 48
    for run, pairs in data['paired'].items():
        for outcome, result in pairs.items():
            title = 'Stock AND closure' if outcome == 'StockSolved@B' else 'Every essential reaction A/B'
            body += label(7, y, 'MCTS vs ' + LABELS[run] + ' · ' + title, size=7, weight='bold')
            cols = (7, 100, 182, 272, 358)
            for x, text in zip(cols, ('Population', 'Both', 'MCTS only', 'Greedy only', 'Neither recorded')):
                body += label(x, y+17, text, size=6, weight='bold')
            body += rule(y+22)
            for dy, row, prefix in ((32, result, 'All'), (47, result['observed_both'], 'Observed')):
                values = [prefix + ' n=' + str(row['denominator'])] + [str(row[k]) for k in
                    ('both', 'method_only', 'baseline_only', 'neither')]
                for x, value in zip(cols, values): body += label(x, y+dy, value, size=6)
            ci = result['difference_95ci']
            interval = 'NA' if ci is None else f'[{ci[0]*100:.2f}, {ci[1]*100:.2f}]'
            body += label(7, y+63, f'Full-population difference (percentage points): {result["difference"]*100:.2f}; 95% CI {interval}.', size=6)
            y += 86
    body += label(7, y, '2,000 paired target bootstrap draws; seed 17. CIs are descriptive, not matched-compute effects.', size=6)
    body += label(7, y+14, 'The observed subset is selected by search completion, not a representative full-cohort estimate.', size=6)
    figures['extended_data_paired_route'] = svg_document(body, style, height_pt=y+26)

    body = label(7, 14, 'Structural retrieval coverage across the full generated reaction pool', size=8, weight='bold')
    body += label(7, 29, f'n={data["unique_reactions"]:,} distinct reactions; all runs retained; unequal work.', size=6)
    names = {'changed_bonds': 'a  Mapped net changed bonds',
             'product_heavy_atoms': 'b  Product heavy atoms',
             'ring_count_change': 'c  Forward net ring-count change'}
    y = 48
    for axis, bins in data['complexity'].items():
        body += label(7, y, names[axis], size=7, weight='bold')
        cols = (7, 130, 252, 375)
        for x, text in zip(cols, ('Stratum', 'Reactions', 'Indexed candidate', 'Checked A/B')):
            body += label(x, y+17, text, size=6, weight='bold')
        body += rule(y+22)
        for i, (name, counts) in enumerate(bins.items()):
            values = (name, *[f'{counts[k]:,}' for k in ('reactions', 'indexed_candidates', 'source_supported')])
            for x, value in zip(cols, values): body += label(x, y+33+15*i, value, size=6)
        y += 48+15*len(bins)
    body += label(7, y, 'An indexed candidate is not checked experimental precedent. Unresolved does not mean invalid.', size=6)
    body += label(7, y+14, 'Ring count is an RDKit graph descriptor, not a physical mechanism or ring-strain measurement.', size=6)
    figures['extended_data_source_retrieval_coverage'] = svg_document(body, style, height_pt=y+26)
    if data['work'] is not None:
        work = data['work']
        body = label(7, 14, 'Measured MCTS generation work at the frozen source boundary', size=8, weight='bold')
        body += label(7, 29, 'All shard work, including pending targets; unequal work. No matched-compute gain claimed.', size=6)
        body += label(7, 48, 'Quantity', size=6, weight='bold') + label(360, 48, 'Recorded value', size=6, weight='bold')
        body += rule(54)
        descriptions = [('actual_generated_output_tokens', 'Newly generated output tokens'),
                        ('cache_return_tokens_excluded', 'Cached output tokens (excluded from generation charge)'),
                        ('unique_scheduled_model_forward_calls', 'Unique scheduled model calls'),
                        ('cpu_executor_thread_seconds', 'CPU executor thread seconds (additive across threads)'),
                        ('supported_disconnections', 'Distinct A/B-supported complete reactions'),
                        ('covered_targets', 'Targets with every essential route reaction A/B'),
                        ('supported_disconnections_per_million_generated_tokens', 'Supported reactions per million generated tokens'),
                        ('covered_targets_per_million_generated_tokens', 'Covered targets per million generated tokens'),
                        ('actual_gpu_active_seconds', 'GPU-active seconds')]
        for i, (key, description) in enumerate(descriptions):
            value = work[key]
            number = 'Unknown' if value is None else f'{value:,}' if type(value) is int else f'{value:,.6g}'
            body += label(7, 68+20*i, description, size=6) + label(360, 68+20*i, number, size=6)
        body += label(7, 265, 'No conversion of cache-return tokens or summed worker elapsed time into fictitious GPU time.', size=6)
        body += label(7, 279, 'Source-checking coverage is incomplete; these efficiency values are evidence-discovery lower bounds.', size=6)
        figures['extended_data_work_attribution'] = svg_document(body, style, height_pt=293)
    return figures
