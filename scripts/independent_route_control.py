"""Independent complete-route policy episodes, with no shared search frontier.

This is a prepared baseline, not a report of a model experiment. It reuses
the existing private-state executor, stock-AND semantics and Stage-II prompt.
"""
from __future__ import annotations

import hashlib
import json
import math
import gzip
from collections import Counter
from pathlib import Path

from mechet.dynamic_electron_tree import Proposal
from mechet.reverse_route_network import RouteNetwork
from scripts.build_natural_language_event_sft import SYSTEM
from scripts.run_natural_language_value_search import policy_prompt


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class IndependentRouteEpisode:
    """One n=1 stochastic path at each reaction, recursively closing all leaves.

    Distinct episodes own separate networks/caches and never observe another
    episode's successes, frontier or N/W statistics. Deterministic stock tests
    are shared read-only. An invalid action ends the attempt, not a retry tree.
    """
    def __init__(self, row, stock, *, episode_index, seed):
        if type(episode_index) is not int or episode_index < 0 or type(seed) is not int or seed < 0:
            raise ValueError('invalid independent episode index/seed')
        self.target_id = row['id']
        self.episode_index, self.seed = episode_index, seed
        self.network = RouteNetwork(self.target_id, row['product_unmapped'], stock,
                                    mode='greedy', shared_store=None)
        self.uid = fingerprint(['independent_complete_route_v1', self.target_id,
                                self.network.target, episode_index, seed])
        self.draws = 0
        self.final_status = ''
        self.pending = None

    def next_request(self):
        if self.final_status:
            return None
        if self.pending is not None:
            return self.pending
        if self.network.already_in_stock:
            self.final_status = 'already_stock'
            return None
        if self.network.target in self.network._closure():
            self.final_status = 'stock_solved'
            return None
        context = self.network.next_context()
        if context is None:
            self.final_status = 'sampled_path_exhausted'
            return None
        node = context.node
        self.pending = {'episode_uid': self.uid, 'context_key': context.key,
            'seed': int(fingerprint([self.uid, self.draws, context.key])[:16], 16) % (2**63),
            'completion_budget': context.max_new_tokens,
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user',
                'content': policy_prompt(node.target, node.state, include_inventory=True,
                                         actions=node.actions, compact_history=True)}]}
        return self.pending

    def censor(self, reason):
        if self.pending is not None:
            self.network.contexts[self.pending['context_key']].resource_closed = reason
        self.network.stop_reason = reason
        self.final_status = 'resource_censored'
        self.pending = None

    def accept(self, context_key, proposal: Proposal, *, completion_room, max_tokens):
        if self.final_status or self.pending is None or context_key != self.pending['context_key']:
            raise ValueError('independent episode has no matching pending decision')
        if (type(completion_room) is not int or type(max_tokens) is not int or
                not 0 < max_tokens <= completion_room or
                max_tokens != min(self.pending['completion_budget'], completion_room) or
                type(proposal.tokens) is not int or not 0 <= proposal.tokens <= max_tokens or
                not math.isfinite(proposal.logprob)):
            raise ValueError('invalid independent generation receipt')
        self.draws += 1
        edges = self.network.incorporate(context_key, [proposal])
        if proposal.truncated:
            # Identical 384->768->... room adaptation to the archived actor
            # runner. Every truncated draw remains charged and journalled.
            if max_tokens >= completion_room:
                self.censor('COMPLETION_CONTEXT_BUDGET')
            else:
                self.pending = None
        elif not any(edge.get('child_context') for edge in edges):
            reason = edges[0].get('error', 'INVALID_ACTION')
            if any(word in reason.lower() for word in ('memory', 'resource', 'allocation', 'timeout')):
                self.censor(reason)
            else:
                self.network.stop_reason = reason
                self.final_status = 'rejected_action'
                self.pending = None
        else:
            self.pending = None
        return edges

    def report(self):
        return {**self.network.report(), 'id': self.target_id,
                'episode_index': self.episode_index, 'episode_uid': self.uid,
                'draws': self.draws, 'final_status': self.final_status,
                'search_complete': bool(self.final_status and self.final_status != 'resource_censored'),
                'independent_episode': True, 'shared_frontier': False,
                'matched_compute_claim': False, 'reference_visible_to_policy': False}


def prepare_plan(freeze_root, assets_path, adapter, output_root, *,
                 expected_actor_sha='40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75'):
    """Prepare the full control population and provenance; never load a model.

    Metadata's old adapter hash is reported, not trusted over actual weights.
    Every source is validated before creating the new destination directory.
    """
    from mechet.analysis.literature_sources import file_sha256
    from scripts.run_natural_language_value_search import validate_v2_adapter_manifest
    freeze_root, assets_path, adapter, output = map(Path, (freeze_root, assets_path, adapter, output_root))
    if output.exists():
        raise FileExistsError(output)
    bound = {}

    def bind(path, expected=None):
        path = path.resolve()
        actual = file_sha256(path)
        if expected is not None and actual != expected:
            raise ValueError('independent plan source checksum differs: '+str(path))
        bound[str(path)] = actual
        return path

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_bytes())

    freeze = read(freeze_root/'prediction_freeze_manifest.json')
    if freeze.get('complete') is not True or freeze['actor_sha256'] != expected_actor_sha:
        raise ValueError('independent plan requires the frozen completed actor analysis')
    for name, value in freeze['output_files'].items():
        if Path(name).name != name:
            raise ValueError('invalid frozen artifact path')
        bind(freeze_root/name, value['sha256'])
    assets = read(assets_path, freeze['route_assets_manifest_sha256'])
    if (assets.get('stock_preparation_complete') is not True or assets['full_target_rows'] != 460 or
            assets.get('normalization') != 'mapless_canonical_stereo_isotope_charge_preserved_whole_salts_v1'):
        raise ValueError('full original stock/target contract required')
    targets_path = bind(Path(assets['full_targets_path']), assets['full_targets_sha256'])
    targets = [json.loads(line) for line in targets_path.read_text().splitlines()]
    cohorts = {'ursa_expert100': 100, 'ursa_drugs100': 100,
               'retrocast_mkt_cnv160': 160, 'paroutes_derived_n5_100': 100}
    if (len(targets) != 460 or len({r['id'] for r in targets}) != 460 or
            Counter(r['cohort'] for r in targets) != cohorts):
        raise ValueError('full original460 target universe/cohorts differ')
    for cohort, count in cohorts.items():
        if assets['cohorts'][cohort]['rows'] != count:
            raise ValueError('cohort cardinality differs')
        if 'target_ids' in assets['cohorts'][cohort]:
            if set(assets['cohorts'][cohort]['target_ids']) != {r['id'] for r in targets if r['cohort'] == cohort}:
                raise ValueError('cohort target IDs differ')
    for stock in assets['stocks'].values():
        stock_path = bind(Path(stock['canonical_path']), stock['canonical_sha256'])
        stock_rows = stock_path.read_text().splitlines()
        if len(stock_rows) != stock['normalized_rows'] or len(set(stock_rows)) != len(stock_rows):
            raise ValueError('canonical stock cardinality differs')
    with gzip.open(freeze_root/'target_population.jsonl.gz', 'rt') as f:
        population = [r for r in map(json.loads, f) if r['run_id'] == 'r1_mcts_pr88']
    if (len(population) != 460 or len({r['target_id'] for r in population}) != 460 or
            {r['target_id'] for r in population} != {r['id'] for r in targets} or
            Counter(r['report_status'] for r in population) != {'observed': 200, 'pending': 260}):
        raise ValueError('frozen full460/fixed200 paired population differs')
    actor_path = bind(adapter/'adapter_model.safetensors', expected_actor_sha)
    manifest = validate_v2_adapter_manifest(adapter, compact_history=True,
        expected_model='Qwen/Qwen3-0.6B', expected_revision='c1899de289a04d12100db370d81485cdf75e47ca')
    bind(adapter/'adapter_manifest.json')
    contract = read(adapter/'data_contract.json')
    configuration = read(adapter/'adapter_config.json')
    env = 'natural_language_electron_event_history_v2'
    if (contract.get('environment_revision') != env or
            manifest.get('tokenizer_revision') != manifest['base_model_revision'] or
            configuration.get('base_model_name_or_path') != 'Qwen/Qwen3-0.6B'):
        raise ValueError('checkpoint/tokenizer/training protocol differs')
    headers = {}
    for source in freeze['source_reports']:
        if source['run_id'] != 'r1_mcts_pr88':
            continue
        path = source['manifest_path']
        if path in headers:
            if headers[path]['sha256'] != source['manifest_sha256']:
                raise ValueError('conflicting frozen actor headers')
            continue
        header = read(Path(path), source['manifest_sha256'])
        checks = {'model': 'Qwen/Qwen3-0.6B', 'revision': manifest['base_model_revision'],
                  'adapter_sha': expected_actor_sha, 'world': 8, 'full_denominator': 460,
                  'history_contract': env, 'max_model_len': 8192}
        if source['actor_sha256'] != expected_actor_sha or any(header.get(k) != v for k, v in checks.items()):
            raise ValueError('original actor runtime protocol differs')
        headers[path] = {'sha256': source['manifest_sha256'], 'rank': header['rank']}
    if len(headers) != 8 or {r['rank'] for r in headers.values()} != set(range(8)):
        raise ValueError('all original eight actor headers required')
    product_only = [{k: r[k] for k in ('id', 'product_unmapped', 'cohort')} for r in targets]
    paired = sorted(r['target_id'] for r in population if r['report_status'] == 'observed')
    result = {'schema': 'independent_complete_route_prepared_v1', 'prepared_only': True,
        'new_model_inference': False, 'full_target_denominator': 460, 'paired_observed_denominator': 200,
        'episode_counts': [1, 4, 10], 'seed': 17, 'actor_weight_path': str(actor_path),
        'actor_weight_sha256': expected_actor_sha, 'original_adapter_manifest_sha_field': manifest.get('adapter_sha256'),
        'original_adapter_manifest_sha_field_matches_weights': manifest.get('adapter_sha256') == expected_actor_sha,
        'source_files': bound, 'original_actor_headers': headers,
        'sampling': {'n': 1, 'temperature': 1, 'top_p': 1, 'top_k': -1,
                     'max_model_len': 8192, 'initial_completion_budget': 384,
                     'truncation': 'double_per_action_until_actual_context_room', 'penalties': 'none'},
        'shared_frontier': False, 'shared_policy_samples': False, 'reference_visible_to_policy': False,
        'stop_after_first_success': False, 'matched_compute_claim': False,
        'model_execution_requires_separate_authorization': True, 'producer_sha256': file_sha256(__file__)}
    output.mkdir(parents=True, exist_ok=False)
    (output/'targets_product_only.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in product_only))
    (output/'paired_observed_target_ids.json').write_text(json.dumps(paired, indent=2)+'\n')
    result['output_files'] = {p.name: {'sha256': file_sha256(p)} for p in sorted(output.iterdir())}
    (output/'independent_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
