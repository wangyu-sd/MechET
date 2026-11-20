"""Full-cohort target-only validation and predeclared checkpoint selection.

The actor's sampling identity is independent of its weight SHA. Behavior receipts
still bind the real checkpoint; only the per-request RNG key is held common.
"""
from collections import Counter

from .external_route_rl import TargetOnlyRouteEpisode
from .route_refit import RefitRouteEpisode


class ValidationRouteEpisode(TargetOnlyRouteEpisode):
    def next_decision(self):
        request = super().next_decision()
        if request is not None:
            # A flat decision index remains stable even across reaction resets.
            # Keep /r as the episode/journal delimiter expected by recovery.
            request.request_id = self.uid + '/r0/a' + str(len(self.records))
        return request


class RefitValidationRouteEpisode(RefitRouteEpisode, ValidationRouteEpisode):
    """Share the G4 root request keys and assign separate keys to ReFIT forks."""


def validation_boundaries(schedule, fractions):
    """First *published* round at each declared full-exposure boundary."""
    total = sum(len(chunk) for _, chunk in schedule)
    if not total or not fractions or fractions[0] != 0 or fractions[-1] != 1:
        raise ValueError('INVALID_FULL_VALIDATION_BOUNDARIES')
    if any(not 0 <= f <= 1 for f in fractions) or list(fractions) != sorted(set(fractions)):
        raise ValueError('INVALID_FULL_VALIDATION_BOUNDARIES')
    result = {0: [0.]}
    visited = 0
    pending = list(fractions[1:])
    for published_round, (_, chunk) in enumerate(schedule, 1):
        visited += len(chunk)
        crossed = []
        while pending and visited >= pending[0] * total:
            crossed.append(pending.pop(0))
        if crossed:
            result[published_round] = crossed
    if pending:
        raise ValueError('UNREACHED_FULL_VALIDATION_BOUNDARY')
    return result


def validation_outcomes(rows, groups, candidates, *, max_forks=0):
    """Retain per-root accounting; no token payloads enter model selection."""
    if len(rows) != len(groups):
        raise ValueError('VALIDATION_GROUP_ALIGNMENT')
    outcomes = []
    for row, group in zip(rows, groups):
        if len(group) != candidates or any(r['task_id'] != row['task_id'] for r in group):
            raise ValueError('VALIDATION_CANDIDATE_ALIGNMENT')
        original_count = len(group)
        refit = {}
        if max_forks:
            from .route_acceleration import SKIPPED, skipped_reward
            from .route_action_feedback import semantic_action

            suffixes = [r['refit']['suffix_reports'] for r in group]
            if any(len(branches) > max_forks for branches in suffixes):
                raise ValueError('VALIDATION_REFIT_FORK_BUDGET_EXCEEDED')
            branches = [branch for tree in suffixes for branch in tree]
            if any(branch['task_id'] != row['task_id'] for branch in branches):
                raise ValueError('VALIDATION_REFIT_SUFFIX_ALIGNMENT')
            pruned = [branch for branch in branches if branch['status'] == SKIPPED]
            if any(branch['reward'] is not None or branch['reward_evidence'] != skipped_reward(branch)
                   for branch in pruned):
                raise ValueError('VALIDATION_REFIT_PRUNED_SUFFIX_REWARD')
            receipts = {}
            for report in [*group, *branches]:
                for record in report['records']:
                    key = record['request_id']
                    if receipts.setdefault(key, record) != record:
                        raise ValueError('VALIDATION_REFIT_RECEIPT_CONFLICT')
            accepted_actions = (semantic_action(record.get('raw', ''))
                                for record in receipts.values() if record.get('accepted') is True)
            commits = sum(action is not None and action[1]['name'] == 'finish_trace'
                          for action in accepted_actions)
            refit = dict(refit_suffix_episodes=len(branches), refit_pruned_suffixes=len(pruned),
                         model_decision_requests=len(receipts), completed_reaction_programs=commits)
            group = [*group, *(r for r in branches if r['status'] != SKIPPED)]
        censored = [r['reward'] is None or r['reward_evidence'].get('censored', False) for r in group]
        causes = Counter((r.get('status', 'unknown') + ':' + r.get('error', ''))
                         for r, censor in zip(group, censored) if censor)
        outcome = dict(task_id=row['task_id'],
            verified_closed=any(r['reward_evidence']['verified_stock_closed'] for r in group),
            native_stock_closed=any(r['status'] in ('stock_solved', 'already_stock') for r in group),
            unsupported_closed=any(r['reward_evidence'].get('structural_stock_closed', False)
                                   and not r['reward_evidence']['verified_stock_closed'] for r in group),
            already_stock=any(r['status'] == 'already_stock' for r in group),
            censored=any(censored), censored_episodes=sum(censored),
            censor_causes=dict(causes), sampled_route_episodes=original_count)
        if max_forks:
            outcome.update(refit)
        else:
            outcome.update(model_decision_requests=sum(len(r['records']) for r in group),
                           completed_reaction_programs=sum(len(r['reactions']) for r in group))
        outcomes.append(outcome)
    return outcomes


def summarize_validation(expected_ids, outcomes, candidates):
    """Censored/unobserved roots remain in the fixed success denominator."""
    expected = set(expected_ids)
    if len(expected) != len(expected_ids) or not expected:
        raise ValueError('INVALID_VALIDATION_DENOMINATOR')
    seen = set()
    causes = Counter()
    totals = Counter()
    for row in outcomes:
        if row['task_id'] not in expected or row['task_id'] in seen:
            raise ValueError('UNKNOWN_OR_DUPLICATE_VALIDATION_ROOT')
        seen.add(row['task_id'])
        if row['sampled_route_episodes'] != candidates:
            raise ValueError('INCOMPLETE_VALIDATION_CANDIDATES')
        for key in ('verified_closed', 'native_stock_closed', 'unsupported_closed', 'already_stock',
                    'censored', 'censored_episodes', 'sampled_route_episodes',
                    'model_decision_requests', 'completed_reaction_programs'):
            totals[key] += row[key]
        if 'refit_suffix_episodes' in row:
            totals['refit_suffix_episodes'] += row['refit_suffix_episodes']
            totals['refit_pruned_suffixes'] += row['refit_pruned_suffixes']
        causes.update(row['censor_causes'])
    denominator = len(expected)
    return dict(denominator=denominator, observed_roots=len(seen),
        unobserved_roots=denominator-len(seen), complete=len(seen) == denominator,
        verified_closed_roots=totals['verified_closed'],
        verified_closed_at_4_lower_bound=totals['verified_closed']/denominator,
        native_stock_closed_roots=totals['native_stock_closed'],
        unsupported_closed_roots=totals['unsupported_closed'], already_stock_roots=totals['already_stock'],
        censored_roots=totals['censored'], censored_episodes=totals['censored_episodes'],
        censored_root_fraction=totals['censored']/denominator,
        sampled_route_episodes=totals['sampled_route_episodes'],
        model_decision_requests=totals['model_decision_requests'],
        decision_requests_per_root=totals['model_decision_requests']/denominator,
        completed_reaction_programs=totals['completed_reaction_programs'],
        censor_causes=dict(causes), candidates=candidates, teacher_action_tokens=0,
        **{k: totals[k] for k in ('refit_suffix_episodes', 'refit_pruned_suffixes') if k in totals})


def selection_key(summary):
    if not summary['complete']:
        raise ValueError('INCOMPLETE_VALIDATION_CANNOT_SELECT_CHECKPOINT')
    return (-summary['verified_closed_at_4_lower_bound'], summary['censored_root_fraction'],
            summary['decision_requests_per_root'], summary['checkpoint_round'])
