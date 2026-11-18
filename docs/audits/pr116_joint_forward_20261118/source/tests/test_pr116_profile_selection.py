from copy import deepcopy

import pytest

from mechet.route_joint_profile import choose_profile


PROTOCOL = {'weights_sha256': 'a'*64, 'figshare_file': 55904912}
SOURCE = 'b'*64


def report(mode, seconds, *, status='PASS'):
    return dict(profile=mode, status=status, source_manifest_sha256=SOURCE,
                protocol=PROTOCOL, train_targets=32,
                hardware=['NVIDIA H20']*4, teacher_queries=882,
                observed_teacher_queries=821, valid_conditional_groups=8,
                complete_step_cost_seconds=seconds)


def test_pick_faster_profile_when_coverage_and_source_are_equal():
    a = report('colocated', 1010.)
    b = report('time_slice', 980.)
    assert choose_profile([a,b], PROTOCOL, SOURCE, 882) == 'time_slice'
    b['observed_teacher_queries'] = 819
    assert choose_profile([a,b], PROTOCOL, SOURCE, 882) == 'colocated'


def test_support_measured_failure_and_reject_mixed_conditions():
    a = report('colocated', 1010., status='FAIL')
    b = report('time_slice', 980.)
    assert choose_profile([a,b], PROTOCOL, SOURCE, 882) == 'time_slice'
    bad = deepcopy(b)
    bad['source_manifest_sha256'] = 'changed'
    with pytest.raises(ValueError, match='common'):
        choose_profile([a,bad], PROTOCOL, SOURCE, 882)
    with pytest.raises(ValueError, match='both'):
        choose_profile([b], PROTOCOL, SOURCE, 882)
