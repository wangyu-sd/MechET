import subprocess

import pytest

from submit import start_once


def test_fresh_start_publishes_acceptance_after_one_request(tmp_path):
    queued = {'instance': 'c'*32, 'platform_state': 'PENDING'}
    observations = iter([{'instance': None}, queued, queued])
    count = 0

    def issue():
        nonlocal count
        count += 1

    assert start_once(tmp_path, lambda: next(observations), issue) == queued
    assert (tmp_path/'start.accepted').exists()
    assert start_once(tmp_path, lambda: next(observations), issue) == queued
    assert count == 1


def test_adopts_an_instance_left_by_a_previous_client(tmp_path):
    live = {'instance': 'a'*32, 'platform_state': 'TRAINING_RUNNING'}

    def duplicate():
        pytest.fail('launch would duplicate an existing instance')

    assert start_once(tmp_path, lambda: live, duplicate) == live
    assert (tmp_path/'start.accepted').exists()


def test_a_lost_acknowledgement_waits_then_adopts_the_same_instance(tmp_path):
    absent = {'instance': None, 'platform_state': 'REGISTERED_WITHOUT_INSTANCE'}
    live = {'instance': 'b'*32, 'platform_state': 'RESOURCE_WAIT_TRANSITION'}
    observations = iter([absent, absent, absent, live])

    def lost_ack():
        raise subprocess.TimeoutExpired('taiji start', 55)

    with pytest.raises(RuntimeError, match='unconfirmed'):
        start_once(tmp_path, lambda: next(observations), lost_ack)
    assert (tmp_path/'start.requested').exists()
    assert not (tmp_path/'start.accepted').exists()

    def duplicate():
        pytest.fail('unconfirmed start may become visible on Taiji later')

    with pytest.raises(RuntimeError, match='unconfirmed'):
        start_once(tmp_path, lambda: next(observations), duplicate)
    assert start_once(tmp_path, lambda: next(observations), duplicate) == live
    assert (tmp_path/'start.accepted').exists()
