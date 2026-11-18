from hashlib import sha256
import json

import pytest

from check_forward_model import verify_full_batch_receipt


def test_full_batch_receipt_binds_query_bytes_and_manifest(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(source_round=0, train_targets=32,
        distinct_teacher_queries=2, queries_sha256='frozen-query-bytes')))
    receipt = tmp_path/'forward_full_batch_execution.json'
    protocol = {'figshare_file': 55904912, 'weights_sha256': 'verified-weights'}
    data = dict(protocol=protocol, scores=[.25, None], observed_scores=1, unavailable_scores=1,
        device={'devices': 1, 'gpu': 'NVIDIA A100'}, elapsed_seconds=4.0, train_targets=32,
        train_batch_manifest_sha256=sha256(manifest.read_bytes()).hexdigest(),
        real_input_sha256='frozen-query-bytes')
    receipt.write_text(json.dumps(data))
    assert verify_full_batch_receipt(receipt, protocol, manifest)['observed_scores'] == 1
    for field, changed in [('real_input_sha256', 'other-inputs'), ('scores', [None, None]),
                           ('train_batch_manifest_sha256', 'unbound-manifest'), ('device', {'devices': 4})]:
        receipt.write_text(json.dumps({**data, field: changed}))
        with pytest.raises(ValueError):
            verify_full_batch_receipt(receipt, protocol, manifest)
