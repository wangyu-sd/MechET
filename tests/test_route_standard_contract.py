import json
import pytest


def test_standard_binding_checks_full_source_without_claiming_clean_split(tmp_path):
    from test_route_standard_data import fixtures,builder
    from mechet.route_standard_contract import dataset_bindings
    from mechet.external_route_data import file_sha
    out=tmp_path/'data';builder()(**fixtures(tmp_path),output=out)
    binding=dataset_bindings(out,file_sha(out/'manifest.json'),expected_counts=(2,1))
    assert binding['train_targets']==2 and binding['validation_targets']==1
    assert binding['standard_data']['known_cross_split_audit']['shared_targets']==1
    assert binding['standard_data']['policy_input']=='target_main_product_only_v1'
    assert str(out/'references.sqlite') in binding['files_sha256']
    with pytest.raises(ValueError,match='DENOMINATOR'):
        dataset_bindings(out,file_sha(out/'manifest.json'),expected_counts=(3,1))


def test_standard_binding_rejects_hidden_censoring_or_changed_payload(tmp_path):
    from test_route_standard_data import fixtures,builder
    from mechet.route_standard_contract import dataset_bindings
    from mechet.external_route_data import file_sha
    out=tmp_path/'data';builder()(**fixtures(tmp_path),output=out)
    original=(out/'manifest.json').read_text();manifest=json.loads(original)
    manifest['source_rows_filtered']=1;(out/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='FILTER'):
        dataset_bindings(out,file_sha(out/'manifest.json'),expected_counts=(2,1))
    (out/'manifest.json').write_text(original)
    (out/'training_protected_identities.json').write_text('["CO"]')
    with pytest.raises(ValueError,match='HASH|CENSOR'):
        dataset_bindings(out,file_sha(out/'manifest.json'),expected_counts=(2,1))
