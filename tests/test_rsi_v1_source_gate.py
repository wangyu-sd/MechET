import json
import pytest


@pytest.mark.parametrize('split',['train','val','valid'])
def test_source_gate_accepts_official_validation_alias_but_never_test(tmp_path,split):
    from scripts.audit_rsi_v1 import reference_classes
    from mechet.rsi_contract import sha256
    path=tmp_path/'proof.jsonl';identifier='example'
    path.write_text(json.dumps({'id':identifier,'metadata':{'source_split':split},
        'target_smiles':'[NH4+:1]','structural_precursor':'[NH3:1]'})+'\n')
    refs=reference_classes(path,[identifier],sha256(path))
    assert refs[identifier]['source_split']==('valid' if split=='val' else split)
    row=json.loads(path.read_text());row['metadata']['source_split']='test';path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='TEST forbidden'):reference_classes(path,[identifier],sha256(path))
