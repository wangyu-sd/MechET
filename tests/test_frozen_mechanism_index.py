"""Full-census source reader units; synthetic fixtures are not model results."""
import hashlib
import json
import pytest
from test_frozen_route_mechanisms import decision, flow_args


def source_file(tmp_path):
    rows=[]
    for source in ['matched','unmatched']:
        for index,name,args in [(0,'apply_electron_flow',flow_args()),(1,'finish_trace',{})]:
            row=decision(index,name,args);row.update(id=f'{source}:{index}',source_id=source)
            rows.append(row)
    path=tmp_path/'train.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_exposure_counts_unmatched_reactions_and_keeps_only_exact_witnesses(tmp_path):
    from scripts.index_frozen_route_mechanisms import scan_sft_split
    path,sha=source_file(tmp_path)
    lookup={'matched':{'structural_precursor':'[CH3+].[OH-]',
        'full_precursor_state':'[CH3+].[OH-]','product_unmapped':'CO'}}
    report,records=scan_sft_split(path,sha,expected_decisions=4,expected_source_ids={'matched','unmatched'},
                                 lookup=lookup,split='train',workers=1)
    assert report['reaction_rows']==2
    assert report['decision_rows']==4
    assert len(records)==1
    assert records[0]['source_id']=='matched'
    assert list(report['primitive_counts'].values())==[2]
    assert list(report['program_counts'].values())==[2]


def test_source_byte_change_and_missing_reaction_cannot_seal_an_exposure_index(tmp_path):
    from scripts.index_frozen_route_mechanisms import scan_sft_split
    path,sha=source_file(tmp_path)
    with pytest.raises(ValueError,match='SHA'):
        scan_sft_split(path,'0'*64,expected_decisions=4,expected_source_ids={'matched','unmatched'},
                       lookup={},split='train',workers=1)
    with pytest.raises(ValueError,match='reaction'):
        scan_sft_split(path,sha,expected_decisions=4,expected_source_ids={'matched'},
                       lookup={},split='train',workers=1)


def test_parent_actual_sft_endpoint_drift_cannot_attach_a_recorded_program(tmp_path):
    from scripts.index_frozen_route_mechanisms import scan_sft_split
    path,sha=source_file(tmp_path)
    lookup={'matched':{'structural_precursor':'C.O','full_precursor_state':'C.O','product_unmapped':'CO'}}
    with pytest.raises(ValueError,match='parent.*SFT'):
        scan_sft_split(path,sha,expected_decisions=4,expected_source_ids={'matched','unmatched'},
                       lookup=lookup,split='train',workers=1)
