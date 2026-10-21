import json
from pathlib import Path
import pytest


@pytest.mark.parametrize('target,precursor,kind',[
    ('[CH3:1][O:2][CH3:3]','[CH3:1][OH:2].[CH3+:3]','bond_change'),
    ('[CH3:1][OH:2]','[CH2:1]=[O:2]','functional_group_or_redox'),
    ('[NH4+:1]','[NH3:1]','acid_base'),
    ('[C@H:1]([F:2])([Cl:3])[Br:4]','[C@@H:1]([F:2])([Cl:3])[Br:4]','stereochemical'),
    ('[CH3:1][OH:2]','[CH3:1][OH:2]','identity'),
    ('CO','C=O','unknown'),
    ('[CH3:1][OH:2]','[CH4:1]','unknown'),
])
def test_net_class_tracks_actual_maps_and_preserves_nonconnectivity_chemistry(target,precursor,kind):
    from mechet.rsi_productivity import net_reaction
    assert net_reaction(target,precursor)['kind']==kind


def test_actual_phosphate_and_noxide_positive_shortcuts_fail_their_reference_classes():
    from mechet.rsi_productivity import net_reaction,productivity
    fixtures=json.loads((Path(__file__).parent/'fixtures/pr92_shortcuts.json').read_text())
    assert set(fixtures)=={'flower_mech_proof_train_214491','flower_mech_proof_train_187098'}
    for source,r in fixtures.items():
        reference=net_reaction(r['reference_target'],r['reference_precursor'])
        assert reference['kind']=='bond_change',source
        episode={'initial_state':r['target_private'],'final_state':r['candidate_private'],
                 'terminal':True,'flow_count':1}
        label=productivity(episode,reference)
        assert label['candidate']['kind']=='acid_base',source
        assert label['task_compatible_net_progress'] is False,source


def test_real_acid_base_task_and_proton_transfer_intermediate_are_not_globally_banned():
    from mechet.rsi_productivity import productivity
    e={'initial_state':'[NH4+:1]','final_state':'[NH3:1]','terminal':True,'flow_count':1}
    assert productivity(e,{'kind':'acid_base'})['task_compatible_net_progress'] is True
    e={'initial_state':'[CH3:1][O:2][CH3:3]','final_state':'[CH3:1][OH:2].[CH3+:3]',
       'terminal':True,'flow_count':2,'steps':[{'accepted':True,'state_after':'[CH3:1][OH+:2][CH3:3]'}]}
    assert productivity(e,{'kind':'bond_change'})['task_compatible_net_progress'] is True


def test_unknown_reference_never_manufactures_a_strong_label():
    from mechet.rsi_productivity import productivity
    e={'initial_state':'[NH4+:1]','final_state':'[NH3:1]','terminal':True,'flow_count':1}
    assert productivity(e,{'kind':'unknown'})['task_compatible_net_progress'] is None
