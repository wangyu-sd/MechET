"""Real regression fixtures plus independent wrong-transition controls."""
import gzip
import json
from pathlib import Path
import unittest

from transition_repair import compile_rule,match_event,match_composition,legacy
import test_representation_match as previous_tests

# Exercise the previous 16 representation regressions against the NEW entry
# point, rather than merely rerunning the unchanged old implementation.
previous_tests.match_event=match_event


class CarriedForwardRepresentationTests(previous_tests.RepresentationMatchTests):
    pass

PD3='[Pd;D3;+0:1](-[P;H0:6])(-[P;H0:5])-[P;H0:8]>>[Pd;D2;+0:1](-[P;H0:6])(-[P;H0:5]).[P;H0:8]'
PD4='[Pd;D4;+0:1](-[P;H0:6])(-[P;H0:4])(-[P;H0:5])-[P;H0:8]>>[Pd;D3;+0:1](-[P;H0:6])(-[P;H0:4])(-[P;H0:5]).[P;H0:8]'
PT='[#6:3]-[O;H1;+0:1].[O;H1;-1:2]>>[#6:3]-[O;H0;-1:1].[O;H2;+0:2]'
DOUBLE_BEFORE='[CH3:1][O-:2].[CH3:3][O-:4].[OH:5][H:7].[OH:6][H:8]'
DOUBLE_AFTER='[CH3:1][O:2][H:7].[CH3:3][O:4][H:8].[OH-:5].[OH-:6]'


class TransitionRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parent.parent/'flower_author_template_audit_20261019_v1/results_v2/event_inputs.jsonl.gz'
        ids={f'gt:textbook-tool-sft:flower_mech_proof_test_37:{i}' for i in (5,6)}
        cls.real={}
        with gzip.open(path,'rt') as f:
            for line in f:
                r=json.loads(line)
                if r['event_id'] in ids:
                    cls.real[r['action_index']]=r
                if len(cls.real)==2:break
        assert len(cls.real)==2

    def test_actual_pd_third_ligand_is_not_rejected_by_radical_gate(self):
        r=self.real[5]
        self.assertEqual(len(match_event(r['state_before'],r['state_after'],[PD3])),1)

    def test_actual_pd_fourth_ligand_is_not_rejected_by_radical_gate(self):
        r=self.real[6]
        self.assertEqual(len(match_event(r['state_before'],r['state_after'],[PD4])),1)

    def test_true_bond_homolysis_requires_the_exact_rule(self):
        rule='[CH3:1]-[Br:2]>>[CH3:1].[Br:2]'
        self.assertEqual(len(match_event('[CH3:7].[Br:8]','[CH3:7][Br:8]',[rule])),1)

    def test_unrelated_spectator_radical_change_is_not_ignored(self):
        r=self.real[5]
        self.assertEqual(match_event(r['state_before']+'.[CH3:999]',r['state_after']+'.[CH4:999]',[PD3]),[])

    def test_unrelated_charge_change_is_not_hidden_by_pd_repair(self):
        r=self.real[5]
        self.assertEqual(match_event(r['state_before']+'.[Na+:999]',r['state_after']+'.[Na:999]',[PD3]),[])

    def test_two_transfers_have_two_author_rule_witnesses(self):
        proof=match_composition(DOUBLE_BEFORE,DOUBLE_AFTER,[compile_rule(PT)])
        self.assertIsNotNone(proof)
        self.assertEqual(len(proof['steps']),2)

    def test_compound_cannot_hide_unexplained_remote_cleavage(self):
        proof=match_composition(DOUBLE_BEFORE+'.[CH3:20][Cl:21]',
                                DOUBLE_AFTER+'.[CH3+:20].[Cl-:21]',[compile_rule(PT)])
        self.assertIsNone(proof)

    def test_compound_cannot_ignore_wrong_isotope(self):
        proof=match_composition(DOUBLE_BEFORE.replace('[H:7]','[2H:7]'),DOUBLE_AFTER,[compile_rule(PT)])
        self.assertIsNone(proof)

    def test_compound_cannot_reverse_author_direction(self):
        self.assertIsNone(match_composition(DOUBLE_AFTER,DOUBLE_BEFORE,[compile_rule(PT)]))


if __name__=='__main__':unittest.main()
