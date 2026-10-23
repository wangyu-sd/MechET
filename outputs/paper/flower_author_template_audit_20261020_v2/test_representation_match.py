import json
from pathlib import Path
import unittest

from representation_match import match_event

SN2 = '[O;H0;-1:1].[C:2]-[Br:3]>>[O;H0;+0:1]-[C:2].[Br;-1:3]'
SNAR_ELIM = '[*;a;-1:1]:[c;+0:2]([F,Cl,Br,I:3])-[#8;+0:5]>>[*;a;+0:1]:[c;+0:2]-[#8;+0:5].[F,Cl,Br,I;-1:3]'
SNAR_ADD = '[*;a;+0:1]:[c;+0:2]-[F,Cl,Br,I:3].[#8;-1;D1:5][*:6]>>[*;a;-1:1]:[c;+0:2]([F,Cl,Br,I:3])-[#8;H0;+0:5][*:6]'
ETHER_HYDRIDE = '[#6:1]-[#8;H1;+1:2]-[C:5].[#1;-1:100]>>[#6:1]-[#8;H0;+0:2]-[C:5].[#1;H1;+0:100]'
PT = '[O;H1;+0:1].[O;H1;-1:2]>>[O;H0;-1:1].[O;H2;+0:2]'


class RepresentationMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = json.loads((Path(__file__).parent/'regression_fixtures.json').read_text())

    def test_actual_gt9_halide_leaving_rearomatization(self):
        r = self.fixtures['gt:textbook-tool-sft:flower_mech_proof_test_9:0']
        self.assertEqual(len(match_event(r['state_before'],r['state_after'],[SNAR_ELIM])), 1)

    def test_actual_gt9_sigma_complex_addition(self):
        r = self.fixtures['gt:textbook-tool-sft:flower_mech_proof_test_9:1']
        self.assertEqual(len(match_event(r['state_before'],r['state_after'],[SNAR_ADD])), 1)

    def test_actual_hydrogen_to_hydride_keeps_hydrogen_identity(self):
        r = self.fixtures['86fab32321c52b529b79d261463a51f29da55892b0ac11397b2fb29bb3324022']
        self.assertEqual(len(match_event(r['state_before'],r['state_after'],[ETHER_HYDRIDE])), 1)

    def test_original_substitution_still_matches(self):
        self.assertEqual(len(match_event('[CH3:7][O:8][CH3:9].[Br-:10]',
                                        '[CH3:7][O-:8].[CH3:9][Br:10]',[SN2])),1)

    def test_wrong_direction_does_not_match(self):
        self.assertEqual(match_event('[CH3:7][O-:8].[CH3:9][Br:10]',
                                     '[CH3:7][O:8][CH3:9].[Br-:10]',[SN2]),[])

    def test_attached_explicit_hydrogen_transfer(self):
        self.assertEqual(len(match_event('[CH3:7][O-:8].[OH:9][H:10]',
                                        '[CH3:7][O:8][H:10].[OH-:9]',[PT])),1)

    def test_unchanged_spectator_is_preserved(self):
        self.assertEqual(len(match_event('[CH3:7][O:8][CH3:9].[Br-:10].[Na+:11]',
                                        '[CH3:7][O-:8].[CH3:9][Br:10].[Na+:11]',[SN2])),1)

    def test_extra_remote_bond_cleavage_is_rejected(self):
        self.assertEqual(match_event('[CH3:7][O:8][CH3:9].[Br-:10].[CH3:11][Cl:12]',
                                     '[CH3:7][O-:8].[CH3:9][Br:10].[CH3+:11].[Cl-:12]',[SN2]),[])

    def test_actual_snar_with_extra_charge_change_is_rejected(self):
        r = self.fixtures['gt:textbook-tool-sft:flower_mech_proof_test_9:0']
        self.assertEqual(match_event(r['state_before']+'.[Na+:999]',r['state_after']+'.[Na:999]',[SNAR_ELIM]),[])

    def test_hydride_charge_is_not_neutralized(self):
        r = self.fixtures['86fab32321c52b529b79d261463a51f29da55892b0ac11397b2fb29bb3324022']
        self.assertEqual(match_event(r['state_before'],r['state_after'].replace('[H-:16]','[H:16]'),[ETHER_HYDRIDE]),[])

    def test_isotope_transmutation_is_rejected(self):
        self.assertEqual(match_event('[CH3:7][O-:8].[OH:9][2H:10]',
                                     '[CH3:7][O:8][H:10].[OH-:9]',[PT]),[])

    def test_valid_deuterium_transfer_preserves_isotope_and_identity(self):
        self.assertEqual(len(match_event('[CH3:7][O-:8].[OH:9][2H:10]',
                                        '[CH3:7][O:8][2H:10].[OH-:9]',[PT])),1)

    def test_remote_alkene_migration_is_not_aromatic_equivalence(self):
        self.assertEqual(match_event('[CH3:7][O:8][CH3:9].[Br-:10].[CH2:11]=[CH:12][CH2:13][CH3:14]',
                                     '[CH3:7][O-:8].[CH3:9][Br:10].[CH3:11][CH:12]=[CH:13][CH3:14]',[SN2]),[])

    def test_aromatic_and_valid_kekule_inputs_give_same_substitution(self):
        before='[O:7]([c:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1)[CH3:8].[Br-:9]'
        after='[O-:7][C:1]1=[CH:2][CH:3]=[CH:4][CH:5]=[CH:6]1.[CH3:8][Br:9]'
        self.assertEqual(len(match_event(before,after,[SN2])),1)

    def test_atom_identity_swap_cannot_be_hidden_by_canonicalization(self):
        self.assertEqual(match_event('[CH3:7][O:8][CH3:9].[Br-:10].[CH3:11][OH:12]',
                                     '[CH3:7][OH:8].[CH3:9][Br:10].[CH3:11][O-:12]',[SN2]),[])

    def test_unchanged_state_does_not_count_as_an_event(self):
        s='[CH3:7][O:8][CH3:9].[Br-:10]'
        self.assertEqual(match_event(s,s,[SN2]),[])


if __name__=='__main__':
    unittest.main()
