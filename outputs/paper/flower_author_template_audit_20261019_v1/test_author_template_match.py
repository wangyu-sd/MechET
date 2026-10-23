"""Literal chemistry fixtures: inverse direction, private identity and full delta."""
import unittest

from author_template_match import match_event, Event, compile_rule, typed_charge_key

SN2 = '[O;H0;-1:1].[C:2]-[Br:3]>>[O;H0;+0:1]-[C:2].[Br;-1:3]'
PT = '[O;H1;+0:1].[O;H1;-1:2]>>[O;H0;-1:1].[O;H2;+0:2]'


class JointStateMatchTests(unittest.TestCase):
    def test_index_keeps_rules_with_mixed_fixed_and_unspecified_h_counts(self):
        rule = compile_rule('[N;H0;+1:1].[N;+1:2]>>[N;H1;+0:1].[N;H0;+0:2]')
        self.assertEqual(typed_charge_key(rule), ((7, -1, None), (7, -1, 1)))

    def test_radical_change_is_not_certified_by_closed_shell_template(self):
        self.assertEqual(match_event('[CH3:7][O:8][CH3:9].[Br-:10].[CH3:11]',
                                     '[CH3:7][O-:8].[CH3:9][Br:10].[CH4:11]', [SN2]), [])

    def test_inverse_substitution_matches_forward_author_rule(self):
        self.assertEqual(len(match_event('[CH3:7][O:8][CH3:9].[Br-:10]',
                                        '[CH3:7][O-:8].[CH3:9][Br:10]', [SN2])), 1)

    def test_forward_direction_is_not_mistaken_for_inverse(self):
        self.assertEqual(match_event('[CH3:7][O-:8].[CH3:9][Br:10]',
                                     '[CH3:7][O:8][CH3:9].[Br-:10]', [SN2]), [])

    def test_unchanged_substructure_is_not_a_transition_match(self):
        s = '[CH3:7][O:8][CH3:9].[Br-:10]'
        self.assertEqual(match_event(s, s, [SN2]), [])

    def test_spectator_does_not_prevent_match(self):
        self.assertEqual(len(match_event('[CH3:7][O:8][CH3:9].[Br-:10].[Na+:11]',
                                        '[CH3:7][O-:8].[CH3:9][Br:10].[Na+:11]', [SN2])), 1)

    def test_disconnected_template_can_describe_intramolecular_ring_closure(self):
        self.assertEqual(len(match_event('[O:1]1[CH2:2][CH2:3][CH2:4][CH2:5]1.[Br-:6]',
                                        '[O-:1][CH2:2][CH2:3][CH2:4][CH2:5][Br:6]', [SN2])), 1)

    def test_extra_changed_bond_cannot_be_ignored(self):
        self.assertEqual(match_event('[CH3:7][O:8][CH3:9].[Br-:10].[CH3:11][Cl:12]',
                                     '[CH3:7][O-:8].[CH3:9][Br:10].[CH3+:11].[Cl-:12]', [SN2]), [])

    def test_mapped_explicit_proton_normalizes_without_losing_charge(self):
        self.assertEqual(len(match_event('[CH3:7][O-:8].[OH:9][H:10]',
                                        '[CH3:7][O:8][H:10].[OH-:9]', [PT])), 1)

    def test_disconnected_matching_must_share_atom_identity_on_both_sides(self):
        self.assertEqual(match_event('[CH3:7][O:8][CH3:9].[Br-:10].[CH3:11][OH:12]',
                                     '[CH3:7][OH:8].[CH3:9][Br:10].[CH3:11][O-:12]', [SN2]), [])

    def test_changed_isotopic_hydrogen_is_not_silently_removed(self):
        self.assertEqual(match_event('[CH3:7][O-:8].[OH:9][2H:10]',
                                     '[CH3:7][O:8][2H:10].[OH-:9]', [PT]), [])

    def test_whole_context_query_is_required(self):
        amide = '[O:4]=[C:5]-[N;H0;-1:1].[C:2]-[Br:3]>>[O:4]=[C:5]-[N;H0;+0:1]-[C:2].[Br;-1:3]'
        self.assertEqual(match_event('[CH3:7][NH:8][CH3:9].[Br-:10]',
                                     '[CH3:7][NH-:8].[CH3:9][Br:10]', [amide]), [])


if __name__ == '__main__':
    unittest.main()
