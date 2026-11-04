"""Hand-derived unit checks for the offline descriptive analysis, not model evaluation."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).with_name('event_template_analysis.py')


def move(sk, sa, tk, ta):
    return {'source': {'kind': sk, 'atoms': sa}, 'sink': {'kind': tk, 'atoms': ta}}


class EventTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert SCRIPT.exists(), 'offline template analyzer is not implemented'
        spec = importlib.util.spec_from_file_location('event_template_analysis', SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_forward_proton_transfer_is_one_coupled_event(self):
        r = self.module.classify_forward('[OH-:1].[H:2][Cl:3]', [
            move('LP', [1], 'BOND', [1, 2]), move('BOND', [2, 3], 'ATOM', [3])])
        self.assertEqual(r['template'], 'proton_transfer')
        self.assertEqual(r['status'], 'template_compatible')

    def test_methyl_sn2_requires_shared_electrophilic_carbon(self):
        r = self.module.classify_forward('[OH-:1].[CH3:2][Br:3]', [
            move('LP', [1], 'BOND', [1, 2]), move('BOND', [2, 3], 'ATOM', [3])])
        self.assertEqual(r['template'], 'concerted_carbon_substitution')
        self.assertEqual(r['status'], 'template_compatible')

    def test_tertiary_carbon_not_certified_as_sn2(self):
        r = self.module.classify_forward('[OH-:1].[C:2]([CH3:4])([CH3:5])([CH3:6])[Br:3]', [
            move('LP', [1], 'BOND', [1, 2]), move('BOND', [2, 3], 'ATOM', [3])])
        self.assertEqual(r['status'], 'template_conflict')
        self.assertIn('tertiary', ' '.join(r['flags']))

    def test_inverse_addition_is_classified_as_forward_elimination(self):
        r = self.module.analyze_moves('[CH2:1]=[O:2].[Cl-:3]', [
            move('LP', [3], 'BOND', [1, 3]), move('BOND', [1, 2], 'ATOM', [2])],
            '[CH2:1]([O-:2])[Cl:3]')
        self.assertEqual(r['template'], 'carbonyl_elimination')
        self.assertEqual(r['transition_check'], 'matched')

    def test_inverse_elimination_is_forward_carbonyl_addition(self):
        r = self.module.analyze_moves('[CH2:1]([O-:2])[Cl:3]', [
            move('LP', [2], 'BOND', [1, 2]), move('BOND', [1, 3], 'ATOM', [3])],
            '[CH2:1]=[O:2].[Cl-:3]')
        self.assertEqual(r['template'], 'carbonyl_addition')

    def test_disconnected_arrows_do_not_match_proton_transfer(self):
        r = self.module.classify_forward('[OH-:1].[H:2][Cl:3].[H:4][Br:5]', [
            move('LP', [1], 'BOND', [1, 2]), move('BOND', [4, 5], 'ATOM', [5])])
        self.assertEqual(r['status'], 'uncovered')

    def test_arrow_cancellation_is_not_a_chemical_transformation(self):
        r = self.module.analyze_moves('[CH3:1][Cl:2]', [
            move('BOND', [1, 2], 'ATOM', [2]), move('LP', [2], 'BOND', [1, 2])],
            '[CH3:1][Cl:2]')
        self.assertEqual(r['template'], 'cancelled_arrows_no_transformation')
        self.assertEqual(r['status'], 'template_conflict')

    def test_recorded_charge_disagreement_is_not_template_support(self):
        r = self.module.analyze_moves('[CH3:1][Cl:2]', [move('BOND', [1, 2], 'ATOM', [2])],
            '[CH3:1][Cl:2]')
        self.assertEqual(r['transition_check'], 'mismatch')
        self.assertEqual(r['status'], 'conditions_insufficient')

    def test_unactivated_hydroxide_leaver_is_not_certified(self):
        r = self.module.classify_forward('[NH2-:1].[CH3:2][OH:3]', [
            move('LP', [1], 'BOND', [1, 2]), move('BOND', [2, 3], 'ATOM', [3])])
        self.assertEqual(r['status'], 'conditions_insufficient')

    def test_proton_transfer_class_is_invariant_to_private_map_renumbering(self):
        r = self.module.classify_forward('[OH-:83].[H:12][Cl:41]', [
            move('LP', [83], 'BOND', [83, 12]), move('BOND', [12, 41], 'ATOM', [41])])
        self.assertEqual(r['template'], 'proton_transfer')

    def test_no_claim_that_aromatic_substitution_is_globally_impossible(self):
        r = self.module.classify_forward('[OH-:1].[c:2]1([Cl:3])[cH:4][cH:5][cH:6][cH:7][cH:8]1', [
            move('LP', [1], 'BOND', [1, 2]), move('BOND', [2, 3], 'ATOM', [3])])
        self.assertEqual(r['status'], 'template_conflict')
        self.assertIn('alternative mechanisms not ruled out', ' '.join(r['flags']))

    def test_ring_closure_reperceiving_remote_aromaticity_is_not_declared_arrow_failure(self):
        # Frozen observed N-C ring closure: the three arrow atoms are not
        # aromatic, but the newly fused ring changes aromatic bond perception.
        before = '[CH3:1][c:2]1[cH:3][c:4]([CH3:5])[n:6]([N+:7](=[C:8]([S:9][CH3:10])[Cl:22])[H:23])[c:20](=[N-:19])[n:21]1'
        after = '[CH3:1][C:2]1=[N:21][C:20]2=[N:19][C:8]([S:9][CH3:10])([Cl:22])[N:7]([H:23])[N:6]2[C:4]([CH3:5])=[CH:3]1'
        r = self.module.analyze_moves(before, [move('BOND', [7,8], 'ATOM', [7]),
                                               move('LP', [19], 'BOND', [8,19])], after)
        self.assertEqual(r['transition_check'], 'aromatic_representation_deferred')


if __name__ == '__main__':
    unittest.main()
