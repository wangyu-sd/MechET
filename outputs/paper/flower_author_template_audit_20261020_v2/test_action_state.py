import unittest
from rdkit import Chem

from action_state import before_action


class RecordedImportStateTests(unittest.TestCase):
    def test_missing_recorded_h_atoms_are_added_without_inventing_h2(self):
        s=before_action('[CH4:1]',['[H:45]','[H:88]'])
        p=Chem.SmilesParserParams();p.removeHs=False
        m=Chem.MolFromSmiles(s,p)
        atoms={a.GetAtomMapNum():a for a in m.GetAtoms()}
        self.assertEqual(set(atoms),{1,45,88})
        self.assertEqual(atoms[45].GetFormalCharge(),0)
        self.assertEqual(atoms[45].GetNumRadicalElectrons(),1)
        self.assertIsNone(m.GetBondBetweenAtoms(atoms[45].GetIdx(),atoms[88].GetIdx()))

    def test_already_present_atom_is_not_imported_twice(self):
        self.assertEqual(before_action('[CH4:1].[H:45]',['[H:45]']),'[CH4:1].[H:45]')

    def test_partial_map_collision_is_rejected(self):
        with self.assertRaises(ValueError):
            before_action('[CH4:1]',['[CH3:1][H:2]'])
