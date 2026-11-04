import unittest
from audit_transition_repair import coverage


class CoverageTests(unittest.TestCase):
    def test_composition_is_not_reported_as_a_single_template_match(self):
        rows=[dict(population='gt',legacy_v2_versions=[],versions=[],composition_versions=['Reaction_templates']),
              dict(population='gt',legacy_v2_versions=['Reaction_templates'],versions=['Reaction_templates'],composition_versions=[])]
        result=coverage(rows).get('gt',{})
        self.assertEqual(result.get('events'),2)
        self.assertEqual(result['v2:Reaction_templates'],1)
        self.assertEqual(result['single:Reaction_templates'],1)
        self.assertEqual(result['single_or_composed:Reaction_templates'],2)


if __name__=='__main__':unittest.main()
