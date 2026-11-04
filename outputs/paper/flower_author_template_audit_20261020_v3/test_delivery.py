import unittest
from delivery import final_row


class DeliveryTests(unittest.TestCase):
    def test_new_match_does_not_keep_old_radical_rejection_label(self):
        row=dict(status='unsupported_radical_transition',repair_status='radical_change_outside_requested_repair',
                 versions=['Reaction_templates'],composition_versions=[],v3_guarded=[])
        out=final_row(row)
        self.assertEqual(out.get('status'),'author_single_template_compatible_current')
        self.assertEqual(out['legacy_v2_status'],'unsupported_radical_transition')
        self.assertEqual(row['status'],'unsupported_radical_transition')

    def test_composition_and_dated_only_match_keep_distinct_labels(self):
        common=dict(status='no_certified_author_transition',repair_status='no_additional_match',v3_guarded=[])
        composed=final_row(dict(common,versions=[],composition_versions=['Reaction_templates']))
        self.assertEqual(composed.get('status'),'author_composition_compatible_current')
        dated=final_row(dict(common,versions=['Reaction_templates_20250216'],composition_versions=[]))
        self.assertEqual(dated.get('status'),'dated_author_version_only')


if __name__=='__main__':unittest.main()
