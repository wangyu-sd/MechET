import unittest

from run_representation_audit import program_links


class FrozenPopulationLinkTests(unittest.TestCase):
    def test_reconciliation_reads_actual_frozen_program_pool_not_author_output(self):
        rows=list(program_links())
        self.assertEqual(len(rows),37922)
        self.assertEqual(len({r['program_record_id'] for r in rows}),37922)
        self.assertEqual(sum(len(r['event_ids']) for r in rows),124808)
