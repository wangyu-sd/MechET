import unittest
import json
from pathlib import Path
import tempfile
import warnings
import gc

from run_author_audit import reconcile_predictions
from author_template_match import Event, load_catalog


class ReconciliationTests(unittest.TestCase):
    def test_shared_events_keep_every_program_and_source(self):
        common = dict(event_id='e', run_id='run', state_before='b', state_after='a',
                      inverse_moves=[], arguments={}, reaction_keys=['r'])
        a = dict(common, program_occurrences=2, program_record_ids=['p1', 'p2'],
                 native_source_path='tree1', native_source_sha256='sha1', native_action_pointer='/a')
        b = dict(common, program_occurrences=1, program_record_ids=['p3'],
                 native_source_path='tree2', native_source_sha256='sha2', native_action_pointer='/b')
        rows = reconcile_predictions([a,b])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['program_occurrences'], 3)
        self.assertEqual(rows[0]['program_record_ids'], ['p1', 'p2', 'p3'])
        self.assertEqual(len(rows[0]['native_sources']), 2)

    def test_conflicting_successors_are_not_silently_merged(self):
        a = dict(event_id='e', run_id='run', state_before='b', state_after='a')
        b = dict(a, state_after='different')
        with self.assertRaises(ValueError):
            reconcile_predictions([a,b])


class AuthorExpansionTests(unittest.TestCase):
    def test_author_pka_threshold_keeps_hydroxide_but_excludes_weaker_amine(self):
        # Changing >= to <= or ignoring pKa would add the amine rule and fail.
        row = dict(reaction_smarts='[C:2]-[O;H1;+0:1]>>[C:2]-[O;H0;-1:1]',
                   pKa_constraint={'B':14}, reaction_classes=['fixture'], mechanism='fixture',
                   stage=0, template_index=0, description='deprotonation',
                   required_reagent_smarts=[], excluded_reagent_smarts=None)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = root/'fixture_index.jsonl'
            index.write_text(json.dumps(row)+'\n')
            acid = root/'acid.py'
            acid.write_text("Acid_base = [\n"
                "{'Acid':'[O;H2;+0:100]','Base':'[O;H1;-1:100]','pKa':14,'role':['B']},\n"
                "{'Acid':'[N;H1;+1:100]','Base':'[N;H0;+0:100]','pKa':10.8,'role':['B']}\n]\n")
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always', ResourceWarning)
                groups, errors, _ = load_catalog([index], acid)
                gc.collect()
            self.assertEqual([x for x in notices if issubclass(x.category, ResourceWarning)], [])
            self.assertEqual(len(errors), 0)
            self.assertEqual(sum(map(len,groups.values())), 1)
            event = Event('[CH3:7][O-:8].[OH:9][H:10]', '[CH3:7][O:8][H:10].[OH-:9]')
            matches = [r for r in event.candidate_rules(groups) if event.match(r)]
            self.assertEqual(len(matches), 1)
