from copy import deepcopy

import pytest

from mechet.route_joint_offline_dpo import batch_preferences, native_columns


def one_group():
    def report(name, tokens, score):
        return dict(task_id='t', root='CO', status='stock_solved',
                    reward_evidence={'forward_joint': {'decisions': [dict(index=1,
                        occurrence='chem', protocol='teacher', score=score)]}},
                    records=[dict(request_id=name, prompt_token_ids=[1, 3],
                        generated_token_ids=tokens, eos_token_id=9, behavior_version='behavior',
                        occurrence='chem', accepted=True)],
                    refit_path=[dict(node_id='shared', action=name, request_id=name,
                        name='apply_electron_flow', index=0, proposal='original_policy')])

    root = report('a', [7, 9], .75)
    suffix = report('b', [8, 9], .25)
    suffix['continuation_declaration_order'] = 1
    root['refit'] = dict(node_censored_receipts={},
                        conditioning={'shared': dict(prompt_token_ids=[1, 3], behavior_version='behavior')},
                        suffix_reports=[suffix])
    return [root]


def test_batch_preferences_reuse_same_teacher_and_unmodified_native_tokens():
    group = one_group()
    actual = batch_preferences([group], [('t', 'CO')], 1, 'behavior', 'teacher', .25)
    assert actual == [dict(task_id='t', behavior_version='behavior',
                          chosen_request='a', rejected_request='b', eos_token_id=9,
                          prompt_ids=[1, 3], chosen_ids=[7, 9], rejected_ids=[8, 9])]
    assert native_columns(actual[0], eos_id=9, vocab_size=12, max_context=4) == dict(
        prompt_ids=[1, 3], chosen_ids=[7, 9], rejected_ids=[8, 9])
    with pytest.raises(ValueError, match='context'):
        native_columns(actual[0], eos_id=9, vocab_size=12, max_context=3)


def test_batch_preferences_refuse_off_protocol_pairs_and_ignore_missing_scores():
    group = one_group()
    invalid = deepcopy(group)
    invalid[0]['refit']['suffix_reports'][0]['reward_evidence']['forward_joint']['decisions'][0]['protocol']='different'
    with pytest.raises(ValueError, match='teacher'):
        batch_preferences([invalid], [('t', 'CO')], 1, 'behavior', 'teacher', .25)
    with pytest.raises(ValueError, match='TRAIN'):
        batch_preferences([group], [('other', 'CO')], 1, 'behavior', 'teacher', .25)
    group[0]['refit']['suffix_reports'][0]['reward_evidence']['forward_joint']['decisions'][0]['score']=None
    assert batch_preferences([group], [('t', 'CO')], 1, 'behavior', 'teacher', .25) == []


def test_pinned_trl_uses_native_tokens_and_masks_out_the_whole_prompt():
    from datasets import Dataset
    from trl.trainer.dpo_trainer import DataCollatorForPreference
    from scripts.run_pr116_offline_dpo import native_dpo_trainer

    data = Dataset.from_list([{'prompt_ids': [1, 3], 'chosen_ids': [7, 9],
                              'rejected_ids': [8, 4, 9]}])
    trainer = native_dpo_trainer()
    prepared = trainer._prepare_dataset(None, data, None, None, 'train')
    result = DataCollatorForPreference(pad_token_id=0)([prepared[0]])
    live = [ids[mask.bool()].tolist() for ids, mask in zip(result['input_ids'], result['attention_mask'])]
    completion = [ids[mask.bool()].tolist() for ids, mask in zip(result['input_ids'], result['completion_mask'])]
    assert live == [[1, 3, 7, 9], [1, 3, 8, 4, 9]]
    assert completion == [[7, 9], [8, 4, 9]]
