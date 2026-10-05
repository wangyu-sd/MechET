from scripts.eval_system_one_context_retrieval import ContextRow, evaluate, rank_batches


def test_rank_batches_keeps_one_best_reaction_per_context_batch():
    ranked = rank_batches(
        [0.3, 0.8, 0.8],
        [("[Cl-]",), ("[Br-]",), ("[Cl-]",)],
    )
    assert ranked == [(("[Br-]",), 0.8, 1), (("[Cl-]",), 0.8, 2)]


def test_context_retrieval_reads_only_training_products_for_proposals():
    train = [
        ContextRow("1", "CCO", ("[Cl-]",)),
        ContextRow("2", "CCN", ("[Br-]",)),
    ]
    heldout = [ContextRow("3", "CCO", ("[Cl-]",))]
    report, cases = evaluate(train, heldout)
    assert report["evaluated"] == report["top1_exact"] == 1
    assert report["reference_batch_in_train_support"] == 1
    assert cases[0]["predicted_context_batch"] == ["[Cl-]"]
