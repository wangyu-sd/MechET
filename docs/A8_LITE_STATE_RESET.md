# A8-Lite: state-reset continuation from A7

A8-Lite is a low-cost continuation experiment initialized from the completed
compact-full-state A7 adapter.  It tests one narrow hypothesis: a policy trained
only on pristine complete histories may over-rely on those histories during a
long executable trace, even though the latest executor state is authoritative.

The training artifact contains 40,000 train-only rows:

- 30,000 deterministic state-reset continuations selected from source traces
  with at least 12 tool calls;
- 10,000 disjoint complete expert anchors to limit forgetting.

A state-reset row keeps the target, the latest accepted authoritative executor
observation, and the exact remaining expert message suffix.  It removes the
earlier accepted dialogue from model context.  No chemical action is invented,
and validation/test rows are never used to build continuation supervision.

This first version is **history-robust continuation**, not off-policy recovery:
it does not claim to teach correction from arbitrary erroneous chemical states.
That stronger claim requires A7 train rollouts, independently executed
alternative states, and a frozen reachability/rejoin labeling procedure.

Build the train artifact with:

```bash
python scripts/build_a8_lite_state_reset_sft.py \
  --source-train /aaa/fionafyang/buddy1/whaleywang/MechET/data/flower_inverse_tool_sft_compact_full_state_v1/train.jsonl \
  --validation-file /aaa/fionafyang/buddy1/whaleywang/MechET/data/flower_inverse_tool_sft_compact_full_state_v1/valid.jsonl \
  --test-file /aaa/fionafyang/buddy1/whaleywang/MechET/data/flower_inverse_tool_sft_compact_full_state_v1/test.jsonl \
  --output-dir /aaa/fionafyang/buddy1/whaleywang/MechET/data/flower_a8_lite_state_reset_v1
```

The initial training comparison is A7 versus A8-Lite on all 2,890 validation
reactions, stratified by expert tool-call count.  A test result must use all
28,967 strict executable FlowER test reactions, count missing predictions as
failures, and report both endpoint success and formal execution.
