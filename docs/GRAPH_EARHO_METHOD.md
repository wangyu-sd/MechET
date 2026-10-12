# Graph-EARHO: Execution-Anchored Receding-Horizon Optimization for the Graph Policy

**Candidate main post-training method for MechET code PR #83.**
The previous script "scripts/train_crn_graph_endpoint_rl.py" is a **fixed
last-three-reference-FLOW diagnostic ablation**, NOT the manuscript method.
The main implementation is "scripts/train_crn_graph_earho.py", with
"src/mechet/graph_earho.py" and "src/mechet/graph_endpoint_rl.py".

## Model roles

- Frozen Qwen3-0.6B State-SFT policy: IMPORT / FLOW / FINISH routing,
  missing precursor fragment SMILES and finish decisions.
- Pretrained CRN-JEPA PR18 Graph Transformer and electron FLOW policy:
  source-pointer, compatible sink-pointer, CONTINUE / COMMIT.
- Unchanged MechET executor: electron validation, authoritative state
  transition and terminal precursor extraction.
- ONLY non-encoder graph policy heads receive Graph-EARHO gradients.

## Core algorithm

1. Replay the train-only reference through the verified executor; separately
   roll out the frozen Qwen+Graph actor from a **fresh product-only mapping**.
   Compare **executed chemical successor states** to local references; find
   the first divergence, NOT the first text string difference.
2. Attribute it to Graph FLOW only if the reference and generated tool
   families BOTH equal apply_electron_flow. If Qwen first misroutes,
   hallucinates an import or finishes incorrectly, it is a frozen Qwen
   controller error. Later gold FLOW states may still supply independently
   labelled **conditional reference replay** supervision, but are NOT called
   on-policy correctable graph frontiers.
3. Reset to a replay-verified FLOW state in a short suffix (initial H=2),
   conditioned on the earliest eligible graph divergence. When verified
   productive support is sufficient, increase H and move the reset
   **backwards toward the actual divergence**, rather than randomly
   selecting one of the last three FLOW events forever. Product-start
   reset is eligible only if the relevant first event is FLOW.
4. For each anchor, decode a Top-32 CRN event slate, sample K=8
   score-weighted distinct events, and **actually execute every event**.
   Pool all accepted source/sink programs that reach the same canonical
   chemical successor, including stereochemistry; keep rejected events
   as distinct failures. Only one continuation per accepted successor class
   is required.
5. For every distinct valid successor, perform **gold-free bounded
   lookahead** using frozen Qwen routing / imports and graph FLOW actions.
   Selection uses the model and executed state only. At training time,
   score whether any completed continuation reached the exact reference
   structural precursor endpoint (reward 1). A verified reference next
   successor without endpoint hit earns 0.2; accepted unsupported earns 0;
   executor-rejected earns -0.2. These are evidence-of-search rewards, not
   physical reaction feasibility claims.
6. Compute group-relative credit **over successor classes**, not repeated
   action strings. An all-negative group gets zero actor advantage, with
   verified reference replay SFT as recovery. When all classes are
   supported but some reach full endpoints and others only immediate
   reference successors, prefer the endpoint classes rather than
   discarding that distinction.
7. Update with the **actual differentiable sum** of source, sink and
   CONTINUE/COMMIT decoder log-probabilities. For a successor class C,
   logsumexp over all sampled action sequences yielding C gives its
   sampled probability mass. With class advantage A_C, optimize the
   signed class-level objective plus 0.1 verified reference-event SFT.
   The slate is truncated and selected from a policy-ranked beam,
   so this is an execution-verified **truncated-slate policy-gradient
   surrogate**, not exact PPO, TRPO, or unbiased GRPO.
8. Every 16 train roots, promote H by one if at least four **on-policy graph-correctable** groups show
   nonzero policy contrast and productive support covers at least 10%
   of these eligible on-policy groups; conditional reference resets after
   Qwen-first mistakes NEVER promote H. Maximum H=8 by default. All checks are train-only.
   Continue monitoring product-start behavior, not just oracle-state
   rewards. Save graph weights, optimizer, Python/Torch/CUDA RNG, horizon
   and per-group audits for SHA-locked resume.

The method is **first graph-correctable divergence + state-equivalence
pooling + bounded recoverability + adaptive backward horizon +
success-gated chemical state credit**. Merely using a group reward and
reference SFT does not constitute Graph-EARHO.

## Scientific/implementation boundaries

- Graph-EARHO's current code uses **bounded executable lookahead**, not
  the old EARHO's *separately trained successor critic*. Do not insert a
  nonexistent v_psi into the current manuscript's implementation description.
  A learned value function is a potential controlled extension, not a
  completed main-model component.
- Qwen tool-name mean-NLL routing is a NEW decision rule relative to the
  historical full-tool-call decoder. Attribute this separately with a
  router-only control.
- CRN graph policy predicts FLOW only, not MechET BE_DELTA; unsupported
  chemistry is counted. Aromatic/Kekulé, hydrogen, stereochemical and map
  permutation/executor-version parity must be audited.
- Before claims of Graph-EARHO performance, run a real CUDA update, verify
  source/sink/commit event log-probability against original graph decoder,
  measure class-pooling/correction/horizon ablations, and compare **exact
  same** valid128 product-start IDs. There are NO authenticated endpoint
  results from this trainer yet.

## Training command (only on a host with actual checkpoint files)

~~~bash
python scripts/train_crn_graph_earho.py \
  --data data/flower_inverse_tool_sft_action_delta_v1/train.jsonl \
  --source-sha256 "$(sha256sum data/flower_inverse_tool_sft_action_delta_v1/train.jsonl | cut -d' ' -f1)" \
  --output outputs/agent/crn_graph_earho_train64 \
  --sample-reactions 64 --rl-updates 64 --slate 8 \
  --graph-flow-topk 2 --rollout-beam-width 2 \
  --continuation-horizon 8 --curriculum-round 16 \
  --policy-adapter outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17 \
  --crn-root /path/to/CRN-JEPA \
  --graph-checkpoint /path/to/CRN-JEPA/outputs/auto_research/pr18/full_corpus_pretrain_v1/train_full_8a100_v2/latest.pt
~~~

Use the resulting "graph_earho_policy_head.pt" under the existing hybrid
evaluator's --graph-rl-head option. The example is NOT a submitted job.
Retain original fixed last-three-reference code for a named ablation;
never attribute its results to the Graph-EARHO method.
