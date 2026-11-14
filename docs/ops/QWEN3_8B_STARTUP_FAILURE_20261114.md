# Corrected8B startup failure and formal retry

At2026-11-14 14:34CST the original eight-H20 task was found END/unsuccessful, without a training checkpoint or evidence of optimizer updates. It waited from2026-11-13 21:19 until2026-11-14 12:11, then exited12:15. The launcher log directly shows the public redaction placeholder was executed as init_cmd: shell syntax error, initCmd failed, shared script unavailable, user exit127. Model loading never started. This was a submission configuration error, not a checkpoint-quality or chemistry failure.

## Repairs and evidence

- Recover real private Ceph init from a successfully completed donor through the platform; retain command only in0600 shared-storage private config. Published task configs remain explicitly non-submittable templates.
- Eight-H20 submission validation rejects public placeholders, empty/non-mount init and shell syntax errors. Syntax checking uses stdin and sanitizes subprocess failures, so exception arguments/logs do not disclose private command text. These guards do not independently prove mounting will succeed.
- Fix a separately reproduced efficiency defect: verify_files stored canonical resolved paths but receipt reuse compared original symlink aliases. The existing15-file receipt had unchanged digest/stat fields yet never matched. Preserve existing receipt key; compare resolved path sets, expected SHA and complete stat triples/cache flag. Retargeted aliases/digest conflicts cannot reuse old evidence.
- Real full-cache receipt reuse measured0.7749s, then formal retry preflight0.4080s; this is input verification timing, not training throughput. No dataset/weight/sampling/reward changes or new training subset.
- Initial five placeholder/syntax tests and the symlink reuse test failed as expected; fixes passed32 affected regressions17.38s. Independent review found timeout secret exposure; stdin+sanitized exception fix has a reproducing regression. Two additional monitor tests pass4.86s; counts describe different scopes, not new chemistry results.

Source fix8d07a41199638f6728ccf484dd07b67db6e262ce; frozen436-source snapshot a58c66de183451231bec9eb76bea2b0c5df26059cfa0a4d8864bbaaa0ed74454. [Integration patch](../../patches/qwen3_8b_runtime_startup_20261114.patch) targets the preceding c6d3fb3 runtime.

## Formal retry

Taskmeteor_mechet_qwen3_8b_corrected_continue_8h20_zjk_20261114_02, instance8b1d89c6a1cfaaaf01a1d8d733e10ffc, created14:55:45CST and platformTRAINING_RUNNING14:56. Same old8B parent SHA a0f67c8ab799e5dad18ba97d824a7d4ae2a0280083eeeaca75c48ec903028c9b, full1908035TRAIN/21355VALID cache,8H20/global64/oneepoch. Original failed instance is END; no competing MechET campaign launched. Model/optimizer progress remains unconfirmed until real logs/checkpoints show it.

A read-only host monitor now persists actual platform state and saved trainer_state progress at ROOToutputs/submissions/qwen3_8b_rsi_8h20_20261114_retry02/health.json. It retains potentially sensitive launcher logs privately, does not invent training progress from schedulerRUNNING, and never automatically creates duplicate jobs. [Retry receipt](../../configs/ops/qwen3_8b_retry02_20261114.json).

G1–G6 remain pending; this startup repair is not progress in transferable chemical memory or a learned RSI system. PR105 stays Draft and stacked onPR104.


## Retry02 parser failure and retry03

Retry02 mounted Ceph and launched all8 training ranks, but strict config validation stopped before loading weights: `QWEN3_CONTINUATION_TRAINING_CHANGED:learning_rate`. The generic trainer read a JSON configuration through PyYAML1.1: JSON5e-06 became a string. Local reproduction confirmed JSON float vs YAML string. The validator was retained; the reader now uses json.loads for.json and the original YAML loader for.yaml. Actual full-condition JSON round-trip, legacy YAML, and malformed JSON regressions reproduce/catch this.35affected testsPASS40.51s, plus actual full-config CPU entry `--dry-run` passes1908035TRAIN/21355VALID, oneepoch/maxsteps-1. This is a static preflight, not an experiment or a model smoke.

Retry02 is END and has no saved training checkpoint. Retry03 was created/started15:09CST: `meteor_mechet_qwen3_8b_corrected_continue_8h20_zjk_20261114_03`, same8H20/parent/data/cache/LR5e-6/global64. Frozen437files sourcec43bba1ad1ba89a4c518ffb8a1d653cfbb53bae5, snapshotbe33727cc11b13098217c71d7be484e1fc40e066c57393fadbca9a3fe037ed59. Full-cache receipt reused in0.8328s. [JSON entry patch](../../patches/qwen3_8b_json_entry_20261114.patch), [current retry receipt](../../configs/ops/qwen3_8b_retry03_20261114.json). Post-submission monitor is active forretry03; no optimizer update/throughput claimed until real model logs or trainer_state establish it. G1–G6 remain pending.


## Retry03 dependency failure (15:21 CST correction)

Fresh launcher logs confirm all eight ranks exited at15:14:19 with `importlib.metadata.PackageNotFoundError: No package metadata was found for bitsandbytes`. Platform RUNNING was delayed relative to process exit. The failed task stop command succeeded15:21CST. No optimizer update is confirmed and training throughput has not been measured. No other project task was changed.

The bootstrap forwards only the image site-packages under `python -S`; it incorrectly assumed the image already supplied all configured NF4/PEFT/Liger training dependencies. Existing hash-named bitsandbytes and Liger wheels on shared storage provide candidate deployment artifacts, but compatibility and completeness still require validation. Before another formal submission: audit all configured package imports/versions, prepare a hash-bound shared-storage overlay, and run fail-fast dependency checks before eight-rank launch. Do not change the parent, denominator, training objective or batch configuration. G1–G6 remain pending.
