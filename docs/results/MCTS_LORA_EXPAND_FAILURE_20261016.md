# 31k MCTS GPU failure and bounded implementation repair

## Incident

The ordinary 8xA100 task `meteor_mechet_uspto31k_mcts_ray_resume_8a100_qy_20261016_03`,
instance `8b1d8242a13fde4101a1413c9f36031d`, ended unsuccessfully on
2026-10-16 at 10:17 CST. The runner exited at 10:13:52. Worker PID2626
(data shard06; the TP1 stack calls its GPU rank0) reported:

```
punica_gpu.py:add_lora_linear -> add_expand
 -> lora_expand.py:_lora_expand_kernel[grid]
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

A later CUDA/c10 error is also asynchronous. This identifies the reported
failure site, **not a reproduced root cause in the upstream kernel**. There
is no evidence to relabel it as a chemistry failure, an OOM, or a tree-recovery
error. Other workers received TERM and checkpointed resumable work. The shared
Ray store reported WAL sequence401579 committed401579, with an empty queue.

Saved per-shard report counts are233/229/241/206/194/222/182/204: **1711/3120
(54.84%) complete**,1409 remaining. This is completion, not endpoint accuracy.
The 1711 include125 inherited reports. The current output contains101214583
new-continuation generated tokens and1714538185 bytes of raw JSONL. Prior
native journals, checkpoints and cache evidence remain unchanged. A complete
post-crash recovery audit has not yet been performed; do not assert that
every uncommitted in-flight GPU request is recoverable.

## Implementation

`scripts/vllm_safe_lora_expand.py` supplies a **single-resident-adapter**
specialization of the pinned vLLM0.8.5 eager GPU expand operation. It uses
PyTorch GPU matrix multiplication instead of the raw-pointer Triton expand
kernel. It does not merge weights or replace vLLM scheduling with CPU work.

The runner installs it before LLM profiling, explicitly uses max_loras1, and
records the implementation in the startup log, manifest, and each newly
generated raw receipt. The adapter/base revisions, BF16 inference, attention,
LoRA shrink, prompt/history, sampling, PUCT, full3120 min-proxy DIAGNOSTIC
condition, NLL ranking and ten-total-attempts stopping rule remain unchanged.
CPU/GPU metadata checks reject any adapter slot other than0 or no-adapter-1
before metadata preparation. Slice widths/ranks/devices are checked before
any output write; no per-token copies of whole LoRA matrices are constructed.

Expand retains upstream's cast-to-weight-dtype before GEMM and cast-before-add
semantics. Different GEMM reductions need not be bit-identical. Historical
cached samples/reports are retained under the same unmerged-model policy
contract; their provenance is not overwritten with the new runtime label.
This is an explicitly recorded mixed-runtime continuation, **not a claim of
bitwise sample reproducibility or of a changed/improved model**.

This bypasses the reported expand kernel; it does not prove that an earlier
asynchronous kernel could not have caused the original fault. The remaining
Triton shrink/attention path and real A100 BF16 long-running stability still
require deployment verification. No GPU replacement task has been submitted
by this change, and no model smoke/subset experiment was run.

## Verification

Tests were added first. The runner-ordering test failed because the safe
kernel was not installed before GPU initialization. The new numerical module
was initially absent, then implemented. Two fixture tensors incorrectly used
FP32 weights (unsupported by the original ABI); corrected to FP16 rather than
weakening production validation.

Targeted scheduler/runner/kernel tests:22passed. Separately, the exact staged
vLLM0.8.5 CUDA11 runtime successfully loaded the patch. Ten kernel tests passed
with that runtime on the local T4, including the actual packaged Triton-versus-
PyTorch GPU comparison for QKV(2048/1024/1024), gate/up(3072/3072), and single
1024-wide slices; batch sizes1/31/63/64/65/255/256/257; mixed adapter/no-adapter
rows. FP16 GPU comparisons use explicit numerical tolerances, not byte equality.
CPU arithmetic also checks BF16. The packaged-kernel test is parameterized
over FP16/BF16: a native-SM80+ guard explicitly skips BF16 on T4. An initial
guard used PyTorch's emulation-inclusive capability flag, allowing an
unsupported upstream Triton BF16 compile on T4; that test guard was corrected.
Final staged-runtime tests:10passed,1BF16skip,15.24seconds. T4 FP16 is not
production A100 BF16 evidence.

Final whole-project regression run:939passed,13skipped,one existing PEFT config
warning,98.15seconds. `git diff --check` passed. Independent code review found
no Critical/Important issues; the BF16 parameterization and final-test-result
documentation suggestions were addressed. Tests demonstrate arithmetic
and runner integration, not reproduction/elimination of the original long-run
A100 crash.
