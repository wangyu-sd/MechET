# Route-RL learning signal: first ten rounds and failure diagnosis

The current full-protocol run has produced optimizer updates, but there is no
demonstrated improvement in route planning. Across its first ten completed
rounds, only 15 of 640 target groups have nonzero centered advantages. In round
ten, all four ranks report zero policy-gradient loss. Parameter changes in that
round are not evidence of learning from task rewards; KL, optimizer history and
other optimizer terms must be distinguished from a task-reward gradient.

This is a read-only analysis of already completed training work. No subset
training or evaluation was launched for this report. The scope is fixed at
rounds 1--10 even if the live task subsequently advances. The detailed round is
`round-000009`, the zero-based directory for completed round ten.

## Run and evidence

The task is
`meteor_mechet_stage2_external_route_drgrpo_4h20_zjk_20261111_hf_profile_02`.
It uses corrected Stage-II History-89442 directly as actor and fixed reference,
23,730 TRAIN / 2,986 VALID targets, independent G=4, seed 17, one epoch and four
H20 GPUs. The backend is Hugging Face generation with BF16 autocast and learner
batch size one. The scalar reward is unchanged. This is an external multistep
training corpus, not a new FlowER training run or an R1/R2/SimpRetro result.

The parent adapter SHA256 is
`2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453`.
The submitted contract SHA256 is
`fe59b68b9f6cf48520e64f4e7d9073aaf710ff3877f5d9ba294d53060070a4da`.
The numeric report binds all four detailed-round input files and the relevant
deployed source files:
[aggregate evidence](../../configs/ops/route_rl_first10_signal_diagnosis_20261112.json).
No molecule structures, prompts, per-target identities or hidden reference
indices are published here.

| First ten completed rounds | Observed value |
| --- | ---: |
| Completed / planned rounds | 10 / 371 |
| Targets / independent route episodes | 640 / 2,560 |
| Completed backend model requests | 48,876 |
| Decision records, including physical censor attempts | 49,325 |
| Completed reaction occurrences | 1,761 |
| Usable / whole-group-censored groups | 264 / 376 |
| Nonzero-advantage groups | 15 / 640 = 2.34% |
| Structural stock-closed / reference-supported closed episodes | 101 / 0 |

Nonzero advantage does not establish a nonzero effective clipped gradient.
Structural stock closure is not proof of chemical feasibility, and missing
reference support is not proof a proposed reaction is impossible. These are
training observations, not a matched before/after evaluation. The older
November 6 first-round audit used a different runtime condition and is not a
controlled comparison of these failure mechanisms.

## Why useful signal disappears

The detailed round contains all 64 target groups / 256 episodes:

| Terminal status | Episodes |
| --- | ---: |
| Eight consecutive rejected attempts (`recovery_exhausted`) | 186 |
| Runtime length boundary (`resource_censored`) | 65 |
| Structural stock closure (`stock_solved`) | 5 |

There are 3,103 rejected records with execution errors. Of these, 1,193 have a
raw output byte-identical to the immediately preceding unaccepted output. This
counts repeated model outputs; it does not silently assume every pair has the
same executor state. Frequent errors include invalid chemical state (890),
absent electron-source bond (838), nonproductive FINISH (621), absent destination
bond (348), missing atom handle (`KeyError`, 205), and a repeated state (151).
Error feedback is present in the observation, but the sampled policy often does
not produce a useful correction. Whether obsolete atom handles in old
observations cause particular mistakes still requires state-by-state checking.

The scorer finds only three positive-reward episodes in this round, each with
reward 0.1. All three belong to groups that also contain a censored episode.
There are 42 censored groups; the remaining 22 groups have no reward contrast.

The current implementation in `external_route_rl.py:route_group_advantages`
returns no advantages for a group if any reward is `None`. Otherwise Dr.GRPO
uses

\[
A_i = R_i - \frac{1}{4}\sum_{j=1}^{4}R_j.
\]

Consequently all remaining advantages in this round are zero. Whole-group
censoring avoids pretending unknown outcomes are failures or silently changing
G=4. It also amplifies the cost of runtime boundaries: the three observed
positive episodes provide no task-reward gradient in this update. Dropping only
the censored sibling or relabelling it zero would change the estimator and is
not an engineering fix under the frozen contract.

## What the resource boundary actually means

The deployed settings are `max_context=32768` and `max_action_tokens=2048`.
For a prompt of length L, the generation room is

\[
\min(2048,\;32768-L).
\]

When that room is nonpositive, the sampler returns `PHYSICAL_CONTEXT_LIMIT`
without calling the model. When output ends without EOS at the allowed length,
the episode becomes `ACTION_OUTPUT_CONTEXT_LIMIT`. These are configured
capacity boundaries, not observations of a GPU OOM, a Ceph permission error,
or a resource-allocation queue.

The loaded base config declares 262,144 maximum positions. Therefore 32k is not
the architecture's declared maximum. That declaration alone does not establish
that the deployed kernels, memory workspace or training implementation can
safely execute 262k. Raising the deployment cap requires measured capacity and
long-context correctness checks.

In this round:

- 55 episodes exceed the context cap, with final prompt lengths 32,783--37,031
  and a median of 27 decision records, including the terminal censor attempt.
- 10 episodes have truncated output. Nine exhaust the remaining context room;
  only one reaches the full 2,048-action-output allowance.
- In context-censored prompts, the appended history accounts for a median
  98.08% of message-content characters. This is a character fraction, not a
  tokenizer-derived token fraction.

`RouteEpisode._messages` appends all prior attempts, each including a local
observation, output and executor feedback, as well as the route frontier and
completed reactions. This avoids recursive embedding of previous full prompts,
but still repeatedly carries local state snapshots and their action histories.
The code has no aggregate trajectory token budget; per-call context/output
limits and the eight-consecutive-rejection stop remain active. Saying that
removing the aggregate budget removed all length limits would be incorrect.

The observations support the following failure chain: repeated rejected actions
grow the stored history; long prompts hit the configured cap; one censored
sibling removes its whole G4 group; useful reward contrasts become scarce.
They do not yet prove which change would improve the learned policy.

## Repair sequence and acceptance criteria

The following changes are proposals, not fixes implemented by this commit.

1. **Make failure accounting reproducible first.** Keep error families,
   consecutive repeated outputs, current-state atom handles, accepted state
   changes, prompt lengths and whole-group exclusion reasons distinct. Check
   suspect handle errors against the actual current inventory before changing
   the chemistry validator. Keep unknown reference coverage separate from
   invalid executor states.
2. **Preserve complete history without repeated snapshots.** Design a versioned,
   replayable event representation: explicit route frontier and current state,
   stable state/atom identities, all historical actions and all failure feedback,
   with repeated snapshots referenced once. Verify reconstruction of every
   executor state and precursor obligation on saved complete traces. This
   changes model-visible prompts and must be a separately declared observation
   condition; it is not a byte-identical P2 optimization. Do not silently drop
   old route decisions or retain only the current reaction.
3. **Address repeated failed choices with correct sampling accounting.** First
   distinguish unchanged-state identical attempts from legitimate reuse in a
   changed state. Any exclusion, resampling or branching mechanism changes the
   proposal distribution and must record the probabilities actually sampled
   and used for training. Do not replace outputs after sampling, reuse stale
   probabilities, or treat sibling branches as independent root G4 rollouts.
4. **Set capacity from evidence.** Separate context exhaustion from output-cap
   truncation and actual memory/kernel failure. Measure supported long-context
   generation and learner execution before changing deployment caps. Increasing
   context alone does not correct repeated chemically invalid actions.
5. **Require useful learning evidence.** Preserve the full requested training
   denominator, independent G4, fixed parent/reference, reward and protected
   benchmark sets. Report usable and informative groups, repeated-failure
   frequency, censoring, trusted progress, reference-supported closure, and the
   measured phase costs. Use the declared full validation protocol for claims
   of improvement. No hidden mechanism labels, learned critic or automatic
   positive reward for any accepted FLOW is introduced here.

Changes to observation format, sampling/recovery policy or reward must be
separated from the frozen PR104 baseline. Actual profiler integration and
GPU phase-cost evidence remain separate deliverables; this documentation and
numeric snapshot do not claim those implementation files are already in PR104.

## Reproduce the round-ten group accounting

Run from the repository on the shared host. The command reads all 256 episodes,
verifies all four raw-file hashes against the published report, and checks the
group and terminal counts. It launches no model or training task and writes no
temporary dataset. Full reports stay on shared storage.

```bash
source /aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env
/root/miniconda3/envs/meteor/bin/python -S -B - <<'PY'
import collections, gc, hashlib, json
from pathlib import Path

evidence = json.loads(Path(
    'configs/ops/route_rl_first10_signal_diagnosis_20261112.json').read_text())
root = Path('/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/'
            'external_stage2_hf_profile_route_dr_grpo_20261111_02/round-000009')
counts = collections.Counter()
statuses = collections.Counter()
files = sorted(root.glob('rank*/rewarded_episodes.json'))
assert len(files) == 4
for path in files:
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == evidence['round10']['source_sha256'][path.parent.name]
    groups = json.loads(raw)
    del raw
    assert len(groups) == 16
    for group in groups:
        assert len(group) == 4
        counts['groups'] += 1
        censored = any(ep['reward'] is None for ep in group)
        counts['censored_groups'] += censored
        if not censored:
            counts['informative_groups'] += len({ep['reward'] for ep in group}) > 1
        for ep in group:
            statuses[ep['status']] += 1
            if ep['reward'] is not None and ep['reward'] > 0:
                key = ('positive_episodes_in_censored_groups' if censored
                       else 'positive_episodes_in_usable_groups')
                counts[key] += 1
    del groups
    gc.collect()
for key in ('groups', 'censored_groups', 'informative_groups',
            'positive_episodes_in_censored_groups', 'positive_episodes_in_usable_groups'):
    assert counts[key] == evidence['round10'][key], (key, counts[key])
assert dict(statuses) == evidence['round10']['episode_status']
assert sum(statuses.values()) == 256
print(json.dumps({'group_accounting': dict(counts), 'episode_status': dict(statuses)}, sort_keys=True))
PY
```
