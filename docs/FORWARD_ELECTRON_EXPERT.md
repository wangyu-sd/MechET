# Compact forward electron-flow expert

## Scientific role

The forward expert is an optional independent empirical-evidence layer for H3. It is not part of the causal endpoint path and it is not required to define the trace-owned main method.

```text
causal endpoint path:
product -> trace-owned electron-flow program -> deterministic execution -> precursor

optional evidence path:
precursor and conditions -> compact forward expert -> process, target, competitor and uncertainty evidence
```

A learned forward score cannot rescue a formal failure and does not establish kinetics, yield or experimental success.

## Questions under test

The forward study asks whether an architecturally independent compact model can add evidence beyond:

```text
formal execution alone
ordinary precursor-to-product compatibility
inverse actor likelihood
```

It separates:

1. source prediction;
2. sink prediction conditioned on source;
3. complete forward-move support;
4. precursor-to-target compatibility;
5. target-versus-explicit-competitor ranking;
6. uncertainty and calibration.

These outputs must be reported separately rather than collapsed into one unexplained plausibility score.

## Supported chemistry

Version 1 supports mapped, closed-shell, two-electron polar chemistry with electron containers:

```text
LP(atom)
BOND(atom_i, atom_j)
ATOM(atom)
```

Coupled arrows belonging to one elementary event are applied atomically. Radicals, photochemical one-electron steps, transition-metal orbitals, coordination changes and spin dynamics are out of scope.

## Independence contract

| Component | Role | Authority |
|---|---|---|
| trace-owned inverse actor | proposes explicit reverse electron-flow actions | proposal only |
| deterministic executor | validates transitions and derives precursor | hard formal authority |
| compact forward expert | estimates process, target, competitor and uncertainty evidence | soft learned evidence |

The forward expert is trained independently, frozen during the first actor comparison and calibrated on a frozen validation set. Cross-fitting is preferred when its score is used as actor reward.

## Data contract

A forward row records:

```text
stable ID and source revision
mapped current/reactant state
mapped next or target state
source-to-sink moves when unambiguous
reaction/mechanism family when available
conditions and provenance
explicit competitor products or pathways when available
split and label provenance
```

Rows without unambiguous arrow labels may train target compatibility but not source/sink heads. The normalizer never invents arrows.

## Data preparation

```bash
python scripts/forward_expert_data.py download \
  --dataset mech_uspto_31k \
  --revision <frozen-revision> \
  --output data/raw

python scripts/forward_expert_data.py standardize \
  --input data/raw/mech_uspto_31k \
  --output data/forward_expert/reactions.jsonl \
  --source mech_uspto_31k \
  --quarantine data/forward_expert/quarantine.jsonl

python scripts/forward_expert_data.py build \
  --input data/forward_expert/reactions.jsonl \
  --output-dir data/forward_expert/steps
```

Freeze source revisions, licenses, row counts, quarantine reasons, family distributions and split hashes.

## Model

The default compact graph model contains:

```text
atom and bond embeddings
graph message passing
electron-container encoder
source pointer head
source-conditioned sink pointer head
precursor–product compatibility head
condition channel
uncertainty output
```

Optional pretrained chemistry encoders are baselines, not required components of the main method.

## Training

```bash
python scripts/train_forward_expert.py \
  --config configs/forward/forward_expert_small.yaml
```

Required ablations:

```text
ordinary outcome compatibility
process heads only
compatibility only
process plus compatibility
condition channel removed
random negatives
explicit competitors
compact graph model versus optional pretrained encoder
```

Explicit competitors are required for selectivity claims.

## Independent evaluation

Before any actor reranking or reward, report:

```text
source Top-1/Top-k
sink Top-1/Top-k conditioned on source
complete-move Top-1 and MRR
target rank and target recovery at k
pairwise competitor accuracy
target-minus-best-competitor margin
Brier score
expected calibration error
risk–coverage
uncertainty–error correlation
family-wise false acceptance and rejection
```

A high target score without a competitor set is not selectivity.

## Integration experiments

Only after independent calibration compare:

```text
inverse likelihood only
inverse plus ordinary forward compatibility
inverse plus process evidence
inverse plus process and explicit competitors
inverse plus process, competitors and uncertainty
```

Ranking order is:

```text
formal execution hard gate
then calibrated empirical evidence
then inverse likelihood
then diversity or novelty
```

For actor optimization compare:

```text
formal process reward only
formal plus endpoint reward
formal plus ordinary forward compatibility
formal plus calibrated process/competitor evidence
```

Log every reward component separately. A soft reward may not offset formal failure.

## Alternating updates

Actor–forward disagreements are audit candidates, not automatic negatives.

```text
freeze forward expert
train/evaluate actor
mine disagreements
review or independently verify a subset
update forward expert only with supported labels
recalibrate on frozen validation data
```

Stop alternating updates if mined-example performance improves while the frozen audit set or calibration degrades.

## Planning extension

In multistep planning, formal invalidity is a hard prune and forward evidence may contribute a soft edge cost.

Compare under matched search budgets:

```text
inverse score only
inverse plus formal gate
inverse plus ordinary forward score
inverse plus calibrated process/competitor score
inverse plus uncertainty
```

Planning is a downstream extension. It cannot rescue failed causal-faithfulness or compositional-generalization claims.

## Boundaries

- Forward evidence is learned, not experimental truth.
- Selectivity requires explicit alternatives.
- Condition compatibility is limited by available labels and coverage.
- A calibrated score does not prove a low barrier or successful synthesis.
- The deterministic executor remains the only formal authority.
