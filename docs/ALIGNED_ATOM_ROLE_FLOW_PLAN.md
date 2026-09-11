# Aligned atom-role electron-flow plan

## Objective

Replace the marked-state localization interface in PR #53 with a simpler model-facing contract that does not require the model to predict atom IDs, atom maps, temporary pointers, SMARTS, or a regenerated copy of the molecular string.

The proposed interface is:

\[
\boxed{\text{current molecular state + aligned atom slots}
\rightarrow
\text{atom-role mask + compact electron flow}}
\]

The key change is that localization becomes **aligned role tagging** over the atoms already present in the input, rather than generation of an address or generation of a marked molecule.

## 1. Why the previous PR #53 interface is not sufficient

The previous proposal asked the model to emit a complete marked molecular string such as:

```text
<A>[O-].C<B>C(=O)<C>Br
```

Even when marker insertion is the intended operation, the model still has to regenerate the molecular serialization. In practice, equivalent SMILES traversals, disconnected-component order, symmetric atoms, and local string deviations have to be repaired by graph-aligned flexible inference.

That improves robustness, but it means localization is still mediated by regenerating the molecular representation.

Temporary pointers such as `r7`, `p12`, or atom-map-like labels do not solve the underlying problem either. They still ask the model to map chemical reasoning onto an arbitrary atom identifier.

The revised interface removes both failure modes.

## 2. Model-facing input

For every authoritative current state, the executor/parser produces two synchronized views.

### Molecular view

```text
CURRENT STATE:
[O-].CC(=O)Br
```

### Aligned atom-slot view

```text
ATOM SLOTS:
[O-] | C | C | O | Br
```

The atom-slot list is obtained deterministically from the current molecular graph and the chosen canonical serialization. It contains every atom occurrence exactly once and in the same deterministic order used by the executor-side alignment table.

The slots are **not GT reaction-centre labels**. They are simply a decomposition of the currently observed molecule into its atom occurrences.

At inference time they require only the current molecular state. No precursor, reaction centre, atom mapping, intermediate-state GT, or cross-state correspondence is used.

The executor privately retains the positional alignment

\[
\text{slot }i \leftrightarrow \text{current graph vertex }v_i.
\]

This is not a model target and is never emitted by the model.

## 3. Model-facing output

The model no longer emits a marked copy of the molecule.

For the five atom slots above it emits exactly five role symbols:

```text
CURRENT ROLES:
A | _ | B | _ | C
```

The alignment is positional:

- the first role describes the first atom slot;
- the second role describes the second atom slot;
- and so on.

Therefore the model does not say "choose atom 1" or "choose map 137". It classifies each already-present atom occurrence with an event-local role.

For the example above:

```text
ATOM SLOTS:     [O-] | C | C | O | Br
CURRENT ROLES:    A   | _ | B | _ | C
```

so the executor obtains the event-local bindings

\[
A\leftrightarrow [O^-],\qquad
B\leftrightarrow C,\qquad
C\leftrightarrow Br.
\]

The model then emits the compact electron-flow program:

```text
FLOW:
A>AB ; BC>C
```

with the existing semantics

\[
A>AB \equiv LP(A)\rightarrow BOND(A,B),
\]

\[
BC>C \equiv BOND(B,C)\rightarrow ATOM(C).
\]

The complete event is executed transactionally by the existing executor.

## 4. Imports

Atoms already present in the current state use the `CURRENT ROLES` mask.

If a new precursor fragment must be introduced, the model emits the unmapped fragment and a role mask aligned only to that new fragment.

Example:

```text
IMPORTS:
[OH-]

IMPORT SLOTS:
[OH-]

CURRENT ROLES:
_ | _ | B | _ | C

IMPORT ROLES:
A

FLOW:
A>AB ; BC>C
```

The imported fragment is parsed after generation. Its atom slots are obtained from the generated fragment itself, and `IMPORT ROLES` must have exactly the same number of positions.

Thus existing-state localization never requires the model to regenerate the current molecule, while newly introduced chemistry remains expressible.

## 5. Formal contract

Let the current molecular graph contain ordered atom slots

\[
V_t=(v_1,\ldots,v_n).
\]

The model predicts an aligned role sequence

\[
R_t=(r_1,\ldots,r_n),
\qquad
r_i\in\{\_,A,B,C,\ldots\}.
\]

The binding function is deterministic:

\[
\beta_t(A)=v_i\quad\text{iff}\quad r_i=A.
\]

No generated atom address appears in this definition.

The compact FLOW program is then compiled through \(\beta_t\) into the existing executor-native electron containers and moves.

One model action remains one elementary event:

\[
E_t=\{C_j^{src}\rightarrow C_j^{sink}\}_{j=1}^{k},
\]

and the executor produces

\[
S_{t+1}=\mathcal F(S_t,E_t).
\]

## 6. Why this is different from atom mapping

Atom mapping asks the model or preprocessing pipeline to associate atoms with explicit identities such as `:137` and to preserve correspondence across reaction states.

The aligned-role interface does neither.

The model never predicts an identity token. It only emits one role label per already-present input slot. The role label has no meaning outside the current elementary event.

The task is therefore analogous to sequence tagging:

```text
atom slots:   a1  a2  a3  a4  a5
roles:         A   _   B   _   C
```

rather than address prediction:

```text
selected atoms: 1, 3, 5
```

or atom-map prediction:

```text
selected maps: 82, 137, 211
```

## 7. Why this is different from marked-state regeneration

The previous PR #53 interface required output proportional to the molecular serialization:

```text
<A>[O-].C<B>C(=O)<C>Br
```

The revised interface keeps the molecule entirely on the input side and generates only the aligned roles plus FLOW:

```text
A | _ | B | _ | C
A>AB ; BC>C
```

Consequently:

- the model cannot change SMILES traversal while locating atoms;
- the model cannot reorder disconnected components while locating atoms;
- the model cannot create malformed current-state syntax;
- localization does not require graph-isomorphic recovery from a regenerated molecule;
- the output focuses supervision on event roles and electron flow rather than molecular copying.

Flexible executor logic remains useful for transactional execution, symmetry, rollback, and graph-level validation, but it is no longer the primary mechanism that repairs localization output.

## 8. Symmetry and equivalent localizations

A single offline mapped trace must not make one arbitrary symmetric atom uniquely correct.

If two role assignments produce execution-equivalent successor states,

\[
\mathcal F(S_t,E_1)\equiv\mathcal F(S_t,E_2),
\]

both should be accepted for evaluation.

Supervision should likewise support symmetry-equivalent role masks through augmentation or equivalence-aware targets rather than forcing the model to reproduce one arbitrary atom-map representative.

## 9. Deterministic atom-slot construction

The slot sequence must satisfy four requirements:

1. every current-state atom appears exactly once;
2. slot order is deterministic for the same authoritative graph;
3. the executor can map every slot position back to the corresponding graph vertex without model assistance;
4. no atom map, reaction-centre label, or GT-derived feature is exposed in the slot text.

The molecular string and atom-slot sequence are two synchronized views of the same current state, not two independently generated molecular representations.

## 10. Compact FLOW remains unchanged

The FLOW representation from PR #53 is retained because it is already close to the executor semantics and avoids reaction-family templates.

Examples:

```text
A>AB
```

means `LP(A) -> BOND(A,B)`.

```text
AB>B
```

means `BOND(A,B) -> ATOM(B)`.

```text
AB>BC
```

means `BOND(A,B) -> BOND(B,C)`.

Multiple clauses in one action represent coupled electron-pair movements in one elementary event.

The proposal does not replace electron flow with labels such as `SN2`, `electrophile`, or `leaving_group`.

## 11. Information boundary

Existing atom maps may be used only offline to convert current executable traces into aligned role-mask supervision.

For a training event, the offline converter can determine which canonical atom slots correspond to the gold electron containers, assign event-local roles, and then remove all atom-map identities from the model-facing sample.

At inference:

```text
current molecule
    -> deterministic atom slots
    -> Qwen role mask + FLOW
    -> executor compilation
    -> next state
```

No GT correspondence is required.

## 12. Required audit before implementation is adopted

The next audit should compare this role-mask interface directly with the marked-state interface on the same strict executable events.

Required measurements:

1. **slot determinism** — one current graph produces one reversible slot ordering;
2. **role-mask representability** — every executor event can be converted to roles + FLOW without information loss;
3. **exact move round trip** — original moves -> roles + FLOW -> compiled moves;
4. **exact successor replay** — compiled event reproduces the authoritative successor;
5. **symmetry-equivalent target rate** — quantify how often multiple role masks represent the same executable transition;
6. **model-facing target length** — compare role-mask + FLOW against full marked-state regeneration;
7. **localization failure categories** — especially wrong role assignment versus representation/parsing failure.

Do not improve the denominator by filtering difficult reactions.

## 13. Explicitly rejected alternatives

For this iteration, do not use as the primary localization interface:

- model-generated atom maps;
- temporary atom-ID or pointer tokens;
- free-form SMARTS localization;
- regenerated marked SMILES as the model output;
- a separate neural graph or pointer decoder;
- whole-event A/B/C candidate classification;
- hand-authored reaction templates or semantic chemistry predicates.

## 14. Decision criterion

The representation should be adopted only if the atom-role mask preserves exact executor replay while materially reducing localization-specific failure modes relative to marked-state regeneration.

The intended causal interface is:

\[
\boxed{
\text{current state}
\rightarrow
\text{aligned atom roles}
\rightarrow
\text{compact electron flow}
\rightarrow
\text{executor state transition}
}
\]

The precursor must continue to be derived only from committed executable events. The role mask is a localization interface, not a hidden precursor answer or post-hoc explanation.
