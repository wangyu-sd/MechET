# In-place grounded electron-flow representation plan

## Objective

Define the next MechET representation change without introducing another atom-addressing scheme, free-form SMARTS localization, candidate-label ranking, or a separate learned decoder.

The proposed representation is **in-place grounding + compact electron-flow programs**.

The model should not generate an atom identifier and then ask the executor to recover that atom. Instead, grounding is expressed by inserting event-local role markers directly at atom occurrences in the authoritative current molecular serialization. The model then emits only the compact source-to-sink electron transfers for one elementary event.

## 1. Elementary action unit

One model action corresponds to one elementary electron-flow event:

\[
E_t = \{m_1, m_2, \ldots, m_k\},
\]

where each `m_i` is an explicit two-electron transfer between existing MechET electron containers.

All moves in the event are committed transactionally and induce one authoritative state transition:

\[
S_{t+1}=\mathcal F(S_t,E_t).
\]

This preserves concerted electron-flow structure while reducing decision horizon relative to treating each curved arrow as an independent model turn.

## 2. Remove explicit atom-address generation

Do not replace atom maps with another arbitrary identifier such as `r7`, `p12`, or a new persistent pointer vocabulary. Those remain address-prediction problems under a different surface form.

For current state

```text
[O-].CBr
```

a grounded event may instead be represented as

```text
<A>[O-].<B>C<C>Br
FLOW A>AB ; BC>C
```

`A`, `B`, and `C` are not atom IDs. They are event-local role variables. Their bindings are determined solely by the atom occurrence at which the marker is inserted in the current molecular serialization.

Thus the model predicts **where to mark the existing state**, not an external symbolic address to be resolved later.

## 3. Molecular state is constrained, not freely regenerated

The authoritative molecular serialization must remain unchanged during grounding. The model is not allowed to rewrite the current SMILES.

Given serialization `x(S_t)`, the only legal grounding edits are role-marker insertions at valid atom boundaries:

\[
\mathcal M(S_t)=\{\text{insert event-role markers into valid atom positions of }x(S_t)\}.
\]

No deletion, substitution, bond edit, ring-index edit, or atom insertion is permitted in this grounding stage.

This is intended to eliminate the failure mode in which localization errors are created by generating malformed or semantically ambiguous SMARTS or by regenerating an incorrect molecular state.

## 4. Compact FLOW language

After in-place role grounding, the model expresses the event using a compact language that remains exactly aligned with the existing `ElectronMove(source, sink)` semantics.

For example:

```text
A>AB
```

means

\[
LP(A) \rightarrow BOND(A,B).
\]

Likewise

```text
AB>B
```

means

\[
BOND(A,B) \rightarrow ATOM(B).
\]

and

```text
AB>BC
```

means

\[
BOND(A,B) \rightarrow BOND(B,C).
\]

A substitution-like elementary event can therefore be written compactly as

```text
<A>[O-].<B>C<C>Br
FLOW A>AB ; BC>C
```

which compiles to the two coupled electron transfers

\[
LP(A)\rightarrow BOND(A,B),
\]

\[
BOND(B,C)\rightarrow ATOM(C).
\]

The compact representation must be losslessly compilable into the current executor representation. It must not rely on reaction-family names or semantic predicates such as `nucleophile`, `electrophile`, or `leaving_group`.

## 5. Complex elementary events

Do not force all chemistry into one linear atom path.

The representation remains a set of coupled source-to-sink electron transfers within one elementary event. A multi-arrow event is simply expressed by additional FLOW clauses separated within the same transactional event.

For example, an elimination-like event can contain several coupled transfers in one action. Pericyclic or other concerted events can likewise contain multiple coupled transfer clauses without introducing separate reaction templates such as `CYCLE` or `DIELS_ALDER`.

The semantic object remains

\[
E_t=\{C_i^{src}\rightarrow C_i^{sink}\}_{i=1}^{k}.
\]

The compact syntax is only a serialization of this event graph.

## 6. Higher-level reaction trajectories

A complete reaction remains a sequence of executed elementary events:

\[
S_0\xrightarrow{E_1}S_1\xrightarrow{E_2}\cdots\xrightarrow{E_T}S_T.
\]

A catalytic cycle, when the executor chemistry is eventually extended to support the necessary metal/orbital state, should be represented at this trajectory level rather than as one super-path. The event representation itself therefore does not need to change to accommodate cyclic state-space behavior.

The current formal scope remains mapped, closed-shell, two-electron polar chemistry supported by the existing executor. No current claim is made for transition-metal orbital dynamics, spin changes, or single-electron chemistry.

## 7. Training/inference information boundary

Existing atom maps may be used offline only to convert the current executable traces into supervision.

For a gold step, the offline compiler can use atom maps to determine where role markers belong and to convert the existing source/sink moves into the compact FLOW program. Atom maps are then removed from the model-facing representation.

At inference, only the current molecular state is available. The deterministic molecular parser identifies atom boundaries in the current serialization; the model marks those positions and emits the FLOW event. No gold precursor, gold reaction center, gold atom correspondence, or gold intermediate state is required.

The intended claim is therefore **mapping-free inference**, not address-free chemistry.

## 8. Symmetry and equivalence

Atom-level labels should not be treated as uniquely correct when several grounded events are chemically equivalent.

Two grounded events belong to the same execution-equivalence class when their committed executor transitions produce equivalent authoritative successor states:

\[
\mathcal F(S_t,E_1)\equiv\mathcal F(S_t,E_2).
\]

Evaluation and supervision should avoid penalizing symmetry-equivalent alternatives solely because the offline mapped trace selected one representative.

## 9. Required representation audit

Before adopting this representation as a new MechET condition, run a model-free audit on the existing strict executable traces.

The audit should measure:

1. **Representability** — the fraction of existing elementary events that can be serialized without information loss.
2. **Move round-trip fidelity** — original moves -> marked state + FLOW -> compiled moves.
3. **Successor replay fidelity** — whether the compiled event reproduces the authoritative next state under the current executor.
4. **Compression** — model-facing token length and number of decisions per reaction relative to the current program representation.
5. **Failure taxonomy** — unsupported source/sink combinations, unusual BE-delta cases, serialization ambiguity, or other non-representable events.
6. **Symmetry behavior** — rate of multiple execution-equivalent groundings and handling of those equivalence classes.
7. **Serialization determinism** — the same current graph must yield a deterministic atom-boundary serialization and reversible role binding.

The representation must not gain coverage by filtering difficult rows. Any unresolved event remains a denominator failure in the audit.

## 10. Scientific hypothesis

The scientific hypothesis is not that atom roles or a new mini-language are themselves novel.

The hypothesis is that explicit electron-flow reasoning becomes more learnable when molecular grounding is represented as constrained in-place marking of the current state, while the generated action is reduced to a compact, executable source-to-sink electron-flow program.

This changes the causal interface from

\[
\text{generate symbolic address}\rightarrow\text{resolve site}\rightarrow\text{execute move}
\]

to

\[
\text{mark current molecular state}\rightarrow\text{emit compact electron flow}\rightarrow\text{execute event}.
\]

The precursor must still be produced only by executing the committed electron-flow trajectory. The compact representation is therefore part of the causal transformation program, not a post-hoc rationale.

## 11. Explicitly rejected directions for this iteration

Do not use the following as the primary representation in this iteration:

- free-form SMARTS localization;
- atom-map prediction;
- arbitrary persistent or temporary numeric atom handles as the model target;
- A/B/C whole-event candidate classification as the main policy interface;
- a separate graph/pointer decoder attached to Qwen;
- a handcrafted reaction-template or chemistry-predicate ontology;
- a lifted CSP language that requires manually defined labels such as `electrophile` or `leaving_group`.

These directions either retain the original addressing problem, add a second learned architecture, or shift the burden into a hand-engineered symbolic ontology.

## 12. Decision criterion

Proceed with this representation only if the model-free compiler demonstrates near-complete exact replay over the current executable chemistry and materially reduces model-facing action complexity.

The first-order success criterion is therefore not model accuracy but representation fidelity:

\[
\boxed{\text{compact representation}\rightarrow\text{same executable event}\rightarrow\text{same successor state}.}
\]

Only after that contract is established should this representation replace the current SMARTS/address-based interface in a learned MechET condition.
