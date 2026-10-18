# Actual inverse electron-flow fork on the source-backed case

[PDF](figures/extended_data_actual_fork.pdf) ·
[SVG](figures/extended_data_actual_fork.svg) ·
[PNG](figures/extended_data_actual_fork.png)

Two accepted model actions, E16/E17, originate at the **same recorded parent
state and history**, after the program's first IMPORT. They have distinct
canonical chemical successors, not merely different atom addresses.

Choice1/E16 protonates a carbonyl oxygen. Its native child action history is
byte-payload-identical to the first two actions of the fixed five-FLOW terminal
program for acid-chloride formation. That terminal maps to the source-checked
complete reaction in WO2018104954A1 and occurs on the frozen-stock-closed
Cabozantinib route. Choice2/E17 gives a different accepted symbolic successor;
no endpoint, route rescue or experimental support is inferred for that branch.
The figure does not depict rejected actions or invent intermediate structures.

**Caption:** Recorded inverse electronic branching from one executor state.
The top row shows exact source–sink moves compiled from the two native action
payloads; the lower row shows their saved successors. Both are formal executor
constructions. Only the finalized complete acid-chloride transformation has
checked net precedent; the patent does not establish either microscopic
protonation event or a physical intermediate.

This is one source-linked fork, not an exhaustive tree or a matched-compute
comparison. All earlier full-tree artifacts remain unchanged. The manifest
records native/report hashes, pointers, both attempts and the selected
terminal-program prefix. Reproduce using `normalize_native_tree`, restrict
accepted edges to the recorded reaction product and parent context
`e959b28924b31ee3dd6780b75c182764b89be757915e1e911489f73c886af20a`,
then call the existing `render_microscope`/`export` functions. No new model
sampling is involved.

Final PNG inspected; PDF/SVG183mm wide,159.81mm high; fonts embedded and PNG
approximately300dpi. This is not independent expert chemical validation.
