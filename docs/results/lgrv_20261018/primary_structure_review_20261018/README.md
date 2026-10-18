# An indexed exact hit is not automatically an exact experimental witness

One primary preparation checked in this increment illustrates the distinction:
[US7790756B2](https://patents.google.com/patent/US7790756B2/en),ExampleA26.

| Structure | Fluorine relative to aryl NH2 | Fluorine relative to phenolic/ether O |
|---|---|---|
| Frozen generated reaction and extracted indexed candidate |ortho|meta|
| Independently transcribed named starting material/product in ExampleA26 |meta|ortho|

The source names **2-fluoro-4-aminophenol** and
**4-(2-chloropyridin-4-yloxy)-3-fluorophenylamine**. The frozen query instead
contains the other fluorine positional isomer. Independent source key
`49ff55f738fd74c6c77fd9dcec0d3532f19033fce2f2beae3d8ee8be5793addb`
differs from query key
`995f7608989fd72c921c3ace2d4c4223bc86153b17aa7cbac4fb56e03251e477`.
This is a molecular-graph distinction, not atom-map numbering.

The cached original/text hashes, short unique witness, original indexed record,
named source structures, frozen query and source-specific assessment are in
`source_mismatch_review.json`, copied byte-for-byte with a distribution receipt.
The actual source-evidence checker retains **TierC** for this query/source pair.

Scope is deliberately narrow: **this named preparation is not an exact witness
for the generated reaction**. Other source candidates were not ruled out; this
does not establish that the generated chemistry is impossible, that every
other example in the patent differs, or that the source merits TierD. The
original retrieved hit is retained rather than deleted or counted as verified.
