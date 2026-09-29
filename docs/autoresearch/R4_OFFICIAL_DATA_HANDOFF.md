# R4 official-data handoff

R4 requires two independent official evaluation sources that are **not**
present in this workspace. Neither may be substituted by a small mirror,
model-generated pathway archive, or a differently split training file.

## Source acquisition (dataset recipient)

Use the authors' [PMechDB download portal](https://deeprxn.ics.uci.edu/pmechdb/download)
under the recipient's own identity and accept its stated CC-BY-NC-ND terms if
authorized to do so. The portal offers separate **PMechDB Dataset** and
**PMechRP Datasets** packages. The former contains the official challenging
test; the latter's `Pathway` subdirectory contains the 350-textbook-pathway
human benchmark. The authors' [documentation](https://deeprxn.ics.uci.edu/pmechdb/howtouse)
names `manually_curated_test_challenging.csv` for the PMechDB split and says
manually curated CSVs contain step SMIRKS/arrow codes plus orbital-pair class.
The exact PMechRP archive layout must be inspected from the received package;
we do not infer filenames or schemas from a paper description.

Keep original archives and extracted raw files outside Git and public PR
artifacts. Provide the two local archive paths (or a private, accessible
directory) to the campaign maintainer. Do **not** paste account credentials,
download tokens or email links into a PR, task config, or chat transcript.
No agent should submit the agreement form on another person's behalf.

## Intake once the official packages are available

1. Record the portal package identity, retrieval date, license text/version,
   original archive byte count and SHA-256, and every extracted evaluation
   file's path, byte count and SHA-256. Preserve the unmodified raw package.
2. Verify `manually_curated_test_challenging.csv` is the **complete official
   challenging split**, not a sample or a re-export. Independently establish
   its row count and stable row identifiers from the received file; the
   publication describes 300 challenging steps, but the file is authoritative.
3. Inspect the PMechRP `Pathway` README and data schema before parsing. Freeze
   all 350 pathway IDs; do not quietly select only executor-compatible cases.
4. Build **separate** evaluation views: raw official denominator and the
   closed-shell, two-electron executor-compatible view. Record every
   unsupported or failed conversion with source ID and reason. Formal
   execution coverage is an outcome metric, not a row-filtering excuse.
5. Keep R4 evaluation rows out of Base/Mech training. Freeze exact product and
   reaction-ID exclusions before scientific sampling, then run the campaign's
   selected-training-row overlap audit.
6. Only after manifests, privacy boundaries and replay checks pass, configure
   `r4_pmechdb_challenging` and `r4_pmechrp_pathways` in
   `configs/autoresearch/mechanistic_verified_retro_smoke.yaml` and evaluate.

The third R4 source, 12–20 literature-supported closed-shell polar catalytic
cycles, is separately curated; it is not supplied by either archive. Until all
three sources meet their own contract, R4 and the matched scientific-smoke
freeze remain incomplete. This handoff does not itself assert access,
compatibility, or a model result.
