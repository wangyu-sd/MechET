# Complete v6 coverage audit

All18,210 frozen generated reactions are retained. Compared with the preserved
[v5 traceback audit](../coverage_traceback_20261018/README.md), six newly
source-checked exact preparations change primary support9→15 and unresolved
candidate queries158→152. The original165 candidate queries/210 source records,
49 exact-candidate queries,116 analogue-candidate queries and18,045 no-hit
queries are unchanged.

Of15 supported reactions,13 have indexed candidates and2 were independently
checked source seeds. No-hit reactions remain unresolved, not chemically wrong.
`query_coverage_audit.jsonl.gz` includes all18,210 rows;
`primary_review_queue.jsonl.gz` includes all152 remaining candidate queries.
Both input/output byte hashes and original Ceph paths are in the manifests.

The complete query, context and hit row sequences were checked equal between
v5 and v6. The earlier bibliographic traceback therefore remains applicable;
its original source bindings are retained rather than silently changed to v6.
