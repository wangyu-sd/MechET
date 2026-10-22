# 全部 32 条配对 case

每行链接到该 case 的 GT 和两组完整执行记录。动作数是接受的工具调用数，
包含 IMPORT / FLOW / FINISH；0 表示首个动作被拒绝，不代表没有调用模型。
反应顺序沿用冻结的验证集选择，没有按结果筛选。

- 全部真实输入输出：[完整产物侧 raw generations](product_side/records/raw_generations.jsonl)；
  [主产物-only raw generations](main_product_only/records/raw_generations.jsonl)。
  按 `reaction_id` 可读出每一步的完整 prompt、token IDs 和生成原文。
- 全部参考输入：[GT 源行](product_side/selected_source_rows.jsonl)；
  [主产物选择及删除组分](main_product_only/selected_inputs.jsonl)。
- 全部统一评分：[逐例 endpoint comparison](main_product_only/per_reaction_endpoint_comparison.jsonl)。

| Reaction ID | 完整产物侧端点 | 主产物-only 端点 | 接受动作数：完整 / 主产物 | 逐例记录 |
|---|---|---|---:|---|
| 10643 | 未命中 | 未命中 | 1 / 6 | [GT](product_side/selected_source_rows.jsonl#L1) · [完整](product_side/records/episodes.jsonl#L1) · [主产物](main_product_only/records/episodes.jsonl#L1) |
| 10672 | 命中 | 未命中 | 5 / 4 | [GT](product_side/selected_source_rows.jsonl#L2) · [完整](product_side/records/episodes.jsonl#L2) · [主产物](main_product_only/records/episodes.jsonl#L2) |
| 10700 | 命中 | 未命中 | 7 / 0 | [GT](product_side/selected_source_rows.jsonl#L3) · [完整](product_side/records/episodes.jsonl#L3) · [主产物](main_product_only/records/episodes.jsonl#L3) |
| 11270 | 命中 | 未命中 | 5 / 6 | [GT](product_side/selected_source_rows.jsonl#L4) · [完整](product_side/records/episodes.jsonl#L4) · [主产物](main_product_only/records/episodes.jsonl#L4) |
| 1273 | 命中 | 未命中 | 2 / 3 | [GT](product_side/selected_source_rows.jsonl#L5) · [完整](product_side/records/episodes.jsonl#L5) · [主产物](main_product_only/records/episodes.jsonl#L5) |
| 1313 | 命中 | 未命中 | 5 / 1 | [GT](product_side/selected_source_rows.jsonl#L6) · [完整](product_side/records/episodes.jsonl#L6) · [主产物](main_product_only/records/episodes.jsonl#L6) |
| 14983 | 命中 | 未命中 | 4 / 0 | [GT](product_side/selected_source_rows.jsonl#L7) · [完整](product_side/records/episodes.jsonl#L7) · [主产物](main_product_only/records/episodes.jsonl#L7) |
| 14986 | 命中 | 未命中 | 3 / 2 | [GT](product_side/selected_source_rows.jsonl#L8) · [完整](product_side/records/episodes.jsonl#L8) · [主产物](main_product_only/records/episodes.jsonl#L8) |
| 1589 | 命中 | 未命中 | 3 / 0 | [GT](product_side/selected_source_rows.jsonl#L9) · [完整](product_side/records/episodes.jsonl#L9) · [主产物](main_product_only/records/episodes.jsonl#L9) |
| 16397 | 命中 | 未命中 | 5 / 2 | [GT](product_side/selected_source_rows.jsonl#L10) · [完整](product_side/records/episodes.jsonl#L10) · [主产物](main_product_only/records/episodes.jsonl#L10) |
| 1833 | 命中 | 未命中 | 7 / 0 | [GT](product_side/selected_source_rows.jsonl#L11) · [完整](product_side/records/episodes.jsonl#L11) · [主产物](main_product_only/records/episodes.jsonl#L11) |
| 18612 | 命中 | 未命中 | 5 / 4 | [GT](product_side/selected_source_rows.jsonl#L12) · [完整](product_side/records/episodes.jsonl#L12) · [主产物](main_product_only/records/episodes.jsonl#L12) |
| 18643 | 命中 | 未命中 | 7 / 0 | [GT](product_side/selected_source_rows.jsonl#L13) · [完整](product_side/records/episodes.jsonl#L13) · [主产物](main_product_only/records/episodes.jsonl#L13) |
| 198 | 命中 | 未命中 | 5 / 1 | [GT](product_side/selected_source_rows.jsonl#L14) · [完整](product_side/records/episodes.jsonl#L14) · [主产物](main_product_only/records/episodes.jsonl#L14) |
| 20393 | 未命中 | 未命中 | 2 / 4 | [GT](product_side/selected_source_rows.jsonl#L15) · [完整](product_side/records/episodes.jsonl#L15) · [主产物](main_product_only/records/episodes.jsonl#L15) |
| 21092 | 命中 | 未命中 | 7 / 0 | [GT](product_side/selected_source_rows.jsonl#L16) · [完整](product_side/records/episodes.jsonl#L16) · [主产物](main_product_only/records/episodes.jsonl#L16) |
| 23314 | 命中 | 未命中 | 7 / 1 | [GT](product_side/selected_source_rows.jsonl#L17) · [完整](product_side/records/episodes.jsonl#L17) · [主产物](main_product_only/records/episodes.jsonl#L17) |
| 23845 | 命中 | 未命中 | 5 / 5 | [GT](product_side/selected_source_rows.jsonl#L18) · [完整](product_side/records/episodes.jsonl#L18) · [主产物](main_product_only/records/episodes.jsonl#L18) |
| 25074 | 命中 | 未命中 | 2 / 6 | [GT](product_side/selected_source_rows.jsonl#L19) · [完整](product_side/records/episodes.jsonl#L19) · [主产物](main_product_only/records/episodes.jsonl#L19) |
| 25201 | 命中 | 未命中 | 3 / 5 | [GT](product_side/selected_source_rows.jsonl#L20) · [完整](product_side/records/episodes.jsonl#L20) · [主产物](main_product_only/records/episodes.jsonl#L20) |
| 25433 | 命中 | 未命中 | 3 / 3 | [GT](product_side/selected_source_rows.jsonl#L21) · [完整](product_side/records/episodes.jsonl#L21) · [主产物](main_product_only/records/episodes.jsonl#L21) |
| 2657 | 命中 | 未命中 | 3 / 3 | [GT](product_side/selected_source_rows.jsonl#L22) · [完整](product_side/records/episodes.jsonl#L22) · [主产物](main_product_only/records/episodes.jsonl#L22) |
| 27732 | 命中 | 未命中 | 2 / 4 | [GT](product_side/selected_source_rows.jsonl#L23) · [完整](product_side/records/episodes.jsonl#L23) · [主产物](main_product_only/records/episodes.jsonl#L23) |
| 29278 | 未命中 | 未命中 | 1 / 4 | [GT](product_side/selected_source_rows.jsonl#L24) · [完整](product_side/records/episodes.jsonl#L24) · [主产物](main_product_only/records/episodes.jsonl#L24) |
| 29829 | 命中 | 未命中 | 4 / 4 | [GT](product_side/selected_source_rows.jsonl#L25) · [完整](product_side/records/episodes.jsonl#L25) · [主产物](main_product_only/records/episodes.jsonl#L25) |
| 30034 | 命中 | 未命中 | 3 / 5 | [GT](product_side/selected_source_rows.jsonl#L26) · [完整](product_side/records/episodes.jsonl#L26) · [主产物](main_product_only/records/episodes.jsonl#L26) |
| 3645 | 命中 | 未命中 | 7 / 0 | [GT](product_side/selected_source_rows.jsonl#L27) · [完整](product_side/records/episodes.jsonl#L27) · [主产物](main_product_only/records/episodes.jsonl#L27) |
| 5048 | 命中 | 未命中 | 7 / 1 | [GT](product_side/selected_source_rows.jsonl#L28) · [完整](product_side/records/episodes.jsonl#L28) · [主产物](main_product_only/records/episodes.jsonl#L28) |
| 6605 | 命中 | 未命中 | 7 / 3 | [GT](product_side/selected_source_rows.jsonl#L29) · [完整](product_side/records/episodes.jsonl#L29) · [主产物](main_product_only/records/episodes.jsonl#L29) |
| 6919 | 命中 | 未命中 | 7 / 5 | [GT](product_side/selected_source_rows.jsonl#L30) · [完整](product_side/records/episodes.jsonl#L30) · [主产物](main_product_only/records/episodes.jsonl#L30) |
| 8002 | 未命中 | 未命中 | 7 / 0 | [GT](product_side/selected_source_rows.jsonl#L31) · [完整](product_side/records/episodes.jsonl#L31) · [主产物](main_product_only/records/episodes.jsonl#L31) |
| 8798 | 命中 | 未命中 | 7 / 1 | [GT](product_side/selected_source_rows.jsonl#L32) · [完整](product_side/records/episodes.jsonl#L32) · [主产物](main_product_only/records/episodes.jsonl#L32) |

JSONL 文件较大时，GitHub 可能不完整显示单行；请用 Raw 下载后读取。
所有真实 prompt 和生成原文均已包含在 Git 中，不依赖内部共享盘才能查看。
