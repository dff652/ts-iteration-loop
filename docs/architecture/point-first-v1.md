# Point-First V1（方案1）

## 目标
- 在线流程以 `point_id` 为操作单元，而不是文件名。
- 在方案1中，`point_id` 直接采用规范化点位名（`normalize_point_name`）。
- 保持向后兼容：旧字段 `source_id/filename/point_name` 继续可读。

## 规则
- `point_id = normalize_point_name(point_id/source_id/filename/point_name)`（按优先级回退）。
- 文件只用于导入导出，不作为在线主键。
- API 返回以 `point_id` 为主，旧字段作为兼容信息。

## 本周落地范围
- 新增 `point_id` 字段：`annotation_records`、`annotation_segments`、`inference_results`、`review_queue`、`dataset_items`。
- 新增迁移：`0003_point_first_core.sql`。
- 新增回填脚本：`scripts/backfill_point_id.py`。
- 新增点位 API：`/api/v1/points`、`/api/v1/points/{point_id}`、`/api/v1/points/{point_id}/timeline`。

## 已知限制
- 方案1在“点位重命名/别名并存”场景下稳定性较弱。
- 后续若进入多租户/跨系统聚合，建议升级到“UUID + 映射表”方案。
