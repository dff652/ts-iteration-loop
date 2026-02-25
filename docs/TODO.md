# TODO

## 当前执行顺序（已确认）

### P0（先做）
- [x] 修复推理任务状态枚举不一致：`TaskStatus.PROCESSING` 与 schema 枚举统一（已改为 `running`）
- [x] 补齐 `CheckOutlierAdapter.convert_to_annotation_format`，打通 `core/tasks.py` 与 `api/inference.py` 的导出预标注调用
- [x] 恢复 Annotator 鉴权能力（默认鉴权；保留开发态开关）

### P1（P0 完成后）
- [x] 将“数据资产管理”从 WebUI 直连 DB 下沉为 API 层（`/api/v1/assets/...`）
- [x] 引入数据库迁移机制（轻量 SQL 迁移 + `schema_migrations` + `scripts/db_migrate.py`）
- [x] 统一任务执行模型（BackgroundTasks / UI 内部进程 / Celery），保证状态与日志口径一致（data/training/inference API 已切至 Celery 投递；WebUI 推理/数据获取/训练入口及训练 stop/log 轮询均已收敛到 API；三者 status/log 返回字段已对齐）
- [ ] DB-First 统一数据结构（标注/索引/审核/资产共享同一实体与字段口径）

### P2（产品化增强）
- [x] 统一版本号与文档口径（`settings.APP_VERSION` 与 docs 发布版本一致）
- [x] 增加最小回归测试集（当前已覆盖 P0 关键链路：转换与鉴权）
- [x] 时间戳口径统一（移除 `datetime.utcnow()`，改为 UTC helper / timezone-aware 调用）
- [x] 补充权限与审计字段（数据资产 owner/组织维度；新增 `owner_id/org_id/created_by/updated_by` 与筛选口径）
- [x] 文件角色收敛为导出产物（在线流程不再依赖 JSON/CSV 中间态；已完成“推理自动反馈链路”去临时 JSON + “WebUI 转换页”去文件兜底 + “标注管理页”DB-First + `import-inference` 支持 rows 直传（`inference_file` 仅兼容模式） + 标注 GET 默认禁用 CSV 回退 + `inference/export-to-annotation` 默认 rows 返回并支持可选落盘兼容）

## DB-First 实施拆解（新增）

### 阶段 A：统一模型与读写路径
- [x] 定义段级统一实体（segment + annotation + review 字段）并补 migration
- [x] 标注读取 API 改为仅查 DB；工作区保存改为仅写 DB
- [x] 索引数据段筛选与标注结果列表共用同一查询来源
- [x] 引入 point-first 标识（方案1）：`point_id=规范化点位名`，并在核心实体补齐字段与回填脚本
- [x] 新增点位中心 API：`/api/v1/points`、`/api/v1/points/{point_id}`、`/api/v1/points/{point_id}/timeline`
- [x] 推理监控入口切换为点位选择（UI 内部自动映射到最新 CSV 路径）
- [x] 数据获取页切换为点位视角（点位ID列表 + 点位到文件自动解析）

### 阶段 B：转换功能角色收敛
- [x] 新增 `AnnotationExportAdapter` 并将 `assets/annotation` 导出入口切换到独立导出适配器
- [x] 新增 `AnnotationImportAdapter` 并将 `annotation/import-inference` 内部导入链路切换到独立导入适配器
- [x] WebUI 转换页改为直接使用 `AnnotationExportAdapter`（减少通过 `DataProcessingAdapter` 的中转）
- [x] 将“标注数据转换”拆分为导入适配器与导出适配器
- [x] 下线模块间中转转换逻辑（`DataProcessingAdapter` 不再提供 `convert_annotations`）

### 阶段 C：数据资产与导出统一
- [x] 数据资产构建、筛选、冻结全基于 DB（`/api/v1/assets/sources` 的 annotations/inference/training 全改为 DB 查询）
- [x] 训练导出仅从 DB 快照生成文件（JSONL/CSV），并记录导出版本（写入 `dataset_assets.meta.exports/last_export`）

## 存量待办
- [x] 处理缺失点位 `LHS2_20250322_20250325_H2S.csv`：已从标注集与审核队列剔除（`scripts/prune_missing_annotation_points.py --point-id LHS2_20250322_20250325_H2S --apply`）
- 其他缺失源数据点位可按需清理：`python scripts/prune_missing_annotation_points.py --point-id <POINT_ID> --apply`
- 如需拆分版本，运行 `services/data_processing/run_pipeline.py --split true` 并确认命名与用途
