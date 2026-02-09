# TODO

## 当前执行顺序（已确认）

### P0（先做）
- [x] 修复推理任务状态枚举不一致：`TaskStatus.PROCESSING` 与 schema 枚举统一（已改为 `running`）
- [x] 补齐 `CheckOutlierAdapter.convert_to_annotation_format`，打通 `core/tasks.py` 与 `api/inference.py` 的导出预标注调用
- [x] 恢复 Annotator 鉴权能力（默认鉴权；保留开发态开关）

### P1（P0 完成后）
- [x] 将“数据资产管理”从 WebUI 直连 DB 下沉为 API 层（`/api/v1/assets/...`）
- [x] 引入数据库迁移机制（轻量 SQL 迁移 + `schema_migrations` + `scripts/db_migrate.py`）
- [ ] 统一任务执行模型（BackgroundTasks / UI 内部进程 / Celery），保证状态与日志口径一致
- [ ] DB-First 统一数据结构（标注/索引/审核/资产共享同一实体与字段口径）

### P2（产品化增强）
- [x] 统一版本号与文档口径（`settings.APP_VERSION` 与 docs 发布版本一致）
- [x] 增加最小回归测试集（当前已覆盖 P0 关键链路：转换与鉴权）
- [ ] 补充权限与审计字段（数据资产 owner/组织维度）
- [ ] 文件角色收敛为导出产物（在线流程不再依赖 JSON/CSV 中间态）

## DB-First 实施拆解（新增）

### 阶段 A：统一模型与读写路径
- [x] 定义段级统一实体（segment + annotation + review 字段）并补 migration
- [x] 标注读取 API 改为仅查 DB；工作区保存改为仅写 DB
- [x] 索引数据段筛选与标注结果列表共用同一查询来源

### 阶段 B：转换功能角色收敛
- [ ] 将“标注数据转换”拆分为导入适配器与导出适配器
- [ ] 下线模块间中转转换逻辑（仅保留 import/export/migrate）

### 阶段 C：数据资产与导出统一
- [ ] 数据资产构建、筛选、冻结全基于 DB
- [ ] 训练导出仅从 DB 快照生成文件（JSONL/CSV），并记录导出版本

## 存量待办
- 处理缺失点位 `LHS2_20250322_20250325_H2S.csv`：补齐 CSV 或从标注集中剔除该 JSON
- 如需拆分版本，运行 `services/data_processing/run_pipeline.py --split true` 并确认命名与用途
