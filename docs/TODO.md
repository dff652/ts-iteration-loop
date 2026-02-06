# TODO

## 当前执行顺序（已确认）

### P0（先做）
- [ ] 修复推理任务状态枚举不一致：`TaskStatus.PROCESSING` 与 schema 枚举统一（建议改为 `running`）
- [ ] 补齐 `CheckOutlierAdapter.convert_to_annotation_format`，打通 `core/tasks.py` 与 `api/inference.py` 的导出预标注调用
- [ ] 恢复 Annotator 鉴权能力（移除默认绕过逻辑，保留开发态开关）

### P1（P0 完成后）
- [ ] 将“数据资产管理”从 WebUI 直连 DB 下沉为 API 层（`/api/v1/assets/...`）
- [ ] 引入数据库迁移机制（Alembic 或一次性迁移脚本），替代仅 `init_db().create_all`
- [ ] 统一任务执行模型（BackgroundTasks / UI 内部进程 / Celery），保证状态与日志口径一致

### P2（产品化增强）
- [ ] 统一版本号与文档口径（`settings.APP_VERSION` 与 docs 发布版本一致）
- [ ] 增加最小回归测试集（推理索引写入、审核队列、approved 导出、训练后评估）
- [ ] 补充权限与审计字段（数据资产 owner/组织维度）

## 存量待办
- 处理缺失点位 `LHS2_20250322_20250325_H2S.csv`：补齐 CSV 或从标注集中剔除该 JSON
- 如需拆分版本，运行 `services/data_processing/run_pipeline.py --split true` 并确认命名与用途
