-- 0005_add_model_eval_fields.sql
ALTER TABLE model_evals ADD COLUMN task_id VARCHAR(36);
ALTER TABLE model_evals ADD COLUMN model_family VARCHAR(50) DEFAULT 'chatts';
ALTER TABLE model_evals ADD COLUMN dataset_id VARCHAR(36);
