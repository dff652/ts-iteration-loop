CREATE INDEX IF NOT EXISTS idx_tasks_type_status_created
ON tasks (type, status, created_at);

CREATE INDEX IF NOT EXISTS idx_review_queue_status_method_updated
ON review_queue (status, method, updated_at);

CREATE INDEX IF NOT EXISTS idx_inference_results_method_score_created
ON inference_results (method, score_avg, created_at);

CREATE INDEX IF NOT EXISTS idx_dataset_items_dataset_point
ON dataset_items (dataset_id, point_name);

CREATE INDEX IF NOT EXISTS idx_dataset_assets_name_type
ON dataset_assets (name, dataset_type);

CREATE INDEX IF NOT EXISTS idx_model_evals_model_created
ON model_evals (model_path, created_at);
