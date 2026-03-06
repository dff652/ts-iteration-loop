-- 0007_add_task_center_step_dependencies.sql

ALTER TABLE task_step_runs ADD COLUMN depends_on TEXT;
