ALTER TABLE annotation_records ADD COLUMN point_id TEXT;
ALTER TABLE annotation_segments ADD COLUMN point_id TEXT;
ALTER TABLE inference_results ADD COLUMN point_id TEXT;
ALTER TABLE review_queue ADD COLUMN point_id TEXT;
ALTER TABLE dataset_items ADD COLUMN point_id TEXT;

UPDATE annotation_records
SET point_id = source_id
WHERE (point_id IS NULL OR trim(point_id) = '')
  AND source_id IS NOT NULL
  AND trim(source_id) <> '';

UPDATE annotation_segments
SET point_id = source_id
WHERE (point_id IS NULL OR trim(point_id) = '')
  AND source_id IS NOT NULL
  AND trim(source_id) <> '';

UPDATE inference_results
SET point_id = point_name
WHERE (point_id IS NULL OR trim(point_id) = '')
  AND point_name IS NOT NULL
  AND trim(point_name) <> '';

UPDATE review_queue
SET point_id = COALESCE(NULLIF(trim(point_name), ''), NULLIF(trim(source_id), ''))
WHERE (point_id IS NULL OR trim(point_id) = '');

UPDATE dataset_items
SET point_id = point_name
WHERE (point_id IS NULL OR trim(point_id) = '')
  AND point_name IS NOT NULL
  AND trim(point_name) <> '';

CREATE INDEX IF NOT EXISTS idx_annotation_records_user_point
ON annotation_records (user_id, point_id);

CREATE INDEX IF NOT EXISTS idx_annotation_segments_user_point
ON annotation_segments (user_id, point_id);

CREATE INDEX IF NOT EXISTS idx_inference_results_point
ON inference_results (point_id);

CREATE INDEX IF NOT EXISTS idx_review_queue_point
ON review_queue (point_id);

CREATE INDEX IF NOT EXISTS idx_dataset_items_dataset_point
ON dataset_items (dataset_id, point_id);
