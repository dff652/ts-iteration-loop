ALTER TABLE dataset_assets ADD COLUMN owner_id TEXT;
ALTER TABLE dataset_assets ADD COLUMN org_id TEXT;
ALTER TABLE dataset_assets ADD COLUMN created_by TEXT;
ALTER TABLE dataset_assets ADD COLUMN updated_by TEXT;

UPDATE dataset_assets
SET owner_id = COALESCE(NULLIF(trim(owner_id), ''), 'default')
WHERE owner_id IS NULL OR trim(owner_id) = '';

UPDATE dataset_assets
SET org_id = COALESCE(NULLIF(trim(org_id), ''), 'default')
WHERE org_id IS NULL OR trim(org_id) = '';

UPDATE dataset_assets
SET created_by = COALESCE(NULLIF(trim(created_by), ''), owner_id, 'default')
WHERE created_by IS NULL OR trim(created_by) = '';

UPDATE dataset_assets
SET updated_by = COALESCE(NULLIF(trim(updated_by), ''), created_by, owner_id, 'default')
WHERE updated_by IS NULL OR trim(updated_by) = '';

CREATE INDEX IF NOT EXISTS idx_dataset_assets_owner_org
ON dataset_assets (owner_id, org_id);

CREATE INDEX IF NOT EXISTS idx_dataset_assets_owner_org_name
ON dataset_assets (owner_id, org_id, name);
