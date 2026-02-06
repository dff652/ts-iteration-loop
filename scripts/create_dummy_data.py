
import os
import sys
import json
import csv
import random
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import settings
from src.db.database import SessionLocal, InferenceResult

def create_dummy_data():
    print(f"Project Root: {PROJECT_ROOT}")
    
    # User and directories
    # Create for both 'douff' (default) and 'User' (fallback/frontend default)
    target_users = ["douff", "User"]
    
    # 1. Prepare Directories
    # Create a dummy method dir in inference dir
    inference_dir = Path(settings.DATA_INFERENCE_DIR) / "dummy_test"
    inference_dir.mkdir(parents=True, exist_ok=True)
    
    for user in target_users:
        # User annotation dir
        annotation_dir = Path(settings.ANNOTATIONS_ROOT) / user
        annotation_dir.mkdir(parents=True, exist_ok=True)
        print(f"Annotation Dir for {user}: {annotation_dir}")

    print(f"Inference Dir: {inference_dir}")

    # 2. Define Dummy Files
    # We will create 3 files with different scores to test sorting/filtering
    configs = [
        {"name": "dummy_high_score", "score": 0.95},
        {"name": "dummy_mid_score", "score": 0.55},
        {"name": "dummy_low_score", "score": 0.15},
    ]

    db = SessionLocal()

    for cfg in configs:
        base_name = f"{cfg['name']}_{cfg['score']}"
        csv_filename = f"{base_name}.csv"
        json_filename = f"{base_name}.json"
        
        csv_path = inference_dir / csv_filename
        
        # 3. Create CSV File (Shared)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "value", "label"])
            # Write some dummy points
            for i in range(100):
                writer.writerow([i, random.random(), 0])
        print(f"Created CSV: {csv_path}")
        
        # 4. Create JSON Annotation File (For each user)
        # Format: {"annotations": [{"label": {"id": "test", "color": "#000"}, "segments": [{"start": 10, "end": 20}]}]}
        ann_content = {
            "annotations": [
                {
                    "label": {"id": "test_label", "text": "Test Label", "color": "#ff0000"},
                    "segments": [
                        {"start": 10, "end": 20},
                        {"start": 50, "end": 60}
                    ],
                    "prompt": "Auto generated dummy annotation"
                }
            ]
        }
        
        for user in target_users:
            user_ann_dir = Path(settings.ANNOTATIONS_ROOT) / user
            user_json_path = user_ann_dir / json_filename
            with open(user_json_path, 'w') as f:
                json.dump(ann_content, f, indent=2)
            print(f"Created JSON for {user}: {user_json_path}")
        
        # 5. Insert into Database (InferenceResult)
        # Check if exists first
        existing = db.query(InferenceResult).filter(InferenceResult.result_path == str(csv_path)).first()
        if existing:
            print(f"Updating existing DB record for {csv_filename}")
            existing.score_avg = cfg['score']
            existing.score_max = min(1.0, cfg['score'] + 0.05)
            existing.method = "dummy_test"
            existing.meta = json.dumps({"description": "dummy data for testing"})
        else:
            print(f"Inserting new DB record for {csv_filename}")
            record = InferenceResult(
                id=f"dummy_{base_name}",
                task_id="dummy_task_001",
                point_name=base_name,
                result_path=str(csv_path),
                method="dummy_test",
                score_avg=cfg['score'],
                score_max=min(1.0, cfg['score'] + 0.05),
                segment_count=2,
                meta=json.dumps({"description": "dummy data for testing"}),
                created_at=datetime.utcnow()
            )
            db.add(record)
    
    db.commit()
    db.close()
    print("Done! Dummy data created.")

if __name__ == "__main__":
    create_dummy_data()
