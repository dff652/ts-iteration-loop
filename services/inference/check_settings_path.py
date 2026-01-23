
import sys
import os
from pathlib import Path

# Add project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Project Root: {project_root}")

try:
    from configs.settings import settings
    print(f"Loaded settings from {settings}")
    print("-" * 30)
    print(f"USE_LOCAL_MODULES: {settings.USE_LOCAL_MODULES}")
    print(f"CHECK_OUTLIER_PATH: {settings.CHECK_OUTLIER_PATH}")
    print(f"LOCAL_CHECK_OUTLIER_PATH: {settings.LOCAL_CHECK_OUTLIER_PATH}")
    print(f"EXTERNAL_CHECK_OUTLIER_PATH: {settings.EXTERNAL_CHECK_OUTLIER_PATH}")
    print("-" * 30)
    
    if settings.CHECK_OUTLIER_PATH == str(project_root / "services/inference"):
        print("✅ VERIFIED: Using LOCAL path 'services/inference'")
    else:
        print(f"⚠️  WARNING: Using path '{settings.CHECK_OUTLIER_PATH}'")

except Exception as e:
    print(f"Error loading settings: {e}")
