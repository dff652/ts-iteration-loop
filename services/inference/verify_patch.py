
import sys
import os
from pathlib import Path

# Add project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

print("Attempting to import patch_transformers...")
try:
    import services.inference.patch_transformers
    print("✅ Successfully imported patch_transformers")
    
    from transformers.cache_utils import DynamicCache
    if hasattr(DynamicCache, 'seen_tokens'):
        print("✅ DynamicCache.seen_tokens exists (Patched successfully)")
    else:
        print("❌ DynamicCache.seen_tokens MISSING (Patch failed)")

except Exception as e:
    print(f"❌ Failed to import/patch: {e}")
    import traceback
    traceback.print_exc()
