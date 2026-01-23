
import sys

def apply_loss_kwargs_patch():
    # 尝试导入 transformers.utils 以检查 LossKwargs
    try:
        import transformers.utils
        if not hasattr(transformers.utils, "LossKwargs"):
            from typing import TypedDict
            
            # 定义为 TypedDict 以避免 TypeError: cannot inherit from both a TypedDict type and a non-TypedDict base class
            class LossKwargs(TypedDict):
                loss: float
                
            transformers.utils.LossKwargs = LossKwargs
            sys.modules["transformers.utils.LossKwargs"] = LossKwargs
            print("[Patch] Applied transformers.utils.LossKwargs patch (TypedDict)")
    except ImportError:
        pass
    except Exception as e:
        print(f"[Patch] Failed to apply LossKwargs patch: {e}")

# 在模块加载时自动应用
apply_loss_kwargs_patch()
