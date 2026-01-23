"""
ChatTS-8B 调试对比测试脚本

用于验证不同配置下 ChatTS-8B 模型的输出质量差异。
此脚本帮助定位了 attn_implementation='eager' 导致输出乱码的问题。

使用方法：
    conda run -n chatts python tests/test_chatts_8b_compare.py
"""
import sys
sys.path.insert(0, '/home/douff/ilabel/check_outlier')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
import numpy as np
import types

MODEL_PATH = '/home/share/llm_models/bytedance-research/ChatTS-8B'
DEVICE = 'cuda:1'


def apply_monkey_patch(model):
    """应用 Monkey Patch 修复维度不匹配问题"""
    def find_target(m, depth=0):
        if depth > 10:
            return None
        if hasattr(m, '_update_causal_mask'):
            return m
        if hasattr(m, 'base_model'):
            r = find_target(m.base_model, depth + 1)
            if r:
                return r
        if hasattr(m, 'model') and m.model is not m:
            r = find_target(m.model, depth + 1)
            if r:
                return r
        return None

    target = find_target(model)
    if target and not hasattr(target, '_is_patched'):
        target._orig = target._update_causal_mask

        def patched(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
            if input_tensor.shape[1] > cache_position.shape[0]:
                cache_position = torch.arange(input_tensor.shape[1], device=input_tensor.device)
            return self._orig(attention_mask, input_tensor, cache_position, past_key_values, output_attentions)

        target._update_causal_mask = types.MethodType(patched, target)
        target._is_patched = True
        print(f'[Patch] Applied to {target.__class__.__name__}')
        return True
    return False


def load_model(use_4bit=False, use_eager_attention=False):
    """加载模型，可配置量化和注意力实现"""
    print(f'\n=== Loading model (4bit={use_4bit}, eager={use_eager_attention}) ===')

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "device_map": DEVICE,
    }

    if use_eager_attention:
        model_kwargs["attn_implementation"] = "eager"

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb_config
        # 4bit 需要用 device_map={"": index} 格式
        model_kwargs["device_map"] = {"": 1}

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, tokenizer=tokenizer)

    apply_monkey_patch(model)

    return model, tokenizer, processor


def run_inference(model, tokenizer, processor, ts):
    """运行推理"""
    prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n' \
             f'I have a time series of length {len(ts)}: <ts><ts/>. ' \
             'Please identify anomalies: anomalies = [{"range": [start, end]}]' \
             '<|im_end|><|im_start|>assistant\n'

    inputs = processor(text=[prompt], timeseries=[ts.astype(np.float32)], padding=True, return_tensors='pt')
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(DEVICE)
            if v.is_floating_point():
                inputs[k] = inputs[k].to(torch.bfloat16)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=150, use_cache=True)

    text = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    return text


def main():
    # 准备测试数据
    np.random.seed(42)
    ts = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1
    ts[50:55] = 5  # 添加异常

    print('=' * 60)
    print('ChatTS-8B 配置对比测试')
    print('=' * 60)
    print(f'测试数据: 100 点正弦波 + 随机噪声，异常区间 [50, 55]')

    results = {}

    # 测试1: 默认配置（无量化，无 eager）- 预期成功
    try:
        model, tokenizer, processor = load_model(use_4bit=False, use_eager_attention=False)
        output = run_inference(model, tokenizer, processor, ts)
        results['default'] = output
        print(f'\n[默认配置] 输出:\n{output[:300]}')
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        results['default'] = f'ERROR: {e}'
        print(f'\n[默认配置] 错误: {e}')

    # 测试2: 使用 eager attention - 预期失败（乱码）
    try:
        model, tokenizer, processor = load_model(use_4bit=False, use_eager_attention=True)
        output = run_inference(model, tokenizer, processor, ts)
        results['eager'] = output
        print(f'\n[eager attention] 输出:\n{output[:300]}')
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        results['eager'] = f'ERROR: {e}'
        print(f'\n[eager attention] 错误: {e}')

    # 测试3: 使用 4-bit 量化（无 eager）- 需要 Patch
    try:
        model, tokenizer, processor = load_model(use_4bit=True, use_eager_attention=False)
        output = run_inference(model, tokenizer, processor, ts)
        results['4bit'] = output
        print(f'\n[4-bit 量化] 输出:\n{output[:300]}')
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        results['4bit'] = f'ERROR: {e}'
        print(f'\n[4-bit 量化] 错误: {e}')

    print('\n' + '=' * 60)
    print('测试结论')
    print('=' * 60)
    print('- 默认配置: 应输出正确的异常检测结果')
    print('- eager attention: 应输出乱码（与 Qwen3 不兼容）')
    print('- 4-bit 量化: 可能正常或质量下降')


if __name__ == '__main__':
    main()
