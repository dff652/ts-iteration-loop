#!/usr/bin/env python3
"""
验证 ts_downsample 返回值修改是否正确
"""
import re
import sys

def check_file_for_pattern(filepath, pattern, description):
    """检查文件中是否包含特定模式"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        matches = re.findall(pattern, content)
        if matches:
            print(f"✓ {filepath}: {description}")
            print(f"  找到 {len(matches)} 处匹配")
            return True
        else:
            print(f"✗ {filepath}: {description}")
            print(f"  未找到匹配")
            return False
    except Exception as e:
        print(f"✗ {filepath}: 读取文件失败 - {e}")
        return False

def main():
    print("=" * 70)
    print("验证 ts_downsample 返回值修改")
    print("=" * 70)
    
    checks = [
        # signal_processing.py
        ("/home/douff/ilabel/check_outlier/signal_processing.py",
         r"return downsampled_data, downsampled_time, position_index",
         "ts_downsample 返回三个值"),
        
        # chatts_detect.py
        ("/home/douff/ilabel/check_outlier/chatts_detect.py",
         r"downsampled_data,_,\s*position_index = ts_downsample",
         "chatts_detect 正确处理三个返回值"),
        
        # mean_shift_detect.py
        ("/home/douff/ilabel/check_outlier/mean_shift_detect.py",
         r"downsampled_data,\s*ts,\s*_ = ts_downsample",
         "mean_shift_detect 正确处理三个返回值"),
        
        # adtk_hbos_detect.py
        ("/home/douff/ilabel/check_outlier/adtk_hbos_detect.py",
         r"downsampled_data,\s*ts,\s*_ = ts_downsample",
         "adtk_hbos_detect 正确处理三个返回值"),
        
        # run.py
        ("/home/douff/ilabel/check_outlier/run.py",
         r"downsampled_data,\s*ts,\s*position_index = adaptive_downsample",
         "run.py 正确处理 adaptive_downsample 三个返回值"),
        
        # jm_detect.py
        ("/home/douff/ilabel/check_outlier/jm_detect.py",
         r"downsampled_data,\s*ts,\s*position_index = adaptive_downsample",
         "jm_detect 正确处理 adaptive_downsample 三个返回值"),
    ]
    
    results = []
    for filepath, pattern, description in checks:
        result = check_file_for_pattern(filepath, pattern, description)
        results.append(result)
        print()
    
    print("=" * 70)
    print("验证总结")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✓ 所有修改验证通过！")
        return 0
    else:
        print("✗ 部分修改验证失败，请检查")
        return 1

if __name__ == "__main__":
    sys.exit(main())

