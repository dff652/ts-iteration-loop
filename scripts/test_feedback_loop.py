"""
集成测试脚本：验证 Phase 3 反馈回路
流程：模拟推理任务完成 -> 触发自动反馈 -> 检查标注工具模拟端接收情况
"""
import json
import asyncio
from pathlib import Path
import httpx
import pytest
from unittest.mock import MagicMock, patch

# 模拟数据
MOCK_INFERENCE_RESULT = {
    "success": True,
    "results": [
        {
            "file": "test_data_001.csv",
            "success": True,
            "result": {
                "detected_anomalies": [
                    {"type": "点异常", "interval": [100, 105], "reason": "突发峰值"},
                    {"type": "趋势异常", "interval": [500, 600], "reason": "持续漂移"}
                ]
            }
        }
    ],
    "total": 1,
    "successful": 1
}

async def test_feedback_loop_logic():
    """测试结果转换与导入逻辑"""
    from src.adapters.check_outlier import CheckOutlierAdapter
    from src.api.annotation import import_inference_results_internal
    
    adapter = CheckOutlierAdapter()
    
    # 1. 测试转换逻辑
    print("Testing result conversion...")
    temp_file = adapter.convert_to_annotation_format(MOCK_INFERENCE_RESULT)
    assert Path(temp_file).exists()
    
    with open(temp_file, "r", encoding="utf-8") as f:
        converted_data = json.load(f)
    
    assert len(converted_data) == 1
    assert converted_data[0]["filename"] == "test_data_001.csv"
    assert len(converted_data[0]["annotations"]) == 2
    assert converted_data[0]["source"] == "inference"
    
    # 2. 测试导入逻辑 (使用 Mock 模拟 API 调用)
    print("Testing annotation import (mocked)...")
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        
        result = await import_inference_results_internal(temp_file)
        
        assert result["success"] is True
        assert result["count"] == 1
        assert mock_post.called

if __name__ == "__main__":
    print("Starting integration logic test...")
    asyncio.run(test_feedback_loop_logic())
    print("Test passed!")
