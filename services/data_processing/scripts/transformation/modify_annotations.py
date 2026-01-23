#!/usr/bin/env python3
import json
import os
from typing import List, Dict, Any

def extract_point_name_from_image(image_path: str) -> str:
    """Extract point name from image path like '/path/数据集zhlh_100_XXX.jpg' -> 'zhlh_100_XXX'"""
    basename = os.path.basename(image_path)
    # Remove '数据集' prefix and '.jpg' suffix
    if basename.startswith('数据集'):
        basename = basename[3:]  # Remove '数据集'
    # Remove extension
    point_name = os.path.splitext(basename)[0]
    return point_name

def update_image_path(original_path: str, picture_dir: str) -> str:
    """Update image path based on point name"""
    point_name = extract_point_name_from_image(original_path)
    new_path = os.path.join(picture_dir, f"{point_name}.jpg")
    return new_path

def expand_single_point_anomaly(interval: List[int], anomaly_type: str) -> List[int]:
    """Expand single-point anomaly intervals by 5 indices on each side"""
    if anomaly_type == "点异常" and interval[0] == interval[1]:
        # Single point anomaly, expand by 5 on each side
        start = max(0, interval[0] - 5)
        end = interval[1] + 5
        return [start, end]
    return interval

def parse_anomalies(assistant_value: str) -> List[Dict[str, Any]]:
    """Parse the concatenated JSON anomalies from assistant value"""
    anomalies = []
    
    # Split by "},{"  to separate multiple JSON objects
    # The value is multiple JSON objects concatenated like: {...},{...},{...}
    parts = assistant_value.split("},{")
    
    for i, part in enumerate(parts):
        # Reconstruct proper JSON by adding back the brace
        if i == 0:
            json_str = part + "}"
        elif i == len(parts) - 1:
            json_str = "{" + part
        else:
            json_str = "{" + part + "}"
        
        try:
            anomaly_obj = json.loads(json_str)
            if anomaly_obj.get("status") == "success":
                detected = anomaly_obj.get("detected_anomalies", [])
                anomalies.extend(detected)
        except json.JSONDecodeError:
            continue
    
    return anomalies

def modify_annotations(input_file: str, output_file: str, picture_dir: str):
    """Modify annotations according to requirements"""
    
    # Read input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each entry
    for entry in data:
        # 1. Update image path
        if "image" in entry:
            entry["image"] = update_image_path(entry["image"], picture_dir)
        
        # Process conversations
        if "conversations" in entry:
            for conv in entry["conversations"]:
                if conv.get("from") == "assistant":
                    # Parse the anomalies
                    value = conv.get("value", "")
                    anomalies = parse_anomalies(value)
                    
                    # 2. Expand single-point anomalies
                    modified_anomalies = []
                    has_no_anomaly = False
                    
                    for anomaly in anomalies:
                        anomaly_type = anomaly.get("type", "")
                        
                        # Check if this is "无异常" type
                        if anomaly_type == "无异常":
                            has_no_anomaly = True
                            continue
                        
                        interval = anomaly.get("interval", [])
                        if len(interval) == 2:
                            expanded_interval = expand_single_point_anomaly(interval, anomaly_type)
                            anomaly["interval"] = expanded_interval
                        
                        modified_anomalies.append(anomaly)
                    
                    # 3. Add "无异常" entry with interval [0, 4999]
                    if not has_no_anomaly:
                        no_anomaly_entry = {
                            "interval": [0, 4999],
                            "type": "无异常",
                            "reason": "该区间内无检测到异常"
                        }
                        modified_anomalies.append(no_anomaly_entry)
                    
                    # Reconstruct the assistant value
                    new_value = json.dumps({
                        "status": "success",
                        "detected_anomalies": modified_anomalies
                    }, ensure_ascii=False)
                    
                    conv["value"] = new_value
    
    # Write output JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Modified annotations saved to: {output_file}")

if __name__ == "__main__":
    input_file = "/home/douff/converted_annotations/merged_annotations.json"
    output_file = "/home/douff/converted_annotations/merged_annotations_modified.json"
    picture_dir = "/home/douff/数据标注/data/picture_data"
    
    modify_annotations(input_file, output_file, picture_dir)
