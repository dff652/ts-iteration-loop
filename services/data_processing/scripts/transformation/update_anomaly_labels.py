#!/usr/bin/env python3
import json

def update_anomaly_labels(input_file: str, output_file: str):
    """Update anomaly labels from 点异常 to 尖峰"""
    
    # Read input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each entry
    for entry in data:
        if "conversations" in entry:
            for conv in entry["conversations"]:
                if conv.get("from") == "assistant":
                    # Parse the anomalies
                    value = conv.get("value", "")
                    
                    try:
                        anomaly_data = json.loads(value)
                        
                        if "detected_anomalies" in anomaly_data:
                            for anomaly in anomaly_data["detected_anomalies"]:
                                # Replace type
                                if anomaly.get("type") == "点异常":
                                    anomaly["type"] = "尖峰"
                                
                                # Replace in reason field
                                if "reason" in anomaly:
                                    reason = anomaly["reason"]
                                    # Replace various forms
                                    reason = reason.replace("点异常", "尖峰")
                                    reason = reason.replace("孤立异常点", "尖峰点")
                                    reason = reason.replace("存在多个孤立异常点", "存在多个尖峰点")
                                    anomaly["reason"] = reason
                        
                        # Update the assistant value
                        conv["value"] = json.dumps(anomaly_data, ensure_ascii=False)
                    
                    except json.JSONDecodeError:
                        continue
    
    # Write output JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Updated annotations saved to: {output_file}")

if __name__ == "__main__":
    input_file = "/home/douff/converted_annotations/merged_annotations_modified.json"
    output_file = "/home/douff/converted_annotations/merged_annotations_modified.json"
    
    update_anomaly_labels(input_file, output_file)
