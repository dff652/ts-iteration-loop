import json
import numpy as np

file_path = 'data/chatts_tune/train.jsonl'

def check_data_quality(file_path):
    print(f"Checking {file_path}...\n")
    
    total_count = 0
    valid_structure_count = 0
    valid_content_count = 0
    ts_lengths = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total_count += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Error] Line {i+1}: Invalid JSON.")
                continue
                
            # 1. Check Key Existence
            if 'input' in entry and 'output' in entry and 'timeseries' in entry:
                valid_structure_count += 1
            else:
                print(f"[Error] Line {i+1}: Missing keys. Keys found: {list(entry.keys())}")
                continue
                
            # 2. Check Content
            input_text = entry['input']
            output_text = entry['output']
            timeseries = entry['timeseries']
            
            # Check for placeholder
            if '<ts><ts/>' not in input_text:
                print(f"[Warning] Line {i+1}: 'input' field missing <ts><ts/> placeholder.")
            
            # Check timeseries format
            if not isinstance(timeseries, list) or len(timeseries) == 0:
                 print(f"[Error] Line {i+1}: 'timeseries' is empty or not a list.")
                 continue
            
            # Since it's univariate, it should be a list of lists like [[...]]
            if isinstance(timeseries[0], list):
                ts_len = len(timeseries[0])
            else:
                # If it's a flat list [1,2,3], note it (ChatTS usually expects [[...]] for channels)
                # But our preprocessor produces [[...]], so let's verify.
                print(f"[Warning] Line {i+1}: 'timeseries' format might be flat list instead of list of lists.")
                ts_len = len(timeseries)

            ts_lengths.append(ts_len)
            valid_content_count += 1

            # Print sample for the first 2 entries
            if i < 2:
                print(f"--- Sample {i+1} ---")
                print(f"Input (truncated): {input_text[:100]}...")
                print(f"Output (truncated): {output_text[:100]}...")
                print(f"Timeseries Shape: {len(timeseries)} channel(s), Length: {ts_len}")
                print(f"First 5 values: {timeseries[0][:5]}")
                print("------------------\n")

    print(f"Total Lines: {total_count}")
    print(f"Valid Structure: {valid_structure_count}")
    print(f"Valid Content: {valid_content_count}")
    
    if ts_lengths:
        print(f"\nTimeseries Length Stats:")
        print(f"  Min Length: {min(ts_lengths)}")
        print(f"  Max Length: {max(ts_lengths)}")
        print(f"  Avg Length: {sum(ts_lengths) / len(ts_lengths):.2f}")
    else:
        print("\nNo valid timeseries found.")

if __name__ == "__main__":
    check_data_quality(file_path)
