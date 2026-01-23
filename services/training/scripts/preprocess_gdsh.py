import json
import os
import csv
import sys

def preprocess_gdsh_data(json_path, ts_dir, output_path):
    print(f"Loading annotations from {json_path}...")
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Found {len(data)} samples. Processing...")
    
    processed_data = []
    success_count = 0
    missing_files = []

    for item in data:
        # 1. Parse Image Path to find CSV
        # format: /home/wyx/.../gdsh_second_2501FI10007.PV.jpg
        image_path = item.get('image', '')
        if not image_path:
            continue
            
        basename = os.path.basename(image_path)
        # Expected CSV name: replace extension .jpg with .csv
        # Note: some files might be .PV.jpg -> .PV.csv
        csv_filename = basename.replace('.jpg', '.csv')
        local_csv_path = os.path.join(ts_dir, csv_filename)

        # 2. Read Time Series Data
        if not os.path.exists(local_csv_path):
            # Try with prefix "数据集" seen in directory listing
            prefix_filename = "数据集" + csv_filename
            prefix_path = os.path.join(ts_dir, prefix_filename)
            if os.path.exists(prefix_path):
                local_csv_path = prefix_path
            else:
                missing_files.append(csv_filename)
                continue

        ts_values = []
        try:
            with open(local_csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Based on file preview: ,date,category,value
                    if 'value' in row:
                        try:
                            ts_values.append(float(row['value']))
                        except ValueError:
                            pass # Skip non-numeric
        except Exception as e:
            print(f"Error reading {local_csv_path}: {e}")
            continue
            
        if not ts_values:
            print(f"Warning: No valid data found in {csv_filename}")
            continue

        # 3. Construct ChatTS Format
        # Extract conversations
        conversations = item.get('conversations', [])
        prompt = ""
        response = ""
        
        for turn in conversations:
            if turn['from'] == 'user':
                # Replace <image> with <ts><ts/>
                prompt = turn['value'].replace('<image>', '<ts><ts/>')
            elif turn['from'] == 'assistant':
                response = turn['value']

        if not prompt or not response:
            continue

        # Create entry
        # timeseries format: list of lists (batch/channels x length). 
        # Here we have 1 channel (univariate).
        entry = {
            "input": prompt,
            "output": response,
            "timeseries": [ts_values] 
        }
        processed_data.append(entry)
        success_count += 1

    # 4. Save to JSONL
    print(f"Saving {len(processed_data)} processed samples to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print("Done.")
    print(f"Success: {success_count}/{len(data)}")
    if missing_files:
        print(f"Missing {len(missing_files)} CSV files (first 5): {missing_files[:5]}")

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    json_source = os.path.join(base_dir, "data/chatts_tune/json/gdsh.json")
    ts_source_dir = os.path.join(base_dir, "data/chatts_tune/timeseries/gdsh")
    output_target = os.path.join(base_dir, "data/chatts_tune/train.jsonl")

    preprocess_gdsh_data(json_source, ts_source_dir, output_target)
