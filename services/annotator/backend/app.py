import os
import json
import time
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
from tsdownsample import M4Downsampler

# Import authentication module
from auth import login_required, verify_password, generate_token, load_users

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='')
CORS(app)

# Configuration directories
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
ANNOTATIONS_DIR = os.path.join(BASE_DIR, 'annotations')
LABELS_FILE = os.path.join(BASE_DIR, 'config', 'labels.json')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# Current working data path
CURRENT_DATA_PATH = DATA_DIR


# ==================== Static Files ====================
@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory(app.static_folder, 'index.html')


# ==================== Path Management (User-specific) ====================
@app.route('/api/set-path', methods=['POST'])
@login_required
def set_data_path(current_user):
    """Set custom data directory path for current user"""
    try:
        from auth import load_users, save_users
        
        data = request.get_json()
        path = data.get('path', '')
        
        if path and os.path.isdir(path):
            # Save path to user config
            users = load_users()
            if current_user in users:
                users[current_user]['data_path'] = path
                save_users(users)
            
            return jsonify({'success': True, 'path': path})
        else:
            return jsonify({'success': False, 'error': 'Invalid directory path'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/current-path', methods=['GET'])
@login_required
def get_current_path(current_user):
    """Get current data directory path for current user"""
    try:
        from auth import load_users
        
        users = load_users()
        user_path = users.get(current_user, {}).get('data_path', DATA_DIR)
        
        return jsonify({'success': True, 'path': user_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/browse-dir', methods=['GET'])
def browse_directory():
    """Browse server directory structure"""
    try:
        # Default to /home instead of user home directory
        path = request.args.get('path', '/home')
        
        if not os.path.exists(path):
            return jsonify({'success': False, 'error': 'Path does not exist'}), 404
        
        if not os.path.isdir(path):
            path = os.path.dirname(path)
        
        path = os.path.abspath(path)
        parent_path = os.path.dirname(path)
        
        directories = []
        try:
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                try:
                    if os.path.isdir(item_path):
                        has_data_files = False
                        try:
                            for f in os.listdir(item_path):
                                if f.endswith(('.csv', '.xls', '.xlsx')):
                                    has_data_files = True
                                    break
                        except PermissionError:
                            pass
                        
                        directories.append({
                            'name': item,
                            'path': item_path,
                            'is_dir': True,
                            'has_data_files': has_data_files
                        })
                except PermissionError:
                    continue
        except PermissionError:
            return jsonify({'success': False, 'error': 'Permission denied'}), 403
        
        current_has_data = any(
            f.endswith(('.csv', '.xls', '.xlsx')) 
            for f in os.listdir(path) 
            if os.path.isfile(os.path.join(path, f))
        )
        
        return jsonify({
            'success': True,
            'current_path': path,
            'parent_path': parent_path if parent_path != path else None,
            'directories': directories,
            'has_data_files': current_has_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== File Management (User-specific path) ====================
@app.route('/api/files', methods=['GET'])
@login_required
def get_files(current_user):
    """Get all CSV and Excel files in current user's directory"""
    try:
        from auth import load_users
        
        # Get user's data path
        users = load_users()
        user_path = users.get(current_user, {}).get('data_path', DATA_DIR)
        
        print(f"=== get_files for user: {current_user} ===")
        print(f"User path: {user_path}")
        print(f"Path exists: {os.path.exists(user_path)}")
        print(f"Is directory: {os.path.isdir(user_path)}")
        
        if not os.path.exists(user_path):
            return jsonify({'success': False, 'error': 'Path does not exist'}), 404
        
        all_items = os.listdir(user_path)
        print(f"All items in directory: {all_items}")
        
        files = []
        for f in all_items:
            full_path = os.path.join(user_path, f)
            print(f"Checking: {f}, is_file: {os.path.isfile(full_path)}")
            
            if os.path.isfile(full_path) and f.endswith(('.csv', '.xls', '.xlsx')):
                print(f"  -> Matched: {f}")
                # Check for annotations in user's annotation directory
                user_ann_dir = os.path.join(ANNOTATIONS_DIR, current_user)
                
                # Try multiple annotation file name patterns (ordered by frequency)
                annotation_file = None
                annotation_count = 0
                has_annotations = False
                
                # Pattern 5: Direct replacement .csv -> .json (most common for current files)
                # This matches files like: 数据集zhlh_100_XXX.PV.json for CSV files like: 数据集zhlh_100_XXX.PV.csv
                pattern5 = os.path.join(user_ann_dir, f.replace('.csv', '.json'))
                # Pattern 1: filename.csv.json (standard auto-save format)
                pattern1 = os.path.join(user_ann_dir, f"{f}.json")
                # Pattern 4: 数据集filename.json (for CSV files without 数据集 prefix)
                # This matches files like: 数据集zhlh_100_XXX.json for CSV files like: zhlh_100_XXX.csv
                pattern4 = os.path.join(user_ann_dir, f"数据集{f.replace('.csv', '')}.json")
                # Pattern 3: annotations_filename.json (old export without 数据集)
                pattern3 = os.path.join(user_ann_dir, f"annotations_{f.replace('.csv', '')}.json")
                # Pattern 2: annotations_数据集filename.json (old export format)
                pattern2 = os.path.join(user_ann_dir, f"annotations_数据集{f.replace('.csv', '')}.json")
                
                for pattern in [pattern5, pattern1, pattern4, pattern3, pattern2]:
                    if os.path.exists(pattern):
                        annotation_file = pattern
                        print(f"  -> Found annotation: {os.path.basename(pattern)}")
                        break
                
                if annotation_file and os.path.exists(annotation_file):
                    try:
                        with open(annotation_file, 'r', encoding='utf-8') as af:
                            ann_data = json.load(af)
                            # Support both formats
                            annotations = ann_data.get('annotations', [])
                            
                            # Has annotations if there are any annotations (including no-label)
                            if annotations:
                                has_annotations = True
                            
                            # Only count annotations with segments (exclude no-label annotations)
                            annotations_with_segments = [
                                ann for ann in annotations 
                                if ann.get('segments') and len(ann.get('segments', [])) > 0
                            ]
                            annotation_count = len(annotations_with_segments)
                    except Exception as e:
                        print(f"  -> Error reading annotation: {e}")
                        pass
                
                files.append({
                    'name': f,
                    'has_annotations': has_annotations,
                    'annotation_count': annotation_count
                })
        
        print(f"Matched files: {[f['name'] for f in files]}")
        
        return jsonify({
            'success': True,
            'files': files,
            'path': user_path
        })
    except Exception as e:
        print(f"Error in get_files: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/data/<filename>', methods=['GET'])
@login_required
def get_data(filename, current_user):
    """Read CSV or Excel file data with smart column detection and M4 downsampling"""
    try:
        from auth import load_users
        
        # Get user's data path
        users = load_users()
        user_path = users.get(current_user, {}).get('data_path', DATA_DIR)
        
        filepath = os.path.join(user_path, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(filepath)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath, engine='openpyxl' if file_ext == '.xlsx' else None)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file format'}), 400
        
        original_len = len(df)
        columns = df.columns.tolist()
        
        # ============ Smart Column Detection (by data type) ============
        time_col = None
        val_col = None
        series_col = None
        label_col = None
        
        for col in columns:
            # Skip unnamed/index columns
            if col == '' or str(col).startswith('Unnamed'):
                continue
            
            # Detect datetime column
            if time_col is None:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    time_col = col
                elif df[col].dtype == 'object':
                    # Try parsing first few non-null values as datetime
                    sample = df[col].dropna().head(5)
                    if len(sample) > 0:
                        try:
                            pd.to_datetime(sample, errors='raise')
                            time_col = col
                        except:
                            pass
            
            # Detect numeric columns for value
            if pd.api.types.is_numeric_dtype(df[col]):
                if val_col is None:
                    val_col = col
            
            # Detect string columns that might be series/label
            elif df[col].dtype == 'object' and col != time_col:
                unique_count = df[col].nunique()
                if unique_count <= 10 and series_col is None:
                    series_col = col
                elif label_col is None:
                    label_col = col
        
        # Fallback: use second column as value if not detected
        if val_col is None and len(columns) >= 2:
            for col in columns:
                if col != time_col and not str(col).startswith('Unnamed'):
                    val_col = col
                    break
        
        # Fallback: use first column as value for single-column files
        if val_col is None and len(columns) >= 1:
            val_col = columns[-1]  # Use last column
        
        print(f"[Column Detection] time={time_col}, value={val_col}, series={series_col}, label={label_col}")
        
        # ============ Parse Time Column ============
        use_index_mode = True  # Default to index mode
        if time_col is not None:
            try:
                # Use pandas to parse datetime with timezone support
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
                valid_times = df[time_col].notna().sum()
                if valid_times > 0:
                    use_index_mode = False
                    print(f"[Time Parse] Successfully parsed {valid_times}/{len(df)} time values")
            except Exception as e:
                print(f"[Time Parse] Failed: {e}, falling back to index mode")
        
        # ============ Prepare Value Array (handle NaN) ============
        if val_col is not None:
            df[val_col] = pd.to_numeric(df[val_col], errors='coerce').fillna(0.0)
        else:
            return jsonify({'success': False, 'error': 'No numeric value column found'}), 400
        
        # ============ M4 Downsampling (preserves min/max) ============
        MAX_ROWS = 10000
        downsampled = False
        if original_len > MAX_ROWS:
            try:
                # Prepare arrays for M4 downsampler
                x_arr = np.arange(original_len, dtype=np.float64)
                y_arr = df[val_col].values.astype(np.float64)
                
                # M4 returns indices that preserve local min/max
                downsampler = M4Downsampler()
                indices = downsampler.downsample(x_arr, y_arr, n_out=MAX_ROWS)
                
                df = df.iloc[indices].reset_index(drop=True)
                downsampled = True
                print(f"[M4 Downsampling] {original_len} rows -> {len(df)} rows")
            except Exception as e:
                print(f"[M4 Downsampling] Error: {e}, using uniform sampling fallback")
                step = original_len // MAX_ROWS
                df = df.iloc[::step].reset_index(drop=True)
                downsampled = True
        
        # ============ Build Response Data ============
        data = []
        series_set = set()
        
        for idx, row in df.iterrows():
            # Use row index as x-axis value
            index_value = idx
            
            # Value
            val_value = float(row[val_col]) if pd.notna(row[val_col]) else 0.0
            
            # Series
            if series_col and pd.notna(row[series_col]):
                series_value = str(row[series_col])
            else:
                series_value = val_col if val_col else 'value'
            series_set.add(series_value)
            
            # Label
            if label_col and pd.notna(row[label_col]):
                label_value = str(row[label_col])
            else:
                label_value = ''
            
            data.append({
                'idx': index_value,           # Numeric index for X-axis
                'time': index_value,          # Keep as index for D3 compatibility
                'val': val_value,
                'series': series_value,
                'label': label_value
            })
        
        return jsonify({
            'success': True,
            'filename': filename,
            'columns': columns,
            'data': data,
            'seriesList': list(series_set),
            'labelList': [],
            'useIndexMode': True,           # Always use index mode for X-axis
            'originalLength': original_len,
            'downsampled': downsampled,
            'detectedColumns': {
                'time': time_col,
                'value': val_col,
                'series': series_col,
                'label': label_col
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


# ==================== User Authentication ====================
@app.route('/api/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password required'}), 400
        
        if verify_password(username, password):
            token = generate_token(username)
            users = load_users()
            user_info = users.get(username, {})
            
            return jsonify({
                'success': True,
                'token': token,
                'username': username,
                'name': user_info.get('name', username)
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/user', methods=['GET'])
@login_required
def get_current_user(current_user):
    """Get current logged in user info"""
    try:
        users = load_users()
        user_info = users.get(current_user, {})
        return jsonify({
            'success': True,
            'username': current_user,
            'name': user_info.get('name', current_user)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== Annotations (Multi-user Support) ====================
@app.route('/api/annotations/<filename>', methods=['GET'])
@login_required
def get_annotations(filename, current_user):
    """Get annotations for a file (user-specific)"""
    try:
        # User-specific annotation directory
        user_ann_dir = os.path.join(ANNOTATIONS_DIR, current_user)
        os.makedirs(user_ann_dir, exist_ok=True)
        
        # Try multiple patterns to find the annotation file
        annotation_file = None
        annotation_data = None
        
        # Pattern 1: Standard format (filename.json)
        pattern1 = os.path.join(user_ann_dir, f"{filename}.json")
        if os.path.exists(pattern1):
            annotation_file = pattern1
        
        # Pattern 2 & 3: Search all JSON files and match by filename field
        if not annotation_file:
            for json_file in os.listdir(user_ann_dir):
                if not json_file.endswith('.json'):
                    continue
                json_path = os.path.join(user_ann_dir, json_file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Check if this JSON's filename matches
                        if data.get('filename') == filename:
                            annotation_file = json_path
                            annotation_data = data
                            break
                except:
                    continue
        
        if not annotation_file:
            return jsonify({
                'success': True,
                'filename': filename,
                'annotations': []
            })
        
        # Load data if not already loaded
        if not annotation_data:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'annotations': annotation_data.get('annotations', []),
            'overall_attribute': annotation_data.get('overall_attribute', {})
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/annotations/<filename>', methods=['POST'])
@login_required
def save_annotations(filename, current_user):
    """Save annotations for a file (user-specific directory)"""
    try:
        from flask import request
        from auth import load_users
        
        data = request.get_json()
        
        # Support both old and new formats
        # New format: {filename, overall_attribute, annotations, export_time}
        # Old format: {annotations, overall_attributes}
        if 'filename' in data:
            # New unified format
            save_data = data
        else:
            # Old format - convert to new format
            save_data = {
                'filename': filename,
                'overall_attribute': data.get('overall_attributes', {}),
                'annotations': data.get('annotations', []),
                'export_time': datetime.now().isoformat()
            }
        
        # Save to user's annotation directory
        user_ann_dir = os.path.join(ANNOTATIONS_DIR, current_user)
        os.makedirs(user_ann_dir, exist_ok=True)
        
        annotation_file = os.path.join(user_ann_dir, f"{filename}.json")
        
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Annotations saved for {filename}'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/annotations/<filename>', methods=['DELETE'])
def delete_annotation(filename):
    """Delete a specific annotation"""
    try:
        data = request.get_json()
        annotation_id = data.get('annotation_id')
        
        if not annotation_id:
            return jsonify({'success': False, 'error': 'Annotation ID is required'}), 400
        
        annotation_file = os.path.join(ANNOTATIONS_DIR, f"{filename}.json")
        
        if not os.path.exists(annotation_file):
            return jsonify({'success': False, 'error': 'Annotation file not found'}), 404
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = data.get('annotations', [])
        annotations = [a for a in annotations if a.get('id') != annotation_id]
        
        data['annotations'] = annotations
        data['last_updated'] = datetime.now().isoformat()
        
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True, 'message': 'Annotation deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download-annotations/<filename>', methods=['GET'])
def download_annotations(filename):
    """Download annotations in target JSON format"""
    try:
        annotation_file = os.path.join(ANNOTATIONS_DIR, f"{filename}.json")
        
        if not os.path.exists(annotation_file):
            return jsonify({
                'annotations': [],
                'export_time': datetime.now().isoformat(),
                'filename': filename
            })
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format for export
        export_annotations = []
        for ann in data.get('annotations', []):
            export_ann = {
                'categories': ann.get('categories', {}),
                'local_change': ann.get('local_change', {})
            }
            export_annotations.append(export_ann)
        
        return jsonify({
            'annotations': export_annotations,
            'export_time': datetime.now().isoformat(),
            'filename': filename
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/annotations/all', methods=['GET'])
@login_required
def get_all_annotations(current_user):
    """Get all annotations for current user (for training data export)"""
    try:
        user_ann_dir = os.path.join(ANNOTATIONS_DIR, current_user)
        
        if not os.path.exists(user_ann_dir):
            return jsonify({
                'success': True,
                'annotations': [],
                'count': 0
            })
        
        all_annotations = []
        
        for json_file in os.listdir(user_ann_dir):
            if not json_file.endswith('.json'):
                continue
            
            json_path = os.path.join(user_ann_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 确保包含必要字段
                    annotation_entry = {
                        'filename': data.get('filename', json_file.replace('.json', '')),
                        'annotations': data.get('annotations', []),
                        'overall_attribute': data.get('overall_attribute', {}),
                        'export_time': data.get('export_time', '')
                    }
                    all_annotations.append(annotation_entry)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'annotations': all_annotations,
            'count': len(all_annotations),
            'user': current_user
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== Label Configuration ====================
@app.route('/api/labels', methods=['GET'])
def get_labels():
    """Get label configuration"""
    try:
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            return jsonify({'success': True, 'labels': labels})
        else:
            return jsonify({
                'success': True,
                'labels': {
                    'overall_attribute': {},
                    'local_change': {},
                    'custom_labels': []
                }
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/labels', methods=['POST'])
def save_labels():
    """Save label configuration"""
    try:
        data = request.get_json()
        
        with open(LABELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True, 'message': 'Labels saved successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/labels/custom', methods=['POST'])
def add_custom_label():
    """Add a custom label"""
    try:
        data = request.get_json()
        label_text = data.get('label', '')
        label_color = data.get('color', '#3b82f6')
        
        if not label_text:
            return jsonify({'success': False, 'error': 'Label cannot be empty'}), 400
        
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r', encoding='utf-8') as f:
                labels = json.load(f)
        else:
            labels = {'overall_attribute': {}, 'local_change': {}, 'custom_labels': []}
        
        new_label = {
            'id': f'custom_{int(time.time() * 1000)}',
            'text': label_text,
            'color': label_color
        }
        
        existing_texts = [l.get('text', l) if isinstance(l, dict) else l 
                         for l in labels.get('custom_labels', [])]
        
        if label_text not in existing_texts:
            labels.setdefault('custom_labels', []).append(new_label)
            
            with open(LABELS_FILE, 'w', encoding='utf-8') as f:
                json.dump(labels, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'custom_labels': labels['custom_labels']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Time Series Annotator v2 - Backend Starting...")
    print("API Server: http://localhost:5000")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Annotations Directory: {ANNOTATIONS_DIR}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
