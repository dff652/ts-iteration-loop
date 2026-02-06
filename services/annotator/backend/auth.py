"""
用户认证模块
提供JWT token生成、验证和密码管理功能
"""

import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import json
import os

# JWT配置
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-this-in-production')
TOKEN_EXPIRATION_HOURS = 24

# 用户数据文件
USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')


def load_users():
    """加载用户配置"""
    if not os.path.exists(USERS_FILE):
        # 创建默认用户
        default_users = {
            'admin': {
                'password_hash': generate_password_hash('admin123'),
                'name': '管理员'
            }
        }
        save_users(default_users)
        return default_users
    
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_users(users):
    """保存用户配置"""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def generate_token(username):
    """生成JWT token"""
    payload = {
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')


def verify_token(token):
    """验证JWT token"""
    try:
        if token.startswith('Bearer '):
            token = token[7:]
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload['username']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def verify_password(username, password):
    """验证用户密码"""
    import hashlib
    users = load_users()
    if username not in users:
        return False
    
    password_hash = users[username]['password_hash']
    
    # Support both sha256 (from manage_users.py) and werkzeug format
    if password_hash.startswith('sha256:'):
        # Simple sha256 hash
        expected_hash = password_hash[7:]  # Remove 'sha256:' prefix
        actual_hash = hashlib.sha256(password.encode()).hexdigest()
        return expected_hash == actual_hash
    else:
        # Werkzeug password hash (pbkdf2)
        return check_password_hash(password_hash, password)


def get_current_user_from_request():
    """从请求中获取当前用户"""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    return verify_token(auth_header)


def _is_truthy(value):
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _is_auth_bypass_enabled():
    # Disabled by default. Enable only in local development when needed.
    return _is_truthy(os.environ.get("ANNOTATOR_AUTH_BYPASS", "false"))


def login_required(f):
    """登录验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if _is_auth_bypass_enabled():
            username = os.environ.get("ANNOTATOR_AUTH_BYPASS_USER", "douff")
            return f(*args, current_user=username, **kwargs)

        username = get_current_user_from_request()
        if not username:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401

        # Pass username to route
        return f(*args, current_user=username, **kwargs)
    return decorated_function
