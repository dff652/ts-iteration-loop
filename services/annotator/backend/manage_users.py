#!/usr/bin/env python3
"""
用户管理工具
用于添加、删除、查看用户
"""
import json
import os
import sys

# 确保在backend目录运行
USERS_FILE = 'users.json'

def load_users():
    """加载用户"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_users(users):
    """保存用户"""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def hash_password(password):
    """生成密码哈希"""
    try:
        from werkzeug.security import generate_password_hash
        return generate_password_hash(password)
    except ImportError:
        # 如果werkzeug不可用，使用简单的方法（不推荐生产环境）
        import hashlib
        return 'sha256:' + hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password, name=None):
    """添加用户"""
    users = load_users()
    if username in users:
        print(f"❌ 用户 {username} 已存在")
        return False
    
    users[username] = {
        'password_hash': hash_password(password),
        'name': name or username
    }
    save_users(users)
    print(f"✅ 已添加用户: {username}")
    return True

def list_users():
    """列出所有用户"""
    users = load_users()
    if not users:
        print("📋 暂无用户")
        return
    
    print("📋 用户列表:")
    for username, info in users.items():
        name = info.get('name', username)
        print(f"  - {username} ({name})")

def init_default_user():
    """初始化默认管理员账号"""
    users = load_users()
    if 'admin' not in users:
        users['admin'] = {
            'password_hash': hash_password('admin123'),
            'name': '管理员'
        }
        save_users(users)
        print("✅ 已创建默认账号: admin / admin123")
    else:
        print("ℹ️  默认账号已存在")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法:")
        print("  python manage_users.py init           # 初始化默认账号")
        print("  python manage_users.py list           # 查看所有用户")
        print("  python manage_users.py add <user> <pwd> [name]  # 添加用户")
        print("\n示例:")
        print("  python manage_users.py init")
        print("  python manage_users.py add alice 123456 Alice")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == 'init':
        init_default_user()
    elif cmd == 'list':
        list_users()
    elif cmd == 'add':
        if len(sys.argv) < 4:
            print("❌ 用法: python manage_users.py add <username> <password> [name]")
            sys.exit(1)
        username = sys.argv[2]
        password = sys.argv[3]
        name = sys.argv[4] if len(sys.argv) > 4 else None
        add_user(username, password, name)
    else:
        print(f"❌ 未知命令: {cmd}")
        sys.exit(1)
