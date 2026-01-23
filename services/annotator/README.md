# 时序数据标注工具 v2

> 基于 D3.js 的时序数据可视化标注工具，支持多标签、多数据段标注

## 项目状态

🟢 **当前版本**: v0.3.7 (2025-12-24)

## 功能特性

### 📊 数据可视化
- **双视图**：主图 + 缩略图（全局导航）
- **框选统计**：实时显示索引、点数、范围、均值、标准差
- **悬停信息**：显示时间、数值、标签信息
- **多数据集管理**：支持CSV和JSON文件，标签页切换

### 🏷️ 标签管理
- **分层标签**：整体属性 + 局部变化两类标签
- **自定义颜色**：每个标签独立颜色，整体属性横向排列
- **标签列表**：紧凑布局，支持新增、编辑、删除分类和标签

### ✏️ 标注工作区
- **多标签同显**：图上所有标签一目了然
- **点击切换**：点击标签查看对应数据段索引
- **自动切换**：主图框选时自动显示当前标签的段
- **颜色一致**：数据段索引与标签颜色统一
- **固定布局**：标签、数据段、问题、分析结论、按钮始终可见

### 💾 数据持久化
- **自动保存**：添加、更新、删除标注后自动保存到服务器
- **文件关联**：每个CSV文件独立保存标注，切换文件自动加载
- **JSON导出**：包含标签、数据段、问题、分析结论
- **一标签多段**：一个标注可包含多个数据段

### 🔐 多用户协作
- **用户认证**：JWT token登录认证
- **独立空间**：每个用户标注保存在独立目录
- **避免冲突**：多用户可同时标注同名文件
- **用户管理**：命令行工具管理用户账号

## 快速开始

### 环境要求

- Python 3.8+
- Node.js 14+

### 安装依赖

**后端**：
```bash
cd backend
pip install -r requirements.txt
pip install PyJWT  # JWT认证
```

**前端**：
```bash
cd frontend
npm install
```

### 初始化用户

```bash
cd backend
python manage_users.py init
```

默认账号：`admin / admin123`

### 启动服务

**后端**（5000端口）：
```bash
cd backend
python app.py
```

**前端**（3003端口）：
```bash
cd frontend
npm run dev
```

访问：http://localhost:3003

## 用户管理

### 添加新用户

**命令格式**：
```bash
python manage_users.py add <用户名> <密码> [显示名称]
```

**示例**：
```bash
cd backend
# 添加用户alice，密码123456，显示名称Alice
python manage_users.py add alice 123456 Alice

# 添加用户bob，密码secretpwd，显示名称Bob
python manage_users.py add bob secretpwd Bob
```

### 查看用户列表

```bash
python manage_users.py list
```

### 用户数据存储

每个用户的标注保存在独立目录：

```
backend/annotations/
├── admin/
│   ├── dataset1.csv.json
│   └── dataset2.csv.json
├── alice/
│   └── experiment.csv.json
└── bob/
    └── test.csv.json
```

**优点**：
- ✅ 完全避免标注冲突
- ✅ 支持多用户并发工作
- ✅ 数据隔离更安全

## 使用流程

```
1. 登录系统 → 使用账号密码登录
2. 左侧选择文件 → 加载时序数据（自动加载已有标注）
3. 左侧选择标签 → 确定标注类型
4. 主图框选区域 → 自动着色并添加到工作区
5. 填写问题/分析结论 → 点击"添加标注"（自动保存到服务器）
6. 导出JSON → 下载最终标注结果到本地
```

## 目录结构

```
timeseries-annotator-v2/
├── backend/              # Flask后端
│   ├── app.py           # API入口
│   ├── auth.py          # JWT认证
│   ├── manage_users.py  # 用户管理工具
│   ├── users.json       # 用户配置
│   ├── config/          # 配置文件
│   └── annotations/     # 用户标注存储
│       ├── admin/
│       ├── alice/
│       └── bob/
├── frontend/            # Vue.js前端
│   ├── src/
│   │   ├── views/       # Index.vue主页 + Login.vue登录页
│   │   └── assets/js/   # LabelerD3.js图表逻辑
│   └── package.json
├── CHANGELOG.md         # 版本更新日志
└── docs/                # 文档目录
```

## 技术栈

- **前端**: Vue.js 2.x, D3.js
- **后端**: Flask, Pandas, PyJWT
- **存储**: JSON文件
- **认证**: JWT token

## 更新日志

见 [CHANGELOG.md](./CHANGELOG.md)

## License

MIT
