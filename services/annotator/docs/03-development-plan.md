# 时序数据标注工具 - 开发方案 (v2)

> 本文档基于架构决策文档更新，采用 **单页应用 + 浅色主题 + TRAINSET交互**

## 一、项目概述

### 1.1 目标
整合 **timeseries-annotator-v1** 的后端能力和 **TRAINSET** 的前端交互体验，开发一款专业的时序数据标注工具，用于生成 ChatTS 模型训练所需的标注数据。

### 1.2 关键决策

| 决策项 | 选择 |
|--------|------|
| 主题风格 | 浅色（TRAINSET风格） |
| 应用架构 | 单页应用（SPA） |
| 交互方式 | 点击/拖拽标注（TRAINSET） |
| 图表库 | D3.js（修复数据格式问题） |

---

## 二、系统架构

### 2.1 单页应用架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                         Index.vue (主入口)                        │
├─────────────────┬────────────────────────┬──────────────────────┤
│   LeftPanel     │      CenterPanel       │     RightPanel       │
│                 │                        │                      │
│ ┌─────────────┐ │  ┌──────────────────┐  │ ┌──────────────────┐ │
│ │PathSelector │ │  │   WelcomePage    │  │ │ AnnotationList   │ │
│ │  +浏览器    │ │  │  (无文件时显示)   │  │ │                  │ │
│ └─────────────┘ │  └──────────────────┘  │ └──────────────────┘ │
│ ┌─────────────┐ │  ┌──────────────────┐  │ ┌──────────────────┐ │
│ │  FileList   │ │  │TimeSeriesChart   │  │ │ AnnotationForm   │ │
│ │             │ │  │  (D3.js封装)     │  │ │  - 选区范围      │ │
│ └─────────────┘ │  │  + ContextBar    │  │ │  - 已选标签      │ │
│ ┌─────────────┐ │  └──────────────────┘  │ │  - 自定义标签    │ │
│ │LabelManager │ │                        │ │  - 问题描述      │ │
│ │ 整体属性    │ │                        │ └──────────────────┘ │
│ │ 局部变化    │ │                        │                      │
│ └─────────────┘ │                        │                      │
└─────────────────┴────────────────────────┴──────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Flask REST API                              │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| 前端框架 | Vue.js 2.x | 保持TRAINSET兼容性 |
| 图表库 | D3.js | TRAINSET原有实现 |
| 构建工具 | Webpack | 已配置 |
| 后端框架 | Flask | 已实现 |
| 存储 | JSON文件 | 简单持久化 |

---

## 三、界面设计

### 3.1 浅色主题配色

```css
:root {
  /* 背景 */
  --bg-primary: #ffffff;
  --bg-secondary: #f8f9fa;
  
  /* 品牌色 */
  --color-primary: #7E4C64;
  --color-primary-hover: #6a3f54;
  
  /* 功能色 */
  --color-success: #28a745;
  --color-danger: #dc3545;
  
  /* 文字 */
  --text-primary: #212529;
  --text-secondary: #6c757d;
  
  /* 边框 */
  --border-color: #dee2e6;
}
```

### 3.2 布局Grid

```css
.main-layout {
  display: grid;
  grid-template-columns: 280px 1fr 320px;
  height: calc(100vh - 60px);
  gap: 12px;
  padding: 12px;
}

/* 无文件时两栏布局 */
.main-layout.no-file {
  grid-template-columns: 280px 1fr;
}
```

### 3.3 交互操作

| 操作 | 效果 |
|------|------|
| **单击** 数据点 | 切换该点的标签状态 |
| **拖拽** 选区 | 批量标注选中区域 |
| **Shift + 拖拽** | 取消选中区域的标签 |
| 拖拽 Context Bar | 调整可视区域 |
| `↑` / `↓` | 缩放图表 |
| `←` / `→` | 平移图表 |
| `Ctrl+S` | 保存标注 |

---

## 四、需要修复的问题

### 4.1 P0 - Labeler页面D3图表崩溃

**根本原因**: Index.vue 将 time 转换为 DateTime 对象后传递给 Labeler，但 LabelerD3.js 的 type() 函数期望接收字符串再进行转换。

**修复方案**: 创建统一的数据转换层
```javascript
// utils/dataTransform.js
export function transformForD3(apiData) {
  return apiData.map((d, idx) => ({
    id: idx.toString(),
    val: parseFloat(d.val),
    time: d.time,  // 保持ISO字符串，让D3内部转换
    series: d.series || 'value',
    label: d.label || ''
  }));
}
```

### 4.2 P0 - 标签管理结构错误

**根本原因**: 模板直接遍历 `labels.overall_attribute`，但API返回的是 `{ name: "整体属性", categories: {...} }` 结构。

**修复方案**:
```vue
<!-- 修改前 -->
<div v-for="(category, catId) in labels.overall_attribute">

<!-- 修改后 -->
<div v-for="(category, catId) in labels.overall_attribute?.categories">
```

### 4.3 P1 - 目录浏览功能缺失

**修复方案**: 从V1移植目录浏览器组件
- 添加 `DirBrowserModal.vue` 组件
- 调用 `/api/browse-dir` API
- 集成到 PathSelector 组件

---

## 五、开发计划

### 5.1 阶段划分

| 阶段 | 内容 | 时间 |
|------|------|------|
| Phase 1 | 修复严重问题 | 1天 |
| Phase 2 | 单页应用重构 | 2天 |
| Phase 3 | 完善交互功能 | 2天 |
| Phase 4 | 测试与优化 | 1天 |

### 5.2 Phase 1: 修复严重问题

- [ ] 修复数据格式转换逻辑
- [ ] 修复标签管理结构映射
- [ ] 移除TRAINSET品牌和无用链接
- [ ] 确保D3图表正常渲染

### 5.3 Phase 2: 单页应用重构

- [ ] 删除Labeler.vue路由
- [ ] 将图表组件嵌入Index.vue
- [ ] 实现条件渲染（有/无文件状态）
- [ ] 添加目录浏览器组件

### 5.4 Phase 3: 完善交互功能

- [ ] 完善标签选择交互
- [ ] 实现选区→标签→保存流程
- [ ] 添加键盘快捷键
- [ ] 实现导出功能

### 5.5 Phase 4: 测试与优化

- [ ] 功能测试
- [ ] 界面优化
- [ ] 性能优化

---

## 六、验证计划

### 6.1 启动测试

```bash
# 1. 启动后端
cd backend && python app.py

# 2. 启动前端
cd frontend && npm run dev

# 3. 访问 http://localhost:8080
```

### 6.2 功能验证清单

| 测试项 | 预期结果 | 实际结果 |
|--------|----------|----------|
| 首页加载 | 显示欢迎页 + 左侧文件列表 | |
| 路径设置 | 输入路径后显示文件列表 | |
| 目录浏览 | 打开模态框，可浏览目录 | |
| 选择文件 | 图表区显示时序曲线 | |
| 标签显示 | 左侧显示预设标签 | |
| 拖拽选区 | 高亮选中区域 | |
| 保存标注 | 标注出现在右侧列表 | |
| 导出JSON | 下载格式正确的JSON | |

### 6.3 手动测试步骤

1. **路径设置测试**
   - 输入有效路径 `/home/douff/ts/timeseries-annotator-v1/timeseries-annotator-v1/data`
   - 验证文件列表显示

2. **标注流程测试**
   - 选择一个CSV文件
   - 在图表上拖拽创建选区
   - 在左侧选择标签
   - 点击"保存标注"
   - 验证标注出现在列表中

3. **导出测试**
   - 点击"下载"按钮
   - 验证JSON格式符合规范

---

## 七、文件结构

```
timeseries-annotator-v2/
├── backend/
│   ├── app.py              # Flask API
│   ├── config/
│   │   └── labels.json     # 标签配置
│   ├── data/               # 数据文件
│   └── annotations/        # 标注存储
│
├── frontend/
│   ├── src/
│   │   ├── views/
│   │   │   └── Index.vue   # 单页应用主入口（重构）
│   │   ├── components/
│   │   │   ├── file/
│   │   │   │   ├── PathSelector.vue
│   │   │   │   ├── FileList.vue
│   │   │   │   └── DirBrowserModal.vue  # 新增
│   │   │   ├── label/
│   │   │   │   └── LabelManager.vue
│   │   │   ├── chart/
│   │   │   │   └── TimeSeriesChart.vue  # D3封装
│   │   │   └── annotation/
│   │   │       ├── AnnotationList.vue
│   │   │       └── AnnotationForm.vue
│   │   ├── assets/
│   │   │   └── js/
│   │   │       └── LabelerD3.js  # 修复
│   │   └── utils/
│   │       ├── api.js
│   │       └── dataTransform.js  # 新增
│   └── package.json
│
└── docs/
    ├── 01-feature-list.md
    ├── 02-issues-list.md
    ├── 03-development-plan.md
    ├── 04-api-reference.md
    ├── 05-label-config-guide.md
    ├── 06-architecture-decision.md
    └── images/
```
