# 时序数据标注工具 - 架构决策文档

## 一、决策背景

在整合 timeseries-annotator-v1 后端和 TRAINSET 前端的过程中，需要对以下关键架构问题做出决策：

1. **主题风格**：浅色 vs 暗色
2. **应用架构**：单页应用(SPA) vs 多页应用(MPA)
3. **交互方式**：框选标注 vs 点击/拖拽标注

---

## 二、决策结果

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 主题风格 | **浅色（TRAINSET风格）** | 用户明确选择 |
| 应用架构 | **单页应用** | 见架构分析 |
| 交互方式 | **点击/拖拽标注（TRAINSET）** | 用户明确选择 |

---

## 三、架构选择分析

### 3.1 多页应用架构（当前状态）

```
┌─────────────┐    路由跳转    ┌─────────────┐
│  Index.vue  │ ────────────> │ Labeler.vue │
│  (首页)      │   params传递   │  (标注页)    │
└─────────────┘               └─────────────┘
```

**当前问题**：
1. ❌ 数据通过路由参数传递，刷新页面数据丢失
2. ❌ 需要在两个页面维护标签状态
3. ❌ Labeler.vue 依赖 LabelerD3.js，与后端数据格式不兼容
4. ❌ 首页右侧标注面板在无文件时无意义

**代码证据**：
```javascript
// Index.vue - 跳转时传递数据
this.$router.push({
  name: 'labeler',
  params: {
    csvData: plotDict,        // 大量数据通过params传递
    filename: file.name,
    headerStr: 'series,time,val,label',
    seriesList: seriesList,
    labelList: [],
    isValid: true
  }
});
```

### 3.2 单页应用架构（推荐）

```
┌──────────────────────────────────────────────────────────┐
│                       Index.vue                          │
├────────────────┬─────────────────────┬──────────────────┤
│   左侧面板      │      中间区域        │    右侧面板      │
│                │                     │                  │
│  文件列表       │  无文件: 欢迎页      │  标注列表        │
│  标签管理       │  有文件: 图表组件    │  标注信息输入    │
│                │                     │                  │
└────────────────┴─────────────────────┴──────────────────┘
```

**优势**：
1. ✅ 状态集中管理，刷新不丢失（可结合localStorage）
2. ✅ 标签选择与图表在同一组件树，响应更快
3. ✅ 无需通过路由传递大量数据
4. ✅ 渐进式展示：选文件后右侧面板才显示
5. ✅ 更符合V1的交互模式

**实现方案**：
```vue
<template>
  <div class="main-layout">
    <!-- 左侧始终显示 -->
    <aside class="left-panel">
      <FilePanel />
      <LabelManager />
    </aside>
    
    <!-- 中间区域条件渲染 -->
    <main class="center-panel">
      <WelcomePage v-if="!currentFile" />
      <ChartArea v-else :data="chartData" @select="handleSelect" />
    </main>
    
    <!-- 右侧仅在有文件时显示 -->
    <aside v-if="currentFile" class="right-panel">
      <AnnotationList />
      <AnnotationForm />
    </aside>
  </div>
</template>
```

### 3.3 决策理由

选择 **单页应用** 架构，理由如下：

| 评估维度 | 多页应用 | 单页应用 |
|----------|:--------:|:--------:|
| 状态管理复杂度 | 高 | 低 |
| 数据传递 | 路由参数 | 组件props |
| 刷新数据保持 | ❌ | ✅ |
| 代码复用 | 低 | 高 |
| 开发工作量 | 高 | 中 |
| 与V1体验一致性 | 低 | 高 |

---

## 四、图表库选择

### 4.1 当前状态

| 项目 | 图表库 | 代码量 |
|------|--------|--------|
| V1 | Chart.js | 2384行 (app.js) |
| TRAINSET/V2 | D3.js | 898行 (LabelerD3.js) |

### 4.2 对比分析

| 特性 | Chart.js (V1) | D3.js (TRAINSET) |
|------|---------------|------------------|
| 学习曲线 | 简单 | 陡峭 |
| 定制灵活性 | 中等 | 极高 |
| 拖拽选区 | 插件支持 | 原生brush |
| 点击标注 | 事件处理 | 原生支持 |
| 与Vue集成 | 良好 | 需要适配 |
| 现有代码完成度 | 高 | 高 |

### 4.3 建议

保留 **D3.js** 方案，理由：
1. TRAINSET的拖拽标注交互已经实现完整
2. 用户明确选择了TRAINSET的交互方式
3. D3的brush功能非常适合区间选择

**但需要修复**：
1. 数据格式转换逻辑
2. 移除jQuery依赖或确保正确加载
3. 响应式尺寸适配

---

## 五、重构方案

### 5.1 文件结构调整

```
frontend/src/
├── views/
│   └── Index.vue          # 单页应用主入口（重构）
├── components/
│   ├── layout/
│   │   ├── LeftPanel.vue      # 左侧面板容器
│   │   ├── CenterPanel.vue    # 中间区域容器
│   │   └── RightPanel.vue     # 右侧面板容器
│   ├── file/
│   │   ├── PathSelector.vue   # 路径选择（含浏览器）
│   │   └── FileList.vue       # 文件列表
│   ├── label/
│   │   ├── LabelManager.vue   # 标签管理
│   │   └── LabelCategory.vue  # 标签分类
│   ├── chart/
│   │   ├── TimeSeriesChart.vue # 主图表（封装D3）
│   │   └── ContextBar.vue     # 缩略导航
│   ├── annotation/
│   │   ├── AnnotationList.vue  # 标注列表
│   │   └── AnnotationForm.vue  # 标注表单
│   └── common/
│       ├── AppNavbar.vue      # 顶部导航（替换TRAINSET）
│       ├── Toast.vue          # 通知组件
│       └── Modal.vue          # 模态框
├── assets/
│   ├── js/
│   │   └── LabelerD3.js       # D3图表逻辑（修复）
│   └── css/
│       └── theme.css          # 浅色主题
└── utils/
    ├── api.js                 # API封装
    └── dataTransform.js       # 数据格式转换
```

### 5.2 删除的文件/路由

```javascript
// 移除不再需要的路由
routes: [
  { path: '/', name: 'home', component: Index },
  // 删除: { path: '/labeler', ... }
  // 删除: { path: '/help', ... }
  // 删除: { path: '/license', ... }
]

// 删除的文件
- views/Labeler.vue   (功能合并到Index)
- views/Help.vue      (不需要)
- views/License.vue   (不需要)
```

### 5.3 D3模块修复

```javascript
// 新增 dataTransform.js
export function transformBackendData(apiData) {
  return apiData.map((d, idx) => ({
    id: idx.toString(),
    val: parseFloat(d.val),
    time: d.time,  // 保持字符串，让D3内部转换
    actual_time: d.time,
    series: d.series || 'value',
    label: d.label || '',
    x: idx,
    y: parseFloat(d.val)
  }));
}
```

---

## 六、交互方式确认

采用 **TRAINSET 点击/拖拽标注** 方式：

### 6.1 操作说明

| 操作 | 效果 |
|------|------|
| **单击** 数据点 | 切换该点的标签状态 |
| **拖拽** 选区 | 批量标注选中区域 |
| **Shift + 拖拽** | 取消选中区域的标签 |
| 拖拽 Context Bar | 调整可视区域 |

### 6.2 与V1的差异

| 功能 | V1 | TRAINSET |
|------|:--:|:--------:|
| 框选标注 | ✓ | ✓ |
| 点击标注 | ✗ | ✓ |
| 拖拽取消标注 | ✗ | ✓ |
| Context Bar | ✓ | ✓ |
| 预览图表 | ✓ | ✗ |

---

## 七、主题风格确认

采用 **浅色主题**：

### 7.1 配色方案

```css
:root {
  /* 背景 */
  --bg-primary: #ffffff;
  --bg-secondary: #f8f9fa;
  --bg-panel: #ffffff;
  
  /* 品牌色（保留TRAINSET紫色） */
  --color-primary: #7E4C64;
  --color-primary-hover: #6a3f54;
  
  /* 功能色 */
  --color-success: #28a745;
  --color-warning: #ffc107;
  --color-danger: #dc3545;
  
  /* 文字 */
  --text-primary: #212529;
  --text-secondary: #6c757d;
  
  /* 边框 */
  --border-color: #dee2e6;
}
```

### 7.2 需要修改的组件

- `BaseNavbar.vue` - 替换 TRAINSET 品牌
- `Index.vue` - 统一配色
- `LabelerModal.vue` - 按钮样式统一
