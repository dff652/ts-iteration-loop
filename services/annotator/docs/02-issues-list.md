# 时序数据标注工具 - 问题清单

## 当前状态截图

### V1 界面（timeseries-annotator-v1）
![V1 界面](./images/v1-interface.png)
*图1：V1 项目界面 - 功能完整，暗色主题，三栏布局合理*

### TRAINSET 界面（trainset）
![TRAINSET 界面](./images/trainset-interface.png)
*图2：TRAINSET 项目界面 - 浅色主题，拖拽标注交互流畅*

### V2 界面（timeseries-annotator-v2）
![V2 空状态](./images/v2-empty-state.png)
*图3：V2 项目空状态 - 布局混乱，存在多个问题*

![V2 上传后](./images/v2-after-upload.png)
*图4：V2 上传CSV后 - 界面完全异常，只显示Cancel/Ok按钮*

---

## 一、🔴 严重问题 (Critical)

### 1.1 Labeler页面完全无法工作
**问题描述**：上传CSV后跳转到Labeler页面，D3.js图表未渲染，界面只显示Cancel/Ok两个按钮

**截图位置**：图4

**影响**：完全无法进行标注操作

**根本原因分析**：
```javascript
// Index.vue 中的数据格式转换
const plotDict = data.data.map((d, idx) => ({
  id: idx.toString(),
  val: d.val,
  time: DateTime.fromISO(d.time, {setZone: true}),  // 需要 DateTime 对象
  series: d.series || 'value',
  label: d.label || ''
}));
```

```javascript
// LabelerD3.js 期望的数据格式 (line 648)
function type(d) {
  d.actual_time = DateTime.fromISO(d.time, { setZone: true });
  d.time = DateTime.fromISO(d.time.toISO({ includeOffset: false }));
  // 期望 d.time 已经是 DateTime 对象，但传入的可能是已转换的对象
}
```

**问题链**：
1. 后端返回 `{ time: "2025-01-01T00:00:00", val: 12.5 }` (字符串时间)
2. Index.vue 将 time 转换为 DateTime 对象
3. Labeler.vue 接收数据并传给 LabelerD3
4. LabelerD3.js 的 `type()` 函数再次调用 `DateTime.fromISO(d.time)`
5. 由于 d.time 已是 DateTime 对象而非字符串，导致解析失败
6. D3 图表渲染崩溃，模态框异常触发

### 1.2 标签管理模块显示为空
**问题描述**：左侧"标签管理"区域未加载 `labels.json` 中的内置标签

**截图位置**：图3 - 标签管理部分

**影响**：无法使用预设的整体属性和局部变化标签

**根本原因分析**：
```javascript
// Index.vue 的 loadLabels 方法
async loadLabels() {
  const res = await fetch(`${API_BASE}/labels`);
  const data = await res.json();
  if (data.success) {
    this.labels = data.labels;
    // 问题：labels 结构没有正确映射
    // API返回: { overall_attribute: { name: "整体属性", categories: {...} } }
    // 但模板期望: labels.overall_attribute 直接是 categories
  }
}
```

```html
<!-- 模板中的引用 -->
<div v-for="(category, catId) in labels.overall_attribute" :key="catId">
  <!-- 错误：直接遍历 overall_attribute，但实际需要遍历 .categories -->
```

**修复方案**：
```javascript
// 正确的遍历方式
labels.overall_attribute?.categories
labels.local_change?.categories
```

### 1.3 无法浏览服务器目录
**问题描述**：只能手动输入路径，缺少目录浏览模态框

**截图位置**：图3 - 路径输入区域

**影响**：用户体验差，需要手动输入完整路径

**根本原因分析**：
- V2 的 Index.vue 只有简单的 `setPath()` 方法
- 缺少 V1 中的 `openDirBrowser()` 模态框组件
- 缺少 `/api/browse-dir` 的前端调用逻辑

---

## 二、🟡 其他问题 (Major)

### 2.1 顶部保留 TRAINSET 品牌和无用链接
**问题描述**：页面顶部显示 "TRAINSET" Logo 和 "Help/License" 链接

**截图位置**：图3 红框

**文件位置**：
- `frontend/src/components/BaseNavbar.vue` (第4-7行)
- `frontend/src/views/Labeler.vue` (第4-12行)

```vue
<!-- BaseNavbar.vue -->
<h1 class="navbar-brand">
  <router-link class="homeLink" v-bind:to="'/'">TRAINSET
    <img id="logo" src="/static/trainset_logo.png">
  </router-link>
</h1>
```

**建议**：替换为项目名称 "时序数据标注工具" 或简称

### 2.2 页面布局不合理
**问题描述**：首页三栏布局，但右侧面板在未选择文件时无意义

**对比分析**：

| 项目 | 左侧 | 中间 | 右侧 |
|------|------|------|------|
| V1 | 280px 文件+标签 | flex 图表 | 320px 标注列表 |
| V2 | 280px 文件+标签 | flex 欢迎页 | 300px 空标注面板 |

**问题**：
- 首页显示"暂无标注"的右侧面板没有意义
- 标注信息表单在未选择文件时不应显示
- 欢迎页居中显示但两侧留白过多

**建议**：
- 首页采用两栏布局（左侧文件选择 + 中间欢迎/预览）
- 进入标注模式后切换为三栏布局

### 2.3 输入文件夹路径框被遮挡
**问题描述**：在某些分辨率下路径输入框被其他元素遮挡

**截图位置**：图3

**原因分析**：
- CSS `overflow` 设置问题
- 左侧面板缺少 `min-width` 保护
- 输入框宽度使用固定值而非响应式

### 2.4 浅色/暗色主题不统一
**问题描述**：混合了TRAINSET浅色主题和部分V1暗色元素

**当前状态**：

| 组件 | 当前主题 |
|------|----------|
| 顶部导航 | 深紫色 (#7E4C64) |
| 侧边栏 | 浅灰色 (#f8f9fa) |
| 按钮 | 深紫色 (#7E4C64) |
| 背景 | 白色 |

**用户选择**：使用 **浅色主题**（当前TRAINSET风格）

---

## 三、🟢 新增发现问题

### 3.1 路由配置为多页应用
**问题描述**：当前使用Vue Router配置了4个独立路由

```javascript
// router/index.js
routes: [
  { path: '/', name: 'home', component: Index },
  { path: '/labeler', name: 'labeler', component: Labeler },
  { path: '/help', name: 'help', component: Help },
  { path: '/license', name: 'license', component: License }
]
```

**影响**：
- 从Index跳转到Labeler时需要通过路由参数传递大量数据
- 状态在页面间不共享，需要重新加载
- 浏览器刷新会丢失标注进度

### 3.2 D3模块依赖jQuery
**问题描述**：LabelerD3.js 使用 jQuery 选择器和操作

```javascript
// LabelerD3.js 中的jQuery使用
$("#maindiv").append("<div class=\"loader\"></div>");
$(".loader").css("display", "none");
$("#instrSelect").css("margin-top", viewBox_height + 50);
```

**影响**：
- Vue组件需要确保jQuery全局可用
- jQuery与Vue虚拟DOM可能产生冲突

### 3.3 数据格式不一致
**问题描述**：后端API返回格式与TRAINSET期望格式不完全匹配

| 字段 | 后端返回 | TRAINSET期望 |
|------|----------|--------------|
| time | ISO字符串 | DateTime对象 |
| series | 可选 | 必需 |
| label | 可选 | 空字符串 |

### 3.4 硬编码的CSS尺寸
**问题描述**：LabelerD3.js中的图表尺寸是硬编码的

```javascript
plottingApp.main_height = 500 - plottingApp.main_margin.top - plottingApp.main_margin.bottom;
plottingApp.context_height = 500 - plottingApp.context_margin.top - plottingApp.context_margin.bottom;
```

**影响**：无法适应不同屏幕尺寸

---

## 四、功能缺失清单

| 功能 | V1有 | V2缺失 | 优先级 |
|------|:----:|:------:|:------:|
| 目录浏览模态框 | ✓ | ✗ | P0 |
| D3.js图表渲染 | ✓ | 崩溃 | P0 |
| 预设标签加载 | ✓ | ✗ | P0 |
| Context Bar 导航 | ✓ | ✗ | P1 |
| 选区颜色同步 | ✓ | ✗ | P1 |
| 键盘快捷键 | ✓ | ✗ | P2 |
| 批量删除标注 | ✓ | ✗ | P2 |
| 错误日志系统 | ✓ | ✗ | P3 |

---

## 五、问题优先级排序

| 优先级 | 问题 | 预估工时 | 依赖 |
|:------:|------|:--------:|:----:|
| P0 | Labeler页面D3图表崩溃 | 4h | - |
| P0 | 标签管理结构错误 | 2h | - |
| P0 | 数据格式转换错误 | 2h | P0-1 |
| P1 | 目录浏览功能缺失 | 3h | - |
| P1 | 页面布局优化 | 3h | - |
| P2 | 顶部导航栏替换 | 0.5h | - |
| P2 | 输入框遮挡修复 | 1h | P1-2 |
| P3 | 统一浅色主题 | 2h | - |
