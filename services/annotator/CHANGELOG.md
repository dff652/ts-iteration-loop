# Changelog - 时序标注工具 V2

## [0.3.8] - 2026-01-28

### 🐛 Bug修复
- **新建标注不可见** - 修复在"Create New"流程下，框选后工作区未更新且无法保存的问题
    - [views/Index.vue] `updateSelectionRange` 同步 `workspaceData` 并切换状态
    - [views/Index.vue] `finishWorkspaceEdit` 支持新建保存逻辑
- **Qwen 标签支持** - 后端支持自动识别 Qwen 推理结果并应用 Qwen 标签

### 🔧 技术改进
- **前端重构** - 优化了标注保存与状态重置的逻辑闭环

## [0.3.7] - 2025-12-24

### 🐛 关键Bug修复

#### 路径输入框Enter键无效
- **问题** - 用户在路径输入框输入路径后按Enter键无法设置路径
- **根因** - 前端模板绑定了不存在的方法`setPath`，实际方法名为`setDataPath`
- **修复** - [Index.vue:30](frontend/src/views/Index.vue#L30) 修改 `@keyup.enter="setPath"` → `@keyup.enter="setDataPath"`
- **影响** - 用户现在可以通过输入路径+Enter键快速设置数据目录

#### 标注状态显示问题
- **问题** - 完成100个文件标注后，界面只显示1个文件有标注状态（✓），其他99个未显示
- **根因1** - 异常文件名 `数据集zhlh_100_TI_40301.PV.csv.json` 多了`.csv`后缀
- **根因2** - 后端Pattern 1-4无法匹配CSV文件名已有"数据集"前缀的情况
- **修复** - 重命名异常文件 + 添加Pattern 5直接替换`.csv` → `.json`
- **优化** - 调整pattern匹配顺序，将最常用的Pattern 5放首位，提升查找效率

### 🔧 技术改进

#### 后端API优化 ([app.py:172-194](backend/app.py#L172-194))
- **新增Pattern 5** - `f.replace('.csv', '.json')` 支持直接扩展名替换
- **优化匹配顺序** - `[pattern5, pattern1, pattern4, pattern3, pattern2]` 按使用频率排序
- **向后兼容** - 保留所有5种pattern，确保历史文件命名格式兼容
- **性能提升** - 对于当前100个文件，从平均4次失败检查减少到1次成功匹配

### 📁 文件改动

#### Frontend
- **[MODIFY] views/Index.vue** - 修复路径输入Enter键绑定（1行）

#### Backend
- **[MODIFY] app.py** - 添加Pattern 5并优化匹配顺序（5行）
- **[RENAME] annotations/douff/** - 重命名1个异常JSON文件

---

## [0.3.6] - 2025-12-23

### ✨ 核心功能改进

#### 批量标注保存
- **一键保存所有标签** - "添加标注"按钮现在一次性保存所有工作区标签及其数据段
- **无标签标注支持** - 允许仅保存问题和评价，无需选择标签或框选数据
- **智能合并** - 相同标签的标注自动合并数据段

#### 数据持久化修复
- **统一JSON格式** - 自动保存和手动导出使用相同的数据结构
- **多格式文件名识别** - 后端自动识别三种标注文件名格式
- **字段名映射** - 自动处理`expert_output`和`expertOutput`字段转换

### 🐛 Bug修复

- **[严重] 标注保存后不显示** - 修复`saveActiveLabel`缺少`saveAnnotationsToServer`调用
- **[严重] 刷新后标注丢失** - 修复字段名不匹配导致数据加载失败
- **[关键] 中文文件名URL编码** - 添加`encodeURIComponent`支持中文文件名
- **[关键] 函数签名错误** - 修复`save_annotations`缺少`filename`参数
- **标注计数逻辑** - 无标签标注显示✓徽章但计数为0

### 🎨 UI/UX改进

- **自动保存提示** - 保存成功时显示"已自动保存"toast
- **文件列表标注徽章** - 显示✓和标注数量，hover显示"X 个标注"
- **按标注数排序** - 文件列表支持按标注数量排序
- **错误提示增强** - 保存失败时显示详细错误信息

### 🔧 技术改进

- **后端API增强** - 支持通过JSON内`filename`字段匹配标注文件
- **自动刷新** - 保存后自动刷新文件列表更新徽章
- **向后兼容** - 兼容旧版JSON数据格式

## [0.3.5] - 2025-12-23

### ✨ UI/UX 优化

#### 文件管理改进
- **文件列表扩大** - 高度从180px增至300px，显示更多文件
- **文件边框** - 每个文件添加边框，hover和选中有视觉反馈
- **文件排序** - 添加排序下拉菜单，支持按名称或标注数排序
- **自然排序** - 文件名中的数字作为整体比较，与Windows排序一致
  - 修复：`P6203` 现在正确排在 `P64058` 前面
  - 修复：`LI_12003` 正确排在 `LI_50701` 前面

#### 标签列表紧凑化
- **整体属性布局** - 缩小间距和字体，节省空间
- **标签间距优化** - gap从6px减到4px，字体从0.8125rem减到0.75rem

### 🛠️ 技术改进

- **自然排序算法** - 实现与系统文件管理器一致的自然排序
- **响应式排序** - computed属性自动根据排序选项更新列表
- **调试信息清理** - 移除loadFiles中的console.log

### 📁 文件改动

#### Frontend
- **[MODIFY] views/Index.vue** - 添加自然排序、排序UI、优化文件列表样式（+60行）

---

## [0.3.4] - 2025-12-23

### 🔐 多用户功能改进

#### 用户路径记忆
- **独立路径** - 每个用户的数据路径独立保存在`users.json`
- **自动加载** - 登录后自动加载用户上次设置的路径
- **浏览起点** - 目录浏览器从用户当前路径开始，而非固定的`/home`
- **路径持久化** - 用户设置的路径在重新登录后保持不变

#### 前端功能完善
- **用户信息显示** - 导航栏右上角显示当前登录用户名
- **登出按钮** - 点击登出清除token并跳转登录页
- **路径选择** - 目录浏览器"选择"按钮正常工作
- **Authorization头** - 所有API请求添加JWT token认证

### 🐛 Bug修复

- **[严重] 数据段索引NaN** - 修复框选后显示"NaN-NaN"的问题
  - 原因：`activeSegments`使用了错误的字段`d.id`而非`d.time`
  - 修复：改用`parseInt(d.time)`计算segment范围
  - 影响：数据段索引、标注列表、工作区显示
  
- **数据格式兼容** - `loadAnnotationsForFile`添加数据归一化
  - 支持旧格式数组`[start, end]`和新格式对象`{start, end}`
  - 自动过滤无效segment（start或end为NaN）
  
- **路径API认证** - 修复`get_files`、`get_data`、`get_current_path`等API的认证问题
  - 所有路径相关API添加`@login_required`装饰器
  - 使用用户专属路径而非全局共享路径

### 📁 文件改动

#### Backend
- **[MODIFY] app.py** - API使用用户路径，添加认证装饰器（+25行）
- **[MODIFY] auth.py** - 导出`save_users`函数供路径保存使用（+5行）

#### Frontend
- **[MODIFY] views/Index.vue** - 修复activeSegments、添加用户信息显示、完善auth头（+40行）
- **[MODIFY] router/index.js** - 路由守卫处理（修改）

---

## [0.3.3] - 2025-12-23

### 🔐 多用户协作功能

#### 用户认证系统
- **JWT登录** - 基于token的无状态认证
- **登录页面** - 简洁的登录界面，默认账号 admin/admin123
- **路由守卫** - 未登录自动跳转登录页
- **登出功能** - 导航栏显示用户名和登出按钮
- **密码加密** - 支持sha256和pbkdf2两种哈希格式

#### 用户独立空间
- **独立标注目录** - 每个用户标注保存在 `annotations/{username}/` 
- **避免冲突** - 多用户可同时标注同名文件而不冲突
- **数据隔离** - 用户只能查看和编辑自己的标注

#### 用户管理工具
- **manage_users.py** - 命令行用户管理工具
- **初始化账号** - `python manage_users.py init` 创建默认admin账号
- **添加用户** - `python manage_users.py add <user> <pwd> [name]`
- **查看用户** - `python manage_users.py list`

### 📁 文件改动

#### Backend
- **[NEW] auth.py** - JWT认证模块，token生成和验证（+105行）
- **[NEW] manage_users.py** - 用户管理CLI工具（+105行）
- **[NEW] users.json** - 用户配置文件
- **[MODIFY] app.py** - 添加登录API，标注API使用用户独立目录（+60行）

#### Frontend
- **[NEW] views/Login.vue** - 登录页面组件（+210行）
- **[MODIFY] router/index.js** - 添加登录路由和路由守卫（+25行）
- **[MODIFY] views/Index.vue** - 添加Authorization头、用户信息显示、登出功能（+30行）

### 🐛 Bug修复

- **修复密码验证** - auth.py支持sha256和werkzeug两种密码哈希格式
- **修复导航栏样式** - 调整为白色背景，添加用户信息和登出按钮

---

## [0.3.2] - 2025-12-23

### 🔧 核心功能

#### 数据集切换标注同步
- **问题修复** - 切换数据集时标注结果和工作区现在正确同步
- **自动加载** - 切换文件时自动从服务器加载对应文件的标注结果
- **自动保存** - 添加/更新/删除标注后自动保存到服务器
- **状态重置** - 切换文件时正确重置工作区和标注列表

#### 自动保存机制
- **添加标注自动保存** - 点击"添加标注"后自动调用服务器保存API  
- **更新标注自动保存** - 编辑模式下点击"更新标注"后自动保存
- **删除标注自动保存** - 删除标注后立即同步到服务器
- **移除手动保存按钮** - "💾 保存"按钮改为"💾 保存"（保留但主要依赖自动保存）

### 🎨 UI/UX 改进

#### 右侧工作区布局优化
- **固定布局** - 工作区始终显示完整布局（标签、数据段索引、问题、分析结论、按钮）
- **数据段索引固定高度** - 设置120px固定高度，避免切换标签时UI跳动
- **禁用状态** - 未选中标签时输入框和按钮禁用并显示灰色
- **文本优化** - "专家分析" → "分析结论"，"保存标注" → "添加标注"

#### 左侧标签列表优化
- **标题更新** - "标签管理" → "标签列表"
- **设置按钮** - 添加文字"⚙️ 设置"，更易识别
- **布局紧凑化** - 整体属性横向排列，统一减小间距
  - 分类间距：12px → 2px
  - 标签间距：6px → 2-3px
  - 整体属性改为 flex-wrap 横向排列
  - 局部变化 padding 从 4px 8px 减至 2px 6px

#### 交互改进
- **切换提示优化** - 移除阻断式弹窗，改为温和的toast提示
- **编辑功能修复** - 点击"✏️"按钮正确加载标注到工作区进行编辑

### 📁 文件改动

#### Frontend - `src/views/Index.vue`
- 新增 `loadAnnotationsForFile(filename)` - 从服务器加载指定文件的标注
- 新增 `saveAnnotationsToServer()` - 保存标注到服务器
- 修改 `selectFile()` - 添加状态重置和自动加载逻辑
- 修改 `saveCurrentAnnotation()` - 添加自动保存调用
- 修改 `saveActiveLabel()` - 添加自动保存调用
- 修改 `deleteAnnotation()` - 添加自动保存调用
- 修改 `editAnnotation()` - 设置 activeChartLabel 正确激活工作区
- 优化 CSS - 数据段索引固定高度，标签列表紧凑布局
- 添加 textarea:disabled 样式

### 🐛 Bug修复

- **修复CSS语法错误** - 移除 `.local-label-item.active` 中多余的 `#f3e8ed`
- **修复编辑功能** - 编辑标注时正确设置 activeChartLabel 使工作区可见
- **修复切换文件弹窗** - 改为非阻断式的toast警告

---

## [0.3.1] - 2025-12-22/23

### 🎨 UI/UX 改进

#### 框选统计优化
- **位置调整** - 将框选统计从图表右上角叠加层移至工具栏显示
- **Grid布局** - 使用 CSS Grid 实现整齐的标签-数值对齐
- **标准差单独一行** - 避免与其他指标对齐问题

#### 右侧面板重新设计
- **统一工作区** - 合并"快速标注"和"当前编辑"为"📝 标注工作区"
- **标签区** - 显示所有图上标签，点击切换查看数据段
- **数据段索引** - 更名为"数据段索引"，索引颜色与标签颜色一致
- **自动切换** - 主图框选时自动切换显示当前标签的数据段

#### 交互增强
- **标签点击切换** - 点击图上标签可切换查看其数据段
- **段导航** - 点击数据段可定位到图表对应区域
- **颜色一致性** - 数据段索引颜色、边框与标签颜色统一

### 🐛 Bug修复

#### 缩略图残留问题
- **图上标签取消后缩略图残留** - 修复 `clearLabelFromChart` 方法
- **已框选数据段取消后残留** - 修复 `removeSegmentByRange` 方法

#### 响应式修复
- **框选统计显示** - 修复 `selectionStats` 计算属性响应式更新
- **数据段列表** - 新增 `activeSegments` 计算属性动态获取标签的段

### 📁 文件改动

#### Frontend
- `src/views/Index.vue`
  - 新增 `activeChartLabel` 数据属性跟踪当前选中标签
  - 新增 `activeSegments` 计算属性获取当前标签的数据段
  - 新增 `activeLabelColor` 计算属性获取当前标签颜色
  - 新增 `selectChartLabel()` 方法切换选中标签
  - 新增 `saveActiveLabel()` 方法保存当前标签为标注
  - 新增 `removeSegmentByRange()` 方法删除指定范围段
  - `updateSelectionRange()` 添加自动切换逻辑
  - 重构右侧面板 HTML 结构
  - 新增 `.chart-label-tag` 和 `.workspace-section` CSS 样式

---


## [0.3.0] - 2025-12-22

### 🔧 标注流程重构

#### 核心功能修复
- **标签管理CRUD修复** - 修复编辑分类无法保存的问题，使用直接对象引用替代计算属性副本
- **清除标注逻辑优化** - 清除图上所有点颜色，联动清除已选标签，但不清除已保存的标注列表
- **取消标签联动** - 取消已选标签时，联动清除图上该标签对应的数据点颜色
- **子标签颜色唯一** - 新增 `generateUniqueColor()` 方法，使用20色调色板自动分配不重复颜色
- **编辑标注支持** - 点击已保存标注的✏️按钮可加载到编辑区进行修改

#### 数据结构
- **一标签多段** - 一个标注 = 1个标签 + N个数据段 + 1个问题 + 1个专家分析
- **导出格式适配** - JSON导出格式支持新的一对多数据结构

### 🎨 UI/UX 改进

#### 布局修复
- **序列选择器始终可见** - 不再隐藏单序列文件的「主序列/参考序列」下拉框
- **移除重复导出按钮** - 工具栏只保留「清除标注」，右侧边栏保留「下载」按钮
- **编辑状态高亮** - 正在编辑的标注项显示紫色边框和浅紫色背景
- **动态按钮文案** - 编辑模式时保存按钮显示「更新标注」

#### 交互改进
- **标签选中视觉反馈** - 点击局部变化标签时有明确的高亮效果
- **操作反馈完善** - 添加/删除分类和标签时显示Toast提示

### 📁 文件改动

#### Frontend
- `src/views/Index.vue`
  - 新增 `editingAnnotationIndex` 状态管理编辑模式
  - 新增 `editAnnotation(idx)` 方法加载标注进行编辑
  - 新增 `usedColors` 计算属性收集已使用颜色
  - 新增 `generateUniqueColor()` 方法生成唯一颜色
  - 修复 `clearCurrentLabel()` 联动清除图上点
  - 修复 `clearAllLabels()` 不清除已保存列表
  - 修复 `saveCurrentAnnotation()` 支持更新模式
  - 修复 `addLabelToCategory()` 使用正确的对象引用
  - 修复 `deleteLabelFromCategory()` 使用正确的对象引用
  - 新增编辑状态CSS样式 `.annotation-item.editing`

### 🐛 Bug修复

- **修复框选无法添加数据段问题** - 在 `updateSelectionRange` 中强制Vue响应式更新，使用对象展开运算符更新 `currentAnnotation`
- **添加调试日志** - 在 `updateSelectionRange` 中添加console.log帮助诊断问题

### 🐛 已知待修复问题

- [ ] 标签管理弹窗分类显示问题待进一步验证
- [ ] 缩略图光标易触发浏览器右键菜单
- [ ] 工具栏位置调整（减少图表与工具栏间空白）
- [ ] 主图和缩略图尺寸优化

---

## [0.2.0] - 2025-12-21

### 🎯 核心功能修复

#### 图表渲染
- **修复图表不显示问题** - 重写了数据初始化流程，确保 D3 在 DOM 就绪后渲染
- **修复 NaN 序列化问题** - 后端将所有 NaN 值转换为 `null` 或空字符串，确保 JSON 有效性
- **修复时间格式问题** - 后端统一将时间转换为 ISO 格式（`YYYY-MM-DDTHH:mm:ss.000Z`），支持多种输入格式
- **优化 LabelerD3 初始化** - 修复 selectedSeries/refSeries 从空 DOM 获取的问题，优先使用预设值
- **增强数据验证** - 在 `updateBrushData` 中添加空数据检查和 try-catch 错误处理

#### 后端改进
- **智能列检测** - 支持中文列名（时间、日期、值、序列、标签）
- **多序列支持** - 正确识别和处理多个数据序列
- **索引作为 X 轴** - 无时间列时自动使用行索引
- **时间格式转换** - 支持多种时间格式自动转换为 ISO 标准格式

### 🎨 UI/UX 优化

#### 布局调整
- **Navbar 简化** - 移除右上角操作按钮，添加当前文件名显示
- **数据管理重新设计** - 合并"数据路径"和"数据文件"为一个卡片，添加标签页：
  - 📄 **原始数据** (CSV 文件)
  - 📝 **标注结果** (JSON 文件，支持回看和修改)
- **工具栏重新布局** - 将"清除标注"和"导出"按钮移至图表下方工具栏，与操作提示、序列选择器对齐

#### 交互改进
- **局部标签单选** - 局部变化标签改为单选模式，避免标注冲突
- **颜色一致性** - 统一使用大类颜色：
  - 左侧标签、图表标注点、已选标签区域使用相同颜色
  - 每个大类（异常突变、渐变趋势等）分配一个固定颜色
- **紧凑布局** - 左侧边栏使用卡片样式，模块边框清晰，间距优化

### 🌐 网络配置

#### 远程访问支持
- **配置服务器 IP** - 前端 API_BASE 改为 `http://192.168.199.126:5000/api`
- **数据目录设置** - 支持设置自定义数据目录（`/home/douff/数据标注/data/标注数据`）
- **跨机器访问** - 支持从 PC (192.168.199.242) 访问服务器 (192.168.199.126)

### 📁 文件改动

#### Backend
- `app.py`
  - 重写 `get_data()` 函数，增强 NaN 处理和列检测
  - 添加 `to_iso_time()` 辅助函数，统一时间格式转换
  - 支持多种时间格式：`YYYY-MM-DD HH:MM:SS`、`YYYY/MM/DD HH:MM:SS`、ISO 格式等

#### Frontend
- `src/views/Index.vue`
  - 重构 Navbar 模板（移除操作按钮）
  - 重构数据管理区域（添加 CSV/JSON 标签页）
  - 重构工具栏布局（添加操作按钮）
  - 修改 `toggleLocalLabel()` 为单选逻辑
  - 添加 `csvFiles` 和 `jsonFiles` 计算属性
  - 添加 `loadResultFile()` 方法（TODO：实现 JSON 加载逻辑）
  - 添加工具栏、标签页相关 CSS 样式
  - 添加更多大类颜色配置

- `src/assets/js/LabelerD3.js`
  - 修复 `selectedSeries`/`refSeries` 初始化逻辑
  - 添加调试日志输出
  - 增强 `updateBrushData()` 数据验证

### 🐛 已知问题

- [ ] JSON 结果文件加载功能待实现（`loadResultFile()` 方法为占位符）
- [ ] 多序列切换功能待完善
- [ ] 工具栏按钮位置需进一步调整（用户反馈太靠下）

### 📊 测试数据

- 成功加载 100 个 CSV 文件（`/home/douff/数据标注/data/标注数据`）
- 验证数据格式：date, category, value
- 验证数据量：5000 条记录

---

## [0.1.8] - 2025-12-21 晚上

### 🔧 架构重构

#### 单页应用改造
- **删除独立路由** - 移除 Labeler、Help、License 独立页面
- **功能统一到 Index** - 将标注功能直接集成到 Index.vue 主页面
- **简化路由配置** - 单一路由 `/`，避免状态丢失和数据传递问题
- **集成文件和标签管理** - 左侧面板直接显示文件列表和标签管理

#### 数据转换优化
- **新增 dataTransform.js** - 创建统一的数据转换工具函数
- **修复时间格式问题** - 保持 ISO 字符串格式传递给 D3，由 D3 内部转换
- **规范化数据结构** - 统一 API 返回数据到 D3 所需格式的转换流程

#### 代码改动
**Frontend**:
- `router/index.js` - 删除 labeler/help/license 路由（-19 行）
- `utils/dataTransform.js` - 新增数据转换工具（+130 行）
- `views/Index.vue` - 大幅重构，集成标注功能（+637/-756 行）
- `views/Labeler.vue` - 保留但不再作为独立路由使用
- `components/BaseNavbar.vue` - 简化导航栏（-5 行）

**Backend**:
- `app.py` - 优化 API 响应格式（+133 行）

---

## [0.1.5] - 2025-12-21 下午

### 📚 文档完善

#### 新增文档
- **README.md** - 项目说明、快速开始指南
- **01-feature-list.md** - 功能清单对比（V1 vs TRAINSET vs V2）
- **02-issues-list.md** - 详细的问题清单和根因分析
  - P0 问题：Labeler 页面崩溃、标签管理为空
  - P1 问题：目录浏览缺失、布局不合理
  - P2 问题：品牌和主题问题
- **03-development-plan.md** - 开发方案和阶段划分
- **04-api-reference.md** - 完整的 API 文档
- **05-label-config-guide.md** - 标签配置说明
- **06-architecture-decision.md** - 架构决策文档（SPA vs MPA）

#### D3.js LabelerD3 优化
- **代码格式化** - 516 行代码重新格式化
- **注释优化** - 添加关键函数注释
- **变量命名优化** - 提升可读性（-258/+258 行）

#### Frontend 配置
- **添加 .npmrc** - NPM 配置文件

---

## [0.1.0] - 2025-12-21 下午 (初始版本)

### 🎉 项目初始化

#### 项目集成
- **TRAINSET 前端集成** - 移植 TRAINSET 的 Vue 2.x 前端代码
- **Flask 后端集成** - 集成 timeseries-annotator-v1 的 Flask API
- **D3.js 图表** - 集成 TRAINSET 的 D3.js 时序图表渲染

#### 核心功能（继承自 TRAINSET）
- **文件上传和管理** - CSV 文件上传和服务器文件列表
- **标签配置** - `labels.json` 配置文件支持
  - 整体属性标签（Overall Attribute）
  - 局部变化标签（Local Change）
- **D3 交互式图表**
  - 时间序列可视化
  - Context Bar 导航
  - 点击/拖拽标注
- **标注导出** - JSON 格式导出

#### 项目结构
```
timeseries-annotator-v2/
├── backend/
│   ├── app.py              # Flask API 服务器
│   ├── config/
│   │   └── labels.json     # 标签配置
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── views/
│   │   │   ├── Index.vue     # 主页
│   │   │   ├── Labeler.vue   # 标注页面
│   │   │   ├── Help.vue      # 帮助页面
│   │   │   └── License.vue   # 许可证页面
│   │   ├── components/
│   │   │   ├── BaseNavbar.vue
│   │   │   ├── BaseView.vue
│   │   │   ├── LabelerModal.vue
│   │   │   └── LabelerInstruction.vue
│   │   ├── assets/js/
│   │   │   └── LabelerD3.js  # D3 图表核心
│   │   ├── mixins/
│   │   │   └── LabelerColor.js
│   │   └── router/index.js
│   ├── build/              # Webpack 配置
│   ├── static/             # 静态资源
│   │   ├── files/          # 示例 CSV 文件
│   │   └── trainset_logo.png
│   └── package.json
└── .gitignore
```

#### 技术栈
- **Frontend**: Vue.js 2.x + Webpack + D3.js
- **Backend**: Flask + Python
- **数据存储**: JSON 文件

#### 示例数据
- `colorlist.csv` - 颜色列表示例
- `sample_trainset.csv` - TRAINSET 示例数据集


