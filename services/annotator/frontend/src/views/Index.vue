<template>
  <div class="app-container">
    <!-- Navbar -->
    <nav class="navbar">
      <h1 class="navbar-brand">📊 时序标注工具</h1>
      <span class="navbar-file" v-if="selectedFileName">{{ selectedFileName }}</span>
      <div class="navbar-user">
        <span class="user-name">👤 {{ currentUser }}</span>
        <button class="btn-logout" @click="logout">登出</button>
      </div>
    </nav>
    
    <!-- Main Layout -->
    <div class="main-layout" :class="{ 'no-file': !isChartMode }">
      <!-- Left Sidebar -->
      <aside class="sidebar left-sidebar">
        <!-- 数据管理 - 合并路径和文件 -->
        <div class="panel-card">
          <div class="panel-card-header">
            <span class="panel-card-title">📁 数据管理</span>
            <button class="btn-icon-sm" @click="refreshFiles" title="刷新">🔄</button>
          </div>
          <!-- 标签页切换 -->
          <div class="file-tabs">
            <button class="file-tab" :class="{ active: fileTab === 'csv' }" @click="fileTab = 'csv'">📄 原始数据</button>
            <button class="file-tab" :class="{ active: fileTab === 'json' }" @click="fileTab = 'json'"> 标注结果</button>
          </div>
          <!-- 路径输入 -->
          <div class="path-control">
            <input type="text" v-model="dataPath" placeholder="输入路径" class="input input-sm" @keyup.enter="setDataPath">
            <button class="btn btn-primary btn-xs" @click="openDirBrowser">📂</button>
          </div>
          <p class="current-path" v-if="currentPath">{{ currentPath }}</p>
          <div class="sort-control" v-if="fileTab === 'csv' && csvFiles.length > 0">
            <label>排序:</label>
            <select v-model="fileSortBy" class="sort-select">
              <option value="name">名称</option>
              <option value="annotation">标注数</option>
            </select>
          </div>
          <!-- CSV 文件列表 -->
          <div class="file-list" v-show="fileTab === 'csv'">
            <div v-for="file in csvFiles" :key="file.name" class="file-item" :class="{ active: file.name === selectedFileName }" @click="selectFile(file)">
              <span class="file-name">{{ file.name }}</span>
              <span v-if="file.has_annotations" class="file-badge" :title="`${file.annotation_count} 个标注`">✓ {{ file.annotation_count }}</span>
            </div>
            <p v-if="csvFiles.length === 0 && !loading" class="empty-message">暂无 CSV 文件</p>
          </div>
          <!-- JSON 结果文件列表 -->
          <div class="file-list" v-show="fileTab === 'json'">
            <div v-for="file in jsonFiles" :key="file.name" class="file-item" :class="{ active: file.name === selectedResultFile }" @click="loadResultFile(file)">
              <span class="file-name">{{ file.name }}</span>
              <span class="file-badge" v-if="file.annotation_count">✓</span>
            </div>
            <p v-if="jsonFiles.length === 0" class="empty-message">暂无标注结果</p>
          </div>
          <input type="file" ref="fileInput" @change="fileCheck" accept=".csv" style="display:none">
          <p v-if="loading" class="loading-message">加载中...</p>
        </div>

        <!-- 标签管理 -->
        <div class="panel-card">
          <div class="panel-card-header">
            <span class="panel-card-title">🏷️ 标签列表</span>
            <button class="btn btn-sm" @click="showLabelSettings = true">⚙️ 设置</button>
          </div>
          
          <!-- 整体属性 -->
          <details class="label-section" open>
            <summary>整体属性</summary>
            <div class="label-categories">
              <div v-for="(category, catId) in overallCategories" :key="catId" class="label-category">
                <span class="category-name">{{ category.name }}</span>
                <div class="label-options">
                  <label v-for="label in category.labels" :key="label.id" class="label-option">
                    <input type="radio" :name="'overall_' + catId" :value="label.id" v-model="selectedOverallLabels[catId]">
                    <span>{{ label.text }}</span>
                  </label>
                </div>
              </div>
              <p v-if="Object.keys(overallCategories).length === 0" class="empty-message">暂无标签</p>
            </div>
          </details>

          <!-- 局部变化 -->
          <details class="label-section" open>
            <summary>局部变化</summary>
            <div class="label-categories">
              <div v-for="(category, catId) in localCategories" :key="catId" class="label-category local-category">
                <span class="category-name" :style="{ color: getCategoryColor(catId) }">■ {{ category.name }}</span>
                <div class="local-label-options">
                  <div v-for="label in category.labels" :key="label.id" 
                       class="local-label-item" 
                       :class="{ active: isLocalLabelSelected(label.id) }" 
                       :style="isLocalLabelSelected(label.id) ? { backgroundColor: getLabelColor(catId, label.id) + '22', borderColor: getLabelColor(catId, label.id) } : {}"
                       @click="toggleLocalLabel(label, catId)">
                    <span class="label-color-dot" :style="{ backgroundColor: getLabelColor(catId, label.id) }"></span>
                    <span>{{ label.text }}</span>
                  </div>
                </div>
              </div>
              <p v-if="Object.keys(localCategories).length === 0" class="empty-message">暂无标签</p>
            </div>
          </details>
        </div>
      </aside>

      <!-- Main Content -->
      <main class="main-content">
        <!-- Welcome Page -->
        <div class="welcome-section" v-if="!isChartMode">
          <h2 class="title">时序数据标注工具</h2>
          <p class="subtitle">Time Series Annotation Tool v2</p>
          <button class="btn btn-lg btn-primary" @click="$refs.fileInput.click()">📤 上传CSV文件</button>
          <input type="file" ref="fileInput" @change="fileCheck" accept=".csv" style="display:none">
          <p class="hint">或在左侧选择服务器上的文件</p>
        </div>
        
        <!-- Chart Area -->
        <div class="chart-area" v-show="isChartMode">
          <!-- Hover Info -->
          <div id="hoverbox">
            <div id="hoverinfo" class="hover-card" style="display: none;">
              <div>时间: {{ hoverinfo.time }}</div>
              <div>数值: {{ hoverinfo.val }}</div>
              <div>标签: {{ hoverinfo.label }}</div>
            </div>
          </div>
          
          

          
          <!-- D3 Chart Container -->
          <!-- Instructions & Toolbar (above chart) -->
          <div class="toolbar" v-if="isChartMode" id="instrSelect">
            <div class="toolbar-row">
              <div class="toolbar-section instr compact">
                <span><strong>标注:</strong> 点击切换 | 拖拽框选 | <kbd>Shift</kbd>+拖拽取消</span>
              </div>
              <div class="toolbar-section instr compact">
                <span><strong>导航:</strong> <kbd>←</kbd><kbd>→</kbd>平移 | <kbd>↑</kbd><kbd>↓</kbd>或滚轮缩放</span>
              </div>
              <div class="toolbar-section actions-inline">
                <button class="btn btn-secondary btn-sm" @click="resetChartView">🔄 重置视图</button>
                <button class="btn btn-warning btn-sm" @click="clearAllLabels">清除标注</button>
              </div>
            </div>
            <div class="toolbar-row">
              <div class="toolbar-section selectors" id="selectors">
                <div class="selector-item"><label>主序列:</label><select id="seriesSelect"></select></div>
                <div class="selector-item"><label>参考序列:</label><select id="referenceSelect"></select></div>
              </div>
              <!-- Selection Stats (grid layout in toolbar) -->
              <div class="toolbar-section selection-stats-box" v-if="selectionStats">
                <div class="stats-header">📊 框选统计</div>
                <div class="stats-grid">
                  <span class="stat-label">索引</span><span class="stat-value">{{ selectionStats.start }} - {{ selectionStats.end }}</span>
                  <span class="stat-label">点数</span><span class="stat-value">{{ selectionStats.count }}</span>
                  <span class="stat-label">范围</span><span class="stat-value">{{ formatNumber(selectionStats.minVal) }} ~ {{ formatNumber(selectionStats.maxVal) }}</span>
                  <span class="stat-label">均值</span><span class="stat-value">{{ formatNumber(selectionStats.mean) }}</span>
                </div>
                <div class="stats-grid">
                  <span class="stat-label">标准差</span><span class="stat-value">{{ formatNumber(selectionStats.std) }}</span>
                </div>
              </div>
            </div>
          </div>
          
          <!-- D3 Chart Container -->
          <div id="maindiv"></div>
        </div>
      </main>

      <!-- Right Sidebar -->
      <aside class="sidebar right-sidebar" v-if="isChartMode">
        
        <!-- 📝 标注工作区 -->
        <div class="panel-section workspace-section">
          <h3 class="section-title">📝 标注工作区</h3>
          
          <!-- 标签 -->
          <div class="form-group">
            <label>标签</label>
            <div class="chart-labels-container" v-if="chartLabelStats.length > 0">
              <div v-for="stat in chartLabelStats" :key="stat.text" 
                   class="chart-label-tag" 
                   :class="{ 'active': activeChartLabel === stat.text }"
                   :style="{ backgroundColor: stat.color }"
                   @click="selectChartLabel(stat)">
                <span class="label-text">{{ stat.text }}</span>
                <span class="label-count">({{ stat.count }})</span>
                <button class="label-remove" @click.stop="clearLabelFromChart(stat.text)" title="清除">×</button>
              </div>
            </div>
            <div v-else class="empty-message">← 左侧选择标签后在图中框选</div>
          </div>
          
          <!-- 数据段索引（显示当前选中标签的段） -->
          <div class="form-group">
            <label v-if="activeChartLabel">数据段索引 ({{ activeSegments.length }})</label>
            <label v-else>数据段索引</label>
            <div class="segment-index-area">
              <div v-if="activeChartLabel && activeSegments.length > 0" class="segments-list">
                <div v-for="(seg, idx) in activeSegments" :key="idx" class="segment-item clickable" @click="navigateToSegment(seg)" :style="{ borderLeft: '3px solid ' + activeLabelColor }">
                  <span class="segment-range" :style="{ color: activeLabelColor }">{{ seg.start }} - {{ seg.end }}</span>
                  <span class="segment-count">({{ seg.count }}点)</span>
                  <button class="btn-icon-sm" @click.stop="removeSegmentByRange(seg)" title="删除">×</button>
                </div>
              </div>
              <div v-else-if="activeChartLabel" class="empty-placeholder">该标签暂无数据段</div>
              <div v-else-if="chartLabelStats.length > 0" class="empty-placeholder">↑ 点击标签查看数据段</div>
              <div v-else class="empty-placeholder">← 选择标签框选，或直接输入问题和分析</div>
            </div>
          </div>
          
          <!-- 问题和评价 - 始终显示 -->
          <div class="form-group">
            <label>问题</label>
            <textarea v-model="currentAnnotation.prompt" 
                      rows="2" 
                      placeholder="描述发现的问题..."></textarea>
          </div>
          <div class="form-group">
            <label>评价</label>
            <textarea v-model="currentAnnotation.expertOutput" 
                      rows="2" 
                      placeholder="评价..."></textarea>
          </div>
          
          <!-- 操作按钮 - 始终显示 -->
          <div class="form-actions">
            <button class="btn btn-primary" 
                    @click="saveActiveLabel" 
                    :disabled="!canSaveCurrentAnnotation">
              {{ editingAnnotationIndex !== null ? '更新标注' : '添加标注' }}
            </button>
            <button class="btn" 
                    @click="resetCurrentAnnotation">重置</button>
          </div>
        </div>

        <!-- 📋 标注结果 -->
        <div class="panel-section">
          <div class="section-header">
            <h3 class="section-title">📋 标注结果 ({{ savedAnnotations.length }})</h3>
            <div style="display: flex; gap: 6px;">
              <button class="btn btn-sm btn-primary" @click="saveAnnotationsToServer" :disabled="savedAnnotations.length === 0" title="保存到服务器">💾 保存</button>
              <button class="btn btn-sm" @click="downloadAnnotations" :disabled="savedAnnotations.length === 0" title="导出到本地">📥 导出</button>
            </div>
          </div>
          <div class="annotation-list">
            <div v-for="(ann, idx) in savedAnnotations" :key="ann.id" class="annotation-item" :class="{ 'editing': editingAnnotationIndex === idx }">
              <div class="annotation-header">
                <span class="label-tag clickable" :style="{ backgroundColor: ann.label.color }" @click="cycleAnnotationSegments(idx)" :title="'点击定位数据段'">{{ ann.label.text }}</span>
                <span class="segment-summary">({{ ann.segments.length }}段)</span>
                <div class="annotation-actions">
                  <button class="btn-icon-sm" @click="editAnnotation(idx)" title="编辑">✏️</button>
                  <button class="btn-delete" @click="deleteAnnotation(idx)" title="删除">×</button>
                </div>
              </div>
              <div class="annotation-segments">
                <span v-for="(seg, sidx) in ann.segments" :key="sidx" class="segment-badge clickable" @click="navigateToAnnotationSegment(ann, sidx)">
                  {{ seg.start }}-{{ seg.end }}
                </span>
              </div>
              <div class="annotation-text" v-if="ann.prompt">
                <small>Q: {{ ann.prompt.substring(0, 50) }}{{ ann.prompt.length > 50 ? '...' : '' }}</small>
              </div>
            </div>
            <p v-if="savedAnnotations.length === 0" class="empty-message">暂无标注</p>
          </div>
        </div>
      </aside>
    </div>

    <!-- Toast -->
    <div v-if="toast.show" class="toast" :class="toast.type">{{ toast.message }}</div>
    
    <!-- Directory Browser Modal -->
    <div v-if="showDirBrowser" class="modal-overlay" @click.self="showDirBrowser = false">
      <div class="modal-box">
        <div class="modal-header">
          <h3>📂 浏览目录</h3>
          <button class="close-btn" @click="showDirBrowser = false">&times;</button>
        </div>
        <div class="modal-body">
          <div class="browser-toolbar">
            <button class="btn btn-sm" @click="goToParentDir">⬆ 上级</button>
            <input type="text" v-model="browsePath" @keyup.enter="loadDirectory(browsePath)" class="input">
            <button class="btn btn-sm btn-primary" @click="loadDirectory(browsePath)">转到</button>
          </div>
          <div class="dir-list">
            <div v-for="dir in directories" :key="dir.path" class="dir-item" :class="{ 'has-data': dir.has_data_files }" @click="loadDirectory(dir.path)">
              <span>📁 {{ dir.name }}</span>
              <span v-if="dir.has_data_files" class="data-badge">含数据</span>
            </div>
            <p v-if="directories.length === 0" class="empty-message">无子目录</p>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn" @click="showDirBrowser = false">取消</button>
          <button class="btn btn-primary" @click="selectCurrentDir">选择</button>
        </div>
      </div>
    </div>
    
    <!-- Add Label Modal -->
    <div v-if="showAddLabelModal" class="modal-overlay" @click.self="showAddLabelModal = false">
      <div class="modal-box modal-sm">
        <div class="modal-header">
          <h3>添加标签</h3>
          <button class="close-btn" @click="showAddLabelModal = false">&times;</button>
        </div>
        <div class="modal-body">
          <input type="text" v-model="newLabelName" placeholder="输入标签名称" class="input" @keyup.enter="addLabel">
        </div>
        <div class="modal-footer">
          <button class="btn" @click="showAddLabelModal = false">取消</button>
          <button class="btn btn-primary" @click="addLabel">添加</button>
        </div>
      </div>
    </div>
    
    <!-- Label Settings Modal -->
    <div v-if="showLabelSettings" class="modal-overlay" @click.self="showLabelSettings = false">
      <div class="modal-box modal-lg">
        <div class="modal-header">
          <h3>🏷️ 标签管理</h3>
          <button class="close-btn" @click="showLabelSettings = false">&times;</button>
        </div>
        <div class="modal-body">
          <!-- 标签页切换：整体属性 / 局部变化 -->
          <div class="label-settings-tabs">
            <button class="settings-tab" :class="{ active: labelSettingsTab === 'overall' }" @click="labelSettingsTab = 'overall'">
              整体属性
            </button>
            <button class="settings-tab" :class="{ active: labelSettingsTab === 'local' }" @click="labelSettingsTab = 'local'">
              局部变化
            </button>
          </div>
          
          <!-- 分类列表 -->
          <div class="category-editor-list">
            <div v-for="(cat, catId) in editableCategories" :key="catId" class="category-editor-card">
              <div class="category-editor-header">
                <input v-model="cat.name" class="input input-sm category-name-input" placeholder="分类名称">
                <div class="category-actions">
                  <input v-if="labelSettingsTab === 'local'" type="color" v-model="cat.color" class="color-picker" title="分类颜色">
                  <button class="btn-icon-danger" @click="deleteCategory(catId)" title="删除分类">🗑️</button>
                </div>
              </div>
              <div class="label-editor-list">
                <div v-for="(label, idx) in cat.labels" :key="label.id" class="label-editor-item">
                  <input v-model="label.text" class="input input-xs label-name-input" placeholder="标签名">
                  <input v-if="labelSettingsTab === 'local'" type="color" v-model="label.color" class="color-picker-sm" title="标签颜色">
                  <button class="btn-icon-sm" @click="deleteLabelFromCategory(catId, idx)" title="删除">×</button>
                </div>
                <button class="btn btn-xs btn-outline" @click="addLabelToCategory(catId)">+ 添加标签</button>
              </div>
            </div>
            <button class="btn btn-primary btn-sm add-category-btn" @click="addCategory">+ 添加分类</button>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn" @click="showLabelSettings = false">取消</button>
          <button class="btn btn-primary" @click="saveLabelsToServer">保存</button>
        </div>
      </div>
    </div>
    
    <!-- Hidden triggers for D3 -->
    <button id="updateHover" style="display:none" @click="updateHoverinfo"></button>
    <button id="triggerReplot" style="display:none" @click="triggerReplot"></button>
    <button id="triggerRecolor" style="display:none" @click="triggerRecolor"></button>
    <button id="clearSeries" style="display:none" @click="clearSeries"></button>
    <button id="updateSelection" style="display:none" @click="updateSelectionRange"></button>
  </div>
</template>

<script>
import * as LabelerD3 from "@/assets/js/LabelerD3.js"
const { DateTime } = require("luxon");

// Use current hostname for API - supports dev (localhost) and remote access
const API_BASE = `http://${window.location.hostname}:5000/api`;
var plottingApp = {};
// Expose to window for D3 access and debugging
window.plottingApp = plottingApp;

// Color palette for labels
const COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#14b8a6', '#3b82f6', '#8b5cf6', '#ec4899'];
let colorIndex = 0;

export default {
  name: 'Index',
  data() {
    return {
      // User info
      currentUser: localStorage.getItem('name') || localStorage.getItem('username') || 'User',
      // Path & Files
      dataPath: '',
      currentPath: '',
      files: [],
      loading: false,
      selectedFileName: '',
      isChartMode: false,
      
      // Labels
      labels: { overall_attribute: {}, local_change: {} },
      selectedOverallLabels: {},
      selectedLocalLabels: [],
      
      // Chart labels
      selectedLabel: '',
      optionsList: [],
      newLabelName: '',
      
      // Hover info
      hoverinfo: { val: '', time: '', label: '' },
      
      // Annotations - New structure: one label to multiple segments
      currentAnnotation: {
        label: null,           // Currently selected local label
        segments: [],          // Array of {start, end, count}
        prompt: '',
        expertOutput: ''
      },
      annotationVersion: 0,    // Increment to force reactivity updates
      savedAnnotations: [],    // Array of completed annotations
      selectionRange: '未选择',
      selectionStats: null,    // Current selection statistics
      chartDataVersion: 0,  // Track chart data changes for reactivity
      
      // UI state
      toast: { show: false, message: '', type: 'info' },
      showDirBrowser: false,
      showAddLabelModal: false,
      showLabelSettings: false,
      browsePath: '',
      parentPath: '',
      directories: [],
      fileTab: 'csv',
      fileSortBy: 'name',  // 'name' or 'annotation'
      selectedResultFile: '',
      
      // Category colors for local changes - each major category gets one color
      categoryColors: {
        'outlier': '#ef4444',
        'level_shift': '#3b82f6',
        'concept_drift': '#22c55e',
        'seasonal': '#f59e0b',
        'trend': '#8b5cf6',
        'spike': '#ef4444',
        'step': '#22c55e',
        'drift': '#3b82f6',
        'anomaly': '#a855f7',
        'default': '#6b7280'
      },
      
      // Label settings modal state
      labelSettingsTab: 'overall',
      
      // Editing state: index of annotation being edited, null if creating new
      editingAnnotationIndex: null,
      
      // Track segment cycle position for each annotation (key: annotation index)
      annotationCyclePositions: {},
      
      // Currently active label in workspace (for viewing its segments)
      activeChartLabel: null
    }
  },
  computed: {
    overallCategories() {
      // Return direct reference for display (read-only)
      return this.labels.overall_attribute || {};
    },
    localCategories() {
      // Return direct reference for display (read-only)
      return this.labels.local_change || {};
    },
    canSaveAnnotation() {
      return this.currentAnnotation.label !== null && this.currentAnnotation.segments.length > 0;
    },
    // 新增：是否可以保存当前标注（允许仅问题和分析）
    canSaveCurrentAnnotation() {
      const hasSegments = this.activeChartLabel && this.activeSegments.length > 0;
      const hasContent = (this.currentAnnotation.prompt || '').trim() || 
                        (this.currentAnnotation.expertOutput || '').trim();
      return hasSegments || hasContent;
    },
    // Filter and sort files for CSV tab
    csvFiles() {
      const filtered = this.files.filter(f => f.name.toLowerCase().endsWith('.csv'));
      return this.sortFiles(filtered, this.fileSortBy);
    },
    // Filter files for JSON results
    jsonFiles() {
      return this.files.filter(f => f.name.toLowerCase().endsWith('.json') || f.has_annotations);
    },
    // Editable categories for label settings modal - return direct reference
    editableCategories() {
      if (this.labelSettingsTab === 'overall') {
        return this.labels.overall_attribute || {};
      }
      return this.labels.local_change || {};
    },
    // Get all used colors to avoid duplicates
    usedColors() {
      const colors = new Set();
      const localCats = this.labels.local_change || {};
      Object.values(localCats).forEach(cat => {
        if (cat.labels) {
          cat.labels.forEach(label => {
            if (label.color) colors.add(label.color);
          });
        }
      });
      return colors;
    },
    // Calculate chart label statistics from plottingApp.allData
    chartLabelStats() {
      // Touch chartDataVersion to make this reactive
      const _version = this.chartDataVersion;
      if (!window.plottingApp || !window.plottingApp.allData) return [];
      
      const stats = {};
      window.plottingApp.allData.forEach(d => {
        if (d.label && d.label !== '') {
          if (!stats[d.label]) {
            stats[d.label] = { text: d.label, count: 0, color: null };
          }
          stats[d.label].count++;
        }
      });
      
      // Associate colors from labelList
      return Object.values(stats).map(s => {
        const labelEntry = window.plottingApp.labelList?.find(l => l.name === s.text);
        s.color = labelEntry?.color || '#7E4C64';
        return s;
      });
    },
    // Computed proxy for segments with forced reactivity
    segmentsList() {
      // Touch annotationVersion to ensure reactivity
      const _v = this.annotationVersion;
      return this.currentAnnotation.segments || [];
    },
    segmentsCount() {
      return this.segmentsList.length;
    },
    annotationLabel() {
      const _v = this.annotationVersion;
      return this.currentAnnotation.label;
    },
    // Get segments for the currently active label in workspace
    activeSegments() {
      const _v = this.chartDataVersion;
      if (!this.activeChartLabel || !window.plottingApp || !window.plottingApp.allData) return [];
      
      // Find all points with this label
      const labeledPoints = window.plottingApp.allData.filter(d => d.label === this.activeChartLabel);
      if (labeledPoints.length === 0) return [];
      
      // Sort by time (numeric index), not id
      const indices = labeledPoints.map(d => parseInt(d.time) || 0).sort((a, b) => a - b);
      
      // Group into contiguous segments
      const segments = [];
      let segStart = indices[0];
      let segEnd = indices[0];
      
      for (let i = 1; i < indices.length; i++) {
        if (indices[i] === segEnd + 1) {
          // Contiguous, extend segment
          segEnd = indices[i];
        } else {
          // Gap found, save current segment and start new one
          segments.push({
            start: segStart,
            end: segEnd,
            count: segEnd - segStart + 1
          });
          segStart = indices[i];
          segEnd = indices[i];
        }
      }
      
      // Push final segment
      segments.push({
        start: segStart,
        end: segEnd,
        count: segEnd - segStart + 1
      });
      
      return segments;
    },
    // Get color for the currently active label
    activeLabelColor() {
      if (!this.activeChartLabel) return '#7E4C64';
      const stat = this.chartLabelStats.find(s => s.text === this.activeChartLabel);
      return stat?.color || '#7E4C64';
    }
  },
  watch: {
    selectedLabel(val) {
      if (plottingApp) plottingApp.selectedLabel = val;
    }
  },
  mounted() {
    // Expose Vue instance to window for D3 direct access
    window.vueApp = this;
    console.log('Vue instance exposed to window.vueApp');
    
    this.loadLabels();
    this.loadCurrentPath();
  },
  methods: {
    // User logout
    logout() {
      localStorage.removeItem('token');
      localStorage.removeItem('username');
      localStorage.removeItem('name');
      this.$router.push('/login');
    },
    
    // API Methods
    async loadLabels() {
      try {
        console.log('Loading labels from:', `${API_BASE}/labels`);
        const res = await fetch(`${API_BASE}/labels`);
        const data = await res.json();
        console.log('Labels API response:', data);
        if (data.success) {
          this.labels = data.labels;
          console.log('Loaded labels - overall:', Object.keys(this.labels.overall_attribute || {}));
          console.log('Loaded labels - local:', Object.keys(this.labels.local_change || {}));
          // Initialize selected labels
          Object.keys(this.labels.overall_attribute || {}).forEach(catId => {
            this.$set(this.selectedOverallLabels, catId, '');
          });
          // Sync category colors from labels.json
          this.updateCategoryColors();
        } else {
          console.error('Labels API error:', data.error);
        }
      } catch (e) {
        console.error('Failed to load labels:', e);
      }
    },
    
    async loadCurrentPath() {
      try {
        const res = await fetch(`${API_BASE}/current-path`, {
          headers: this.getAuthHeaders()
        });
        const data = await res.json();
        if (data.success && data.path) {
          this.currentPath = data.path;
          this.dataPath = data.path;
          await this.loadFiles();
        } else if (res.status === 401) {
          this.$router.push('/login');
        }
      } catch (e) {
        console.error('Failed to load current path:', e);
      }
    },
    
    async setDataPath() {
      if (!this.dataPath) {
        this.showToast('请输入路径', 'error');
        return;
      }
      
      try {
        const res = await fetch(`${API_BASE}/set-path`, {
          method: 'POST',
          headers: this.getAuthHeaders(),
          body: JSON.stringify({ path: this.dataPath })
        });
        const data = await res.json();
        if (data.success) {
          // Update display path
          this.currentPath = data.path;
          this.dataPath = data.path;
          this.showToast('路径已设置', 'success');
          // Refresh file list
          await this.loadFiles();
        } else if (res.status === 401) {
          this.$router.push('/login');
        } else {
          this.showToast('路径设置失败: ' + data.error, 'error');
        }
      } catch (e) {
        console.error('Set path error:', e);
        this.showToast('路径设置失败', 'error');
      }
    },
    
    // Natural sort for filenames with numbers
    naturalSort(a, b) {
      const ax = [];
      const bx = [];
      
      a.replace(/(\d+)|(\D+)/g, (_, num, str) => { ax.push([num || Infinity, str || '']); });
      b.replace(/(\d+)|(\D+)/g, (_, num, str) => { bx.push([num || Infinity, str || '']); });
      
      while (ax.length && bx.length) {
        const an = ax.shift();
        const bn = bx.shift();
        const nn = (an[0] - bn[0]) || an[1].localeCompare(bn[1]);
        if (nn) return nn;
      }
      
      return ax.length - bx.length;
    },
    
    // Sort files by different criteria
    sortFiles(files, sortBy) {
      const sorted = [...files];
      
      if (sortBy === 'annotation') {
        // Sort by annotation count (descending), then by name
        sorted.sort((a, b) => {
          const countDiff = (b.annotation_count || 0) - (a.annotation_count || 0);
          if (countDiff !== 0) return countDiff;
          return this.naturalSort(a.name.toLowerCase(), b.name.toLowerCase());
        });
      } else {
        // Sort by name (natural sort)
        sorted.sort((a, b) => this.naturalSort(a.name.toLowerCase(), b.name.toLowerCase()));
      }
      
      return sorted;
    },
    
    async loadFiles() {
      if (!this.currentPath) return;
      
      try {
        const url = `${API_BASE}/files?path=${encodeURIComponent(this.currentPath)}`;
        
        const res = await fetch(url, {
          headers: this.getAuthHeaders()
        });
        const data = await res.json();
        
        if (data.success) {
          const allFiles = data.files || [];
          
          // Set files array - csvFiles and jsonFiles are computed from this
          this.files = allFiles;
          this.currentPath = data.path || this.currentPath;
          
          const csvCount = allFiles.filter(f => f.name.endsWith('.csv')).length;
          this.showToast(`已加载 ${csvCount} 个CSV文件`, 'success');
        } else if (res.status === 401) {
          this.$router.push('/login');
        } else {
          console.error('Load files failed:', data.error);
          this.showToast('文件加载失败: ' + (data.error || '未知错误'), 'error');
        }
      } catch (e) {
        console.error('Load files error:', e);
        this.showToast('文件加载失败: ' + e.message, 'error');
      }
    },
    
    // Alias for backward compatibility
    refreshFiles() {
      return this.loadFiles();
    },
    
    // Load JSON annotation result file for review/edit
    async loadResultFile(file) {
      this.selectedResultFile = file.name;
      this.showToast('加载标注结果: ' + file.name, 'info');
      // TODO: Implement actual JSON loading logic
      // This would load the annotations and overlay them on the chart
    },
    
    // Select a CSV file to load
    async selectFile(file) {
      if (!file || !file.name) return;
      
      // Reset states
      this.currentAnnotation = { label: null, segments: [], prompt: '', expertOutput: '' };
      this.selectedLocalLabels = [];
      this.savedAnnotations = [];
      this.activeChartLabel = '';
      this.editingAnnotationIndex = null;
      
      // Load data from API
      this.selectedFileName = file.name;
      this.loading = true;
      
      try {
        const res = await fetch(`${API_BASE}/data/${file.name}`, {
          headers: this.getAuthHeaders()
        });
        const data = await res.json();
        
        if (data.success) {
          // Initialize chart with data
          this.initChart(data.data, file.name, data.seriesList, data.labelList || []);
          
          // Load saved annotations for this file
          await this.loadAnnotationsForFile(file.name);
        } else if (res.status === 401) {
          this.$router.push('/login');
        } else {
          this.showToast('加载失败: ' + data.error, 'error');
        }
      } catch (e) {
        console.error('Load file error:', e);
        this.showToast('加载失败: ' + e.message, 'error');
      } finally {
        this.loading = false;
      }
    },
    
    // Chart Initialization
    initChart(csvData, filename, seriesList, labelList) {
      // Switch to chart mode first so DOM renders
      this.isChartMode = true;
      
      // Use nextTick to ensure DOM is rendered before D3 draws
      this.$nextTick(() => {
        // Clear previous chart
        const maindiv = document.getElementById('maindiv');
        if (maindiv) maindiv.innerHTML = '';
        
        // Setup plottingApp with data
        plottingApp.filename = filename;
        plottingApp.csvData = csvData;
        plottingApp.seriesList = seriesList;
        plottingApp.labelList = labelList.length > 0 ? labelList : ['label_1'];
        
        // CRITICAL: Pre-set selectedSeries before D3 initialization
        plottingApp.selectedSeries = seriesList[0] || 'value';
        plottingApp.refSeries = seriesList.length > 1 ? seriesList[1] : seriesList[0];
        
        // Setup selectors in DOM
        this.setupSelectors(seriesList);
        
        // Map labels to colors (but don't auto-select any label)
        this.optionsList = plottingApp.labelList.map(l => ({ name: l, color: this.getNextColor() }));
        plottingApp.labelList = this.optionsList;
        // Don't auto-select a label - user must explicitly choose one
        this.selectedLabel = '';
        plottingApp.selectedLabel = '';
        
        // Draw chart with slight delay to ensure container has width
        setTimeout(() => {
          try {
            LabelerD3.drawLabeler(plottingApp);
          } catch (e) {
            console.error('Chart draw error:', e);
            this.showToast('图表绘制失败: ' + e.message, 'error');
          }
        }, 100);
      });
    },
    
    setupSelectors(seriesList) {
      const seriesSelect = document.getElementById('seriesSelect');
      const refSelect = document.getElementById('referenceSelect');
      if (!seriesSelect || !refSelect) return;
      
      seriesSelect.innerHTML = '';
      refSelect.innerHTML = '';
      
      seriesList.forEach(s => {
        seriesSelect.innerHTML += `<option value="${s}">${s}</option>`;
        refSelect.innerHTML += `<option value="${s}">${s}</option>`;
      });
      
      // Always show selectors - even with single series
      document.getElementById('selectors').style.display = 'flex';
    },
    
    getNextColor() {
      const color = COLORS[colorIndex % COLORS.length];
      colorIndex++;
      return color;
    },
    
    // Label Methods - Single select for local labels
    toggleLocalLabel(label, categoryId) {
      console.log('=== toggleLocalLabel called ===');
      console.log('  - label:', label);
      console.log('  - categoryId:', categoryId);
      console.log('  - currentAnnotation before:', JSON.stringify(this.currentAnnotation));
      
      // Get label color
      const labelColor = this.getLabelColor(categoryId, label.id);
      
      // Check if clicking same label
      if (this.currentAnnotation.label && this.currentAnnotation.label.id === label.id) {
        // Clicked same label - deselect - use $set for reactivity
        console.log('  - Deselecting same label');
        this.$set(this.currentAnnotation, 'label', null);
        this.selectedLocalLabels = [];
        if (plottingApp) {
          plottingApp.selectedLabel = '';
          plottingApp.labelColor = null;
        }
      } else {
        // Set new label
        const labelObj = {
          id: label.id,
          text: label.text,
          color: labelColor,
          categoryId,
          categoryName: this.localCategories[categoryId]?.name
        };
        console.log('  - Setting new label:', labelObj);
        // Use $set for reactivity
        this.$set(this.currentAnnotation, 'label', labelObj);
        this.selectedLocalLabels = [labelObj];  // Keep for backward compatibility
        
        // Update D3 chart with this color for brush labeling
        if (plottingApp) {
          plottingApp.selectedLabel = label.text;
          plottingApp.labelColor = labelColor;
          console.log('  - Set plottingApp.selectedLabel to:', label.text);
          
          // Update labelList
          if (!plottingApp.labelList) plottingApp.labelList = [];
          const existingIdx = plottingApp.labelList.findIndex(l => l.name === label.text);
          if (existingIdx === -1) {
            plottingApp.labelList.push({ name: label.text, color: labelColor });
          } else {
            plottingApp.labelList[existingIdx].color = labelColor;
          }
        }
      }
      
      // Increment version to trigger computed property updates
      this.annotationVersion++;
      console.log('  - currentAnnotation after:', JSON.stringify(this.currentAnnotation));
      console.log('  - plottingApp.selectedLabel:', plottingApp?.selectedLabel);
    },
    
    isLocalLabelSelected(labelId) {
      return this.currentAnnotation.label && this.currentAnnotation.label.id === labelId;
    },
    
    getCategoryColor(categoryId) {
      // Priority: 1. Color from labels.json category 2. categoryColors map 3. default
      const localCat = this.labels.local_change?.[categoryId];
      if (localCat && localCat.color) {
        return localCat.color;
      }
      return this.categoryColors[categoryId] || this.categoryColors['default'];
    },
    
    // Get label-specific color (for individual labels within a category)
    getLabelColor(categoryId, labelId) {
      const localCat = this.labels.local_change?.[categoryId];
      if (localCat && localCat.labels) {
        const label = localCat.labels.find(l => l.id === labelId);
        if (label && label.color) {
          return label.color;
        }
      }
      // Fallback to category color
      return this.getCategoryColor(categoryId);
    },
    
    addLabel() {
      if (!this.newLabelName || this.optionsList.some(l => l.name === this.newLabelName)) {
        this.showToast('标签名无效或已存在', 'error');
        return;
      }
      this.optionsList.push({ name: this.newLabelName, color: this.getNextColor() });
      plottingApp.labelList = this.optionsList;
      this.selectedLabel = this.newLabelName;
      this.newLabelName = '';
      this.showAddLabelModal = false;
    },
    
    removeLabel() {
      if (this.optionsList.length <= 1) return;
      const idx = this.optionsList.findIndex(l => l.name === this.selectedLabel);
      if (idx > -1) {
        this.optionsList.splice(idx, 1);
        plottingApp.labelList = this.optionsList;
        this.selectedLabel = this.optionsList[0].name;
      }
    },
    
    // Annotation Methods - New workflow
    saveCurrentAnnotation() {
      if (!this.canSaveAnnotation) return;
      
      // Group segments by their individual labels (using label.id as key)
      const segmentsByLabel = {};
      this.currentAnnotation.segments.forEach(seg => {
        const labelId = seg.label?.id || 'unknown';
        if (!segmentsByLabel[labelId]) {
          segmentsByLabel[labelId] = {
            label: seg.label,
            segments: []
          };
        }
        // Store segment without the embedded label (avoid duplication in saved data)
        const { label: _, ...segmentData } = seg;
        segmentsByLabel[labelId].segments.push(segmentData);
      });
      
      // Create one annotation per unique label
      const labelIds = Object.keys(segmentsByLabel);
      let savedCount = 0;
      
      labelIds.forEach(labelId => {
        const group = segmentsByLabel[labelId];
        if (!group.label) return; // Skip if no label
        
        const annotation = {
          id: 'ann_' + Date.now() + '_' + labelId,
          label: { ...group.label },
          segments: group.segments,
          prompt: this.currentAnnotation.prompt,
          expertOutput: this.currentAnnotation.expertOutput,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        };
        
        // Check if we're editing an existing annotation with this label
        const existingIdx = this.savedAnnotations.findIndex(
          ann => ann.label.id === labelId
        );
        
        if (existingIdx !== -1) {
          // Merge segments into existing annotation
          const existing = this.savedAnnotations[existingIdx];
          annotation.id = existing.id;
          annotation.createdAt = existing.createdAt;
          annotation.segments = [...existing.segments, ...group.segments];
          this.$set(this.savedAnnotations, existingIdx, annotation);
        } else {
          this.savedAnnotations.push(annotation);
        }
        savedCount++;
      });
      
      if (savedCount > 0) {
        this.showToast(`已保存 ${savedCount} 个标签的标注`, 'success');
        // Auto-save to server after saving annotations
        this.saveAnnotationsToServer();
      }
      this.resetCurrentAnnotation();
    },
    
    resetCurrentAnnotation() {
      this.currentAnnotation = {
        label: null,
        segments: [],
        prompt: '',
        expertOutput: ''
      };
      this.selectedLocalLabels = [];
      this.selectionRange = '未选择';
      this.selectionStats = null;  // Clear selection stats
      this.editingAnnotationIndex = null;  // Reset editing state
      if (plottingApp) {
        plottingApp.selectedLabel = '';
        plottingApp.labelColor = null;
      }
    },
    
    // Select a chart label to view its segments in workspace
    selectChartLabel(stat) {
      if (!stat || !stat.text) return;
      
      // Toggle: if same label is clicked, deselect
      if (this.activeChartLabel === stat.text) {
        this.activeChartLabel = null;
        return;
      }
      
      this.activeChartLabel = stat.text;
      
      // Update plottingApp for visual sync
      if (plottingApp) {
        plottingApp.selectedLabel = stat.text;
        plottingApp.labelColor = stat.color;
      }
      
      // Find label info for currentAnnotation
      let labelObj = { id: stat.text, text: stat.text, color: stat.color };
      const localCats = this.labels.local_change || {};
      for (const [catId, cat] of Object.entries(localCats)) {
        const foundLabel = cat.labels?.find(l => l.text === stat.text);
        if (foundLabel) {
          labelObj = { ...foundLabel, color: stat.color, categoryId: catId, categoryName: cat.name };
          break;
        }
      }
      
      // Update currentAnnotation with this label
      this.currentAnnotation.label = labelObj;
      this.selectedLocalLabels = [labelObj];
      this.annotationVersion++;
    },
    
    // Save all labels with segments as annotations
    saveActiveLabel() {
      // Get all labels that have segments
      const labelsWithSegments = this.chartLabelStats.filter(stat => stat.count > 0);
      
      // Check if we have content-only annotation (no labels selected)
      const hasContent = (this.currentAnnotation.prompt || '').trim() || 
                        (this.currentAnnotation.expertOutput || '').trim();
      
      if (labelsWithSegments.length === 0 && !hasContent) {
        this.showToast('请至少选择标签框选数据段，或输入问题/评价', 'error');
        return;
      }
      
      let savedCount = 0;
      
      // Save each label with its segments
      for (const stat of labelsWithSegments) {
        // Find label info
        let labelObj = { id: stat.text, text: stat.text, color: stat.color };
        const localCats = this.labels.local_change || {};
        for (const [catId, cat] of Object.entries(localCats)) {
          const foundLabel = cat.labels?.find(l => l.text === stat.text);
          if (foundLabel) {
            labelObj = { ...foundLabel, color: stat.color, categoryId: catId, categoryName: cat.name };
            break;
          }
        }
        
        // Get segments for this label
        const labeledPoints = window.plottingApp.allData.filter(d => d.label === stat.text);
        if (labeledPoints.length === 0) continue;
        
        // Sort by time and group into segments
        const indices = labeledPoints.map(d => parseInt(d.time) || 0).sort((a, b) => a - b);
        const segments = [];
        let segStart = indices[0];
        let segEnd = indices[0];
        
        for (let i = 1; i < indices.length; i++) {
          if (indices[i] === segEnd + 1) {
            segEnd = indices[i];
          } else {
            segments.push({
              start: segStart,
              end: segEnd,
              count: segEnd - segStart + 1
            });
            segStart = indices[i];
            segEnd = indices[i];
          }
        }
        segments.push({
          start: segStart,
          end: segEnd,
          count: segEnd - segStart + 1
        });
        
        const annotation = {
          id: Date.now() + savedCount,
          label: labelObj,
          segments: segments,
          prompt: this.currentAnnotation.prompt || '',
          expertOutput: this.currentAnnotation.expertOutput || ''
        };
        
        // Check for existing and merge/update
        const existingIdx = this.savedAnnotations.findIndex(a => 
          a.label.text === labelObj.text
        );
        
        if (existingIdx !== -1) {
          // Merge segments
          const existing = this.savedAnnotations[existingIdx];
          existing.segments = [...existing.segments, ...segments];
          existing.prompt = annotation.prompt || existing.prompt;
          existing.expertOutput = annotation.expertOutput || existing.expertOutput;
          this.$set(this.savedAnnotations, existingIdx, existing);
        } else {
          // Add new
          this.savedAnnotations.push(annotation);
        }
        
        savedCount++;
      }
      
      // If only content without labels
      if (labelsWithSegments.length === 0 && hasContent) {
        const annotation = {
          id: Date.now(),
          label: { id: 'no_label', text: '无标签', color: '#999999' },
          segments: [],
          prompt: this.currentAnnotation.prompt || '',
          expertOutput: this.currentAnnotation.expertOutput || ''
        };
        this.savedAnnotations.push(annotation);
        savedCount = 1;
      }
      
      if (savedCount > 0) {
        this.showToast(`已添加 ${savedCount} 个标注`, 'success');
      }
      
      this.resetCurrentAnnotation();
      
      // Auto-save to server
      this.saveAnnotationsToServer();
    },
    
    // Remove a segment by its range (for activeSegments)
    removeSegmentByRange(seg) {
      if (!seg || !this.activeChartLabel || !plottingApp || !plottingApp.allData) return;
      
      // Clear labels for points in this segment range
      plottingApp.allData.forEach(d => {
        const idx = parseInt(d.id);
        if (idx >= seg.start && idx <= seg.end && d.label === this.activeChartLabel) {
          d.label = '';
        }
      });
      
      // Refresh chart display
      const updatePointStyle = function(d) {
        if (d.label) {
          const labelInfo = plottingApp.labelList?.find(l => l.name === d.label);
          const color = labelInfo?.color || '#7E4C64';
          return `fill: ${color}; stroke: ${color}; opacity: 0.75;`;
        }
        return 'fill: black; stroke: none; opacity: 1;';
      };
      if (plottingApp.main) {
        plottingApp.main.selectAll('.point').attr('style', updatePointStyle);
      }
      if (plottingApp.context) {
        plottingApp.context.selectAll('.point').attr('style', updatePointStyle);
      }
      
      this.chartDataVersion++;
      this.showToast(`已删除数据段: ${seg.start} - ${seg.end}`, 'info');
    },
    
    // Navigate chart to show a specific segment
    navigateToSegment(seg) {
      if (!seg || seg.start === undefined || seg.end === undefined) return;
      this.panChartToRange(seg.start, seg.end);
      this.showToast(`定位到: ${seg.start} - ${seg.end}`, 'info');
    },
    
    // Navigate to a specific segment in a saved annotation
    navigateToAnnotationSegment(ann, segIdx) {
      if (!ann || !ann.segments || !ann.segments[segIdx]) return;
      const seg = ann.segments[segIdx];
      this.panChartToRange(seg.start, seg.end);
      this.showToast(`定位到 ${ann.label.text}: ${seg.start} - ${seg.end}`, 'info');
    },
    
    // Cycle through annotation segments when clicking on the label tag
    cycleAnnotationSegments(annIdx) {
      const ann = this.savedAnnotations[annIdx];
      if (!ann || !ann.segments || ann.segments.length === 0) return;
      
      // Get current position for this annotation
      let currentPos = this.annotationCyclePositions[annIdx] || 0;
      
      // Navigate to current segment
      const seg = ann.segments[currentPos];
      this.panChartToRange(seg.start, seg.end);
      this.showToast(`${ann.label.text}: 段 ${currentPos + 1}/${ann.segments.length} (${seg.start}-${seg.end})`, 'info');
      
      // Increment position for next click (cycle back to 0)
      this.$set(this.annotationCyclePositions, annIdx, (currentPos + 1) % ann.segments.length);
    },
    
    // Pan chart to show a specific range
    panChartToRange(start, end) {
      if (!plottingApp || !plottingApp.plot || !plottingApp.context_brush) {
        console.warn('Chart not ready for panning');
        return;
      }
      
      // Calculate padding (show some context around the segment)
      const segLen = end - start;
      const padding = Math.max(segLen * 0.5, 20);  // At least 20 points padding
      const newStart = Math.max(0, start - padding);
      const newEnd = end + padding;
      
      // Update context brush to pan main chart
      try {
        if (plottingApp.context_xscale && plottingApp.plot.context_brush) {
          const newExtent = [newStart, newEnd].map(d => plottingApp.context_xscale(d));
          plottingApp.plot.context_brush.call(plottingApp.context_brush.move, newExtent);
        }
      } catch (e) {
        console.error('Error panning chart:', e);
      }
    },
    
    // Navigate to points with a specific label on the chart
    navigateToLabelPoints(labelText) {
      if (!plottingApp || !plottingApp.allData || !labelText) return;
      
      // Find all points with this label
      const labeledPoints = plottingApp.allData
        .map((d, idx) => ({ ...d, idx }))
        .filter(d => d.label === labelText);
      
      if (labeledPoints.length === 0) {
        this.showToast(`未找到 "${labelText}" 的标注点`, 'warning');
        return;
      }
      
      // Find the range of labeled points
      const indices = labeledPoints.map(d => d.idx);
      const minIdx = Math.min(...indices);
      const maxIdx = Math.max(...indices);
      
      this.panChartToRange(minIdx, maxIdx);
      this.showToast(`定位到 ${labelText}: ${minIdx}-${maxIdx} (${labeledPoints.length}点)`, 'info');
    },
    
    // Clear a specific label from the chart
    clearLabelFromChart(labelText) {
      if (!plottingApp || !plottingApp.allData || !labelText) return;
      
      let clearedCount = 0;
      plottingApp.allData.forEach(d => {
        if (d.label === labelText) {
          d.label = '';
          clearedCount++;
        }
      });
            if (clearedCount > 0) {
          // Refresh chart display - main and context (thumbnail)
          const updatePointStyle = function(d) {
            if (d.label) {
              const labelInfo = plottingApp.labelList?.find(l => l.name === d.label);
              const color = labelInfo?.color || '#7E4C64';
              return `fill: ${color}; stroke: ${color}; opacity: 0.75;`;
            }
            return 'fill: black; stroke: none; opacity: 1;';
          };
          if (typeof plottingApp.main !== 'undefined' && plottingApp.main) {
            plottingApp.main.selectAll('.point').attr('style', updatePointStyle);
          }
          // Also update context (thumbnail) points
          if (typeof plottingApp.context !== 'undefined' && plottingApp.context) {
            plottingApp.context.selectAll('.point').attr('style', updatePointStyle);
          }
        
        // Also remove segments with this label from currentAnnotation
        const originalLength = this.currentAnnotation.segments.length;
        this.currentAnnotation.segments = this.currentAnnotation.segments.filter(
          seg => seg.label?.text !== labelText
        );
        const removedSegments = originalLength - this.currentAnnotation.segments.length;
        
        // If current label matches, clear it too
        if (this.currentAnnotation.label?.text === labelText) {
          this.currentAnnotation.label = null;
        }
        
        // Trigger reactivity update
        this.chartDataVersion++;
        this.annotationVersion++;
        
        let message = `已清除 ${labelText} 的 ${clearedCount} 个标注点`;
        if (removedSegments > 0) {
          message += `，${removedSegments} 个数据段`;
        }
        this.showToast(message, 'success');
      }
    },
    
    // Clear current label and also clear chart points with this label's color
    clearCurrentLabel() {
      const currentLabelText = this.currentAnnotation.label?.text;
      
      // Clear from plottingApp
      if (plottingApp) {
        // If we have a label, clear all points with this label from chart
        if (currentLabelText && plottingApp.allData) {
          plottingApp.allData.forEach(d => {
            if (d.label === currentLabelText) {
              d.label = '';
            }
          });
          // Refresh chart display
          if (typeof plottingApp.main !== 'undefined' && plottingApp.main) {
            plottingApp.main.selectAll('.point').attr('style', function(d) {
              return d.label ? `fill: ${plottingApp.labelColor || '#7E4C64'}; stroke: ${plottingApp.labelColor || '#7E4C64'}; opacity: 0.75;` : 'fill: black; stroke: none; opacity: 1;';
            });
          }
        }
        plottingApp.selectedLabel = '';
        plottingApp.labelColor = null;
      }
      
      // Clear current annotation state
      this.currentAnnotation.label = null;
      this.currentAnnotation.segments = [];
      this.selectedLocalLabels = [];
      this.showToast('已取消选择标签并清除相关数据点', 'info');
    },
    
    removeSegment(idx) {
      const segment = this.currentAnnotation.segments[idx];
      if (segment && plottingApp && plottingApp.allData) {
        // Clear labels for points in this segment
        const labelText = segment.label?.text || this.currentAnnotation.label?.text;
        if (labelText) {
          plottingApp.allData.forEach(d => {
            const dIdx = parseInt(d.id);
            if (dIdx >= segment.start && dIdx <= segment.end && d.label === labelText) {
              d.label = '';
            }
          });
          // Refresh chart display - main and context
          const updatePointStyle = function(d) {
            if (d.label) {
              const labelInfo = plottingApp.labelList?.find(l => l.name === d.label);
              const color = labelInfo?.color || '#7E4C64';
              return `fill: ${color}; stroke: ${color}; opacity: 0.75;`;
            }
            return 'fill: black; stroke: none; opacity: 1;';
          };
          if (plottingApp.main) {
            plottingApp.main.selectAll('.point').attr('style', updatePointStyle);
          }
          if (plottingApp.context) {
            plottingApp.context.selectAll('.point').attr('style', updatePointStyle);
          }
        }
        this.chartDataVersion++;
      }
      this.currentAnnotation.segments.splice(idx, 1);
      this.annotationVersion++;
    },
    
    deleteAnnotation(idx) {
      this.savedAnnotations.splice(idx, 1);
      this.showToast('标注已删除', 'info');
      // Auto-save after deletion
      this.saveAnnotationsToServer();
    },
    
    // Edit an existing annotation - load it into current editing area
    editAnnotation(idx) {
      const ann = this.savedAnnotations[idx];
      if (!ann) return;
      
      // Set active chart label so the workspace becomes visible
      this.activeChartLabel = ann.label.text;
      
      // Load annotation into current editing state
      this.currentAnnotation = {
        label: { ...ann.label },
        segments: [...ann.segments],
        prompt: ann.prompt || '',
        expertOutput: ann.expertOutput || ''
      };
      this.selectedLocalLabels = [ann.label];
      
      // Update plottingApp
      if (plottingApp) {
        plottingApp.selectedLabel = ann.label.text;
        plottingApp.labelColor = ann.label.color;
      }
      
      // Store editing index for update
      this.editingAnnotationIndex = idx;
      
      this.showToast('已加载标注进行编辑', 'info');
    },
    
    // Get authorization headers
    getAuthHeaders() {
      const token = localStorage.getItem('token');
      return {
        'Content-Type': 'application/json',
        'Authorization': token ? `Bearer ${token}` : ''
      };
    },
    
    // Load annotations for a specific file from server
    async loadAnnotationsForFile(filename) {
      try {
        const res = await fetch(`${API_BASE}/annotations/${encodeURIComponent(filename)}`, {
          headers: this.getAuthHeaders()
        });
        const data = await res.json();
        if (data.success) {
          // Load and normalize annotations - fix old data format
          const annotations = (data.annotations || []).map(ann => {
            // Normalize segments to ensure start/end are numbers
            const normalizedSegments = (ann.segments || []).map(seg => ({
              start: parseInt(seg.start) || parseInt(seg[0]) || 0,
              end: parseInt(seg.end) || parseInt(seg[1]) || 0,
              count: parseInt(seg.count) || parseInt(seg[2]) || 0,
              minVal: parseFloat(seg.minVal),
              maxVal: parseFloat(seg.maxVal),
              mean: parseFloat(seg.mean),
              label: seg.label || ann.label
            })).filter(seg => !isNaN(seg.start) && !isNaN(seg.end));  // Filter out invalid segments
            
            // Map field names: expert_output -> expertOutput
            return {
              ...ann,
              segments: normalizedSegments,
              expertOutput: ann.expert_output || ann.expertOutput || '',
              prompt: ann.prompt || ''
            };
          });
          
          this.savedAnnotations = annotations;
          console.log('Loaded and normalized annotations:', this.savedAnnotations);

          if (this.savedAnnotations.length > 0) {
            this.showToast(`已加载 ${this.savedAnnotations.length} 个标注`, 'success');
          }
        } else if (res.status === 401) {
          this.$router.push('/login');
        } else {
          this.savedAnnotations = [];
        }
      } catch (e) {
        console.error('Failed to load annotations:', e);
        this.savedAnnotations = [];
      }
    },
    
    // Save annotations to server
    async saveAnnotationsToServer() {
      if (!this.selectedFileName) return;
      
      try {
        // Use unified format (same as export)
        const exportData = {
          filename: this.selectedFileName,
          overall_attribute: this.selectedOverallLabels,
          annotations: this.savedAnnotations.map(ann => ({
            label: {
              id: ann.label.id,
              text: ann.label.text,
              categoryId: ann.label.categoryId,
              color: ann.label.color
            },
            segments: ann.segments,
            prompt: ann.prompt,
            expert_output: ann.expertOutput
          })),
          export_time: new Date().toISOString()
        };
        
        const res = await fetch(`${API_BASE}/annotations/${encodeURIComponent(this.selectedFileName)}`, {
          method: 'POST',
          headers: {
            ...this.getAuthHeaders(),
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(exportData)
        });
        const data = await res.json();
        if (data.success) {
          this.showToast('已自动保存', 'success');
          // Refresh file list to update annotation badge
          await this.loadFiles();
        } else if (res.status === 401) {
          this.$router.push('/login');
        } else {
          console.error('Save failed:', data.error);
          this.showToast('保存失败: ' + data.error, 'error');
        }
      } catch (e) {
        console.error('Save annotations error:', e);
        this.showToast('保存失败: ' + e.message, 'error');
      }
    },
    
    downloadAnnotations() {
      const exportData = {
        filename: this.selectedFileName,
        overall_attribute: this.selectedOverallLabels,
        annotations: this.savedAnnotations.map(ann => ({
          label: {
            id: ann.label.id,
            text: ann.label.text,
            categoryId: ann.label.categoryId,
            color: ann.label.color
          },
          segments: ann.segments,
          prompt: ann.prompt,
          expert_output: ann.expertOutput
        })),
        export_time: new Date().toISOString()
      };
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${this.selectedFileName}.json`;
      a.click();
      URL.revokeObjectURL(url);
    },
    
    clearAllLabels() {
      // Clear all point labels from chart data - does NOT clear savedAnnotations!
      if (plottingApp && plottingApp.allData) {
        plottingApp.allData.forEach(d => d.label = '');
      }
      // Refresh chart display
      if (plottingApp && plottingApp.main) {
        plottingApp.main.selectAll('.point').attr('style', 'fill: black; stroke: none; opacity: 1;');
      }
      if (plottingApp && plottingApp.context) {
        plottingApp.context.selectAll('.point').attr('style', 'fill: black; stroke: none; opacity: 1;');
      }
      // Also clear current selection state (linked behavior)
      this.currentAnnotation = {
        label: null,
        segments: [],
        prompt: '',
        expertOutput: ''
      };
      this.selectedLocalLabels = [];
      this.editingAnnotationIndex = null;
      if (plottingApp) {
        plottingApp.selectedLabel = '';
        plottingApp.labelColor = null;
      }
      // Trigger chartLabelStats reactivity update
      this.chartDataVersion++;
      this.showToast('已清除图上所有标注点', 'success');
    },
    
    // File Upload
    fileCheck(e) {
      const file = e.target.files[0];
      if (!file) return;
      
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const lines = reader.result.split('\n');
          const plotDict = [];
          const seriesSet = new Set();
          
          for (let i = 1; i < lines.length; i++) {
            const cols = lines[i].split(',');
            if (cols.length >= 3) {
              const val = parseFloat(cols[2]);
              if (!isNaN(val)) {
                const series = cols[0] || 'value';
                seriesSet.add(series);
                plotDict.push({
                  id: (i-1).toString(),
                  val,
                  time: DateTime.fromISO(cols[1], { setZone: true }),
                  series,
                  label: cols[3] || ''
                });
              }
            }
          }
          
          if (plotDict.length > 0) {
            this.selectedFileName = file.name;
            this.initChart(plotDict, file.name, Array.from(seriesSet), []);
          } else {
            this.showToast('文件解析失败', 'error');
          }
        } catch (err) {
          this.showToast('文件解析错误', 'error');
        }
      };
      reader.readAsText(file);
    },
    
    // Directory Browser
    async openDirBrowser() {
      this.showDirBrowser = true;
      // Start from user's current path, or /home if not set
      await this.loadDirectory(this.currentPath || '/home');
    },
    
    async loadDirectory(path) {
      try {
        const res = await fetch(`${API_BASE}/browse-dir?path=${encodeURIComponent(path)}`);
        const data = await res.json();
        if (data.success) {
          this.browsePath = data.current_path;
          this.parentPath = data.parent_path || '';
          this.directories = data.directories || [];
        }
      } catch (e) {
        console.error('Failed to load directory:', e);
      }
    },
    
    goToParentDir() {
      if (this.parentPath) this.loadDirectory(this.parentPath);
    },
    
    async selectCurrentDir() {
      try {
        // Set the browsed directory as data path
        this.dataPath = this.browsePath;
        this.showDirBrowser = false;
        
        // Call the setDataPath method to save and refresh
        const res = await fetch(`${API_BASE}/set-path`, {
          method: 'POST',
          headers: this.getAuthHeaders(),
          body: JSON.stringify({ path: this.dataPath })
        });
        const data = await res.json();
        if (data.success) {
          this.currentPath = data.path;
          this.showToast('路径已设置', 'success');
          await this.loadFiles();
        } else {
          this.showToast('路径设置失败: ' + data.error, 'error');
        }
      } catch (e) {
        console.error('Select directory error:', e);
        this.showToast('设置失败', 'error');
      }
    },
    
    // D3 Triggers
    updateHoverinfo() {
      this.hoverinfo = { ...plottingApp.hoverinfo };
    },
    
    resetChartView() {
      // Reset chart view to full extent
      if (plottingApp && typeof plottingApp.resetView === 'function') {
        plottingApp.resetView();
        this.showToast('视图已重置', 'info');
      }
    },
    
    triggerReplot() {
      // Trigger chart replot
    },
    
    triggerRecolor() {
      // Trigger point recolor
    },
    
    clearSeries() {
      if (plottingApp.allData) {
        plottingApp.allData.filter(d => d.series === plottingApp.selectedSeries).forEach(d => d.label = '');
      }
    },
    
    updateSelectionRange() {
      // Called by D3 when brush selection changes - add segment to current annotation
      console.log('=== updateSelectionRange called ===');
      
      if (!plottingApp.selection) {
        console.log('No selection data from plottingApp');
        return;
      }
      
      console.log('plottingApp.selection:', plottingApp.selection);
      
      // Parse and validate numeric values
      const start = parseInt(plottingApp.selection.start);
      const end = parseInt(plottingApp.selection.end);
      const count = parseInt(plottingApp.selection.count);
      const minVal = parseFloat(plottingApp.selection.minVal);
      const maxVal = parseFloat(plottingApp.selection.maxVal);
      const mean = parseFloat(plottingApp.selection.mean);
      const std = parseFloat(plottingApp.selection.std);
      const range = parseFloat(plottingApp.selection.range);
      
      // Validate that we have valid numbers
      if (isNaN(start) || isNaN(end)) {
        console.error('Invalid selection range:', { start, end });
        this.showToast('框选数据错误', 'error');
        return;
      }
      
      // Store selection stats for display - use $set to ensure reactivity
      this.$set(this, 'selectionStats', {
        start,
        end,
        count: isNaN(count) ? 0 : count,
        minVal: isNaN(minVal) ? 0 : minVal,
        maxVal: isNaN(maxVal) ? 0 : maxVal,
        mean: isNaN(mean) ? 0 : mean,
        std: isNaN(std) ? 0 : std,
        range: isNaN(range) ? 0 : range
      });
      
      console.log('selectionStats set to:', this.selectionStats);
      
      // Determine the label to use - current label from UI or from plottingApp
      let labelToUse = this.currentAnnotation.label;
      if (!labelToUse && plottingApp.selectedLabel) {
        console.log('Syncing label from plottingApp:', plottingApp.selectedLabel);
        labelToUse = this.findLabelByText(plottingApp.selectedLabel);
        console.log('Found label:', labelToUse);
      }
      
      if (!labelToUse) {
        console.log('No label selected');
        this.selectionRange = `${start} - ${end} (${count}点) - 请先选择标签`;
        this.showToast('请先选择一个标签', 'warning');
        return;
      }
      
      // Create segment object WITH its label info - each segment remembers which label was active
      const segment = {
        start,
        end,
        count: isNaN(count) ? 0 : count,
        minVal: isNaN(minVal) ? 0 : minVal,
        maxVal: isNaN(maxVal) ? 0 : maxVal,
        mean: isNaN(mean) ? 0 : mean,
        label: { ...labelToUse }  // Store the label with the segment!
      };
      
      // Create new segments array with the new segment using $set for reactivity
      const newSegments = [...this.currentAnnotation.segments, segment];
      
      // Use $set for all nested properties to ensure full reactivity
      // Keep currentAnnotation.label as the last selected label (for UI display)
      this.$set(this, 'currentAnnotation', {
        label: labelToUse,
        segments: newSegments,
        prompt: this.currentAnnotation.prompt || '',
        expertOutput: this.currentAnnotation.expertOutput || ''
      });
      
      // Increment version to trigger computed property updates
      this.annotationVersion++;
      this.chartDataVersion++;
      
      // Auto-switch to show this label's segments in workspace
      if (labelToUse && labelToUse.text) {
        this.activeChartLabel = labelToUse.text;
      }
      
      console.log('NEW segment with label:', JSON.stringify(segment));
      console.log('Total segments:', newSegments.length);
      this.selectionRange = `已添加 ${newSegments.length} 段`;
      this.showToast(`已添加数据段: ${segment.start}-${segment.end} (${labelToUse.text})`, 'success');
      
      // Trigger chartLabelStats reactivity update
      this.chartDataVersion++;
      
      // Force Vue to re-render after D3 hidden button trigger
      this.$nextTick(() => {
        this.$forceUpdate();
        console.log('Vue forceUpdate triggered');
      });
    },
    
    // Helper to find label by text
    findLabelByText(labelText) {
      console.log('findLabelByText called with:', labelText);
      console.log('  - labels.local_change:', this.labels.local_change);
      const localCats = this.labels.local_change || {};
      for (const [catId, cat] of Object.entries(localCats)) {
        console.log('  - Checking category:', catId, cat);
        if (cat.labels) {
          const label = cat.labels.find(l => l.text === labelText);
          if (label) {
            console.log('  - Found label:', label);
            return {
              id: label.id,
              text: label.text,
              color: label.color || this.getCategoryColor(catId),
              categoryId: catId,
              categoryName: cat.name
            };
          }
        }
      }
      console.log('  - Label not found, creating fallback');
      // Fallback: create label from plottingApp if available
      if (plottingApp && plottingApp.labelColor) {
        return {
          id: 'fallback_' + Date.now(),
          text: labelText,
          color: plottingApp.labelColor,
          categoryId: 'unknown',
          categoryName: '自定义'
        };
      }
      return null;
    },
    
    // Utilities
    getSelectedLabelColor() {
      if (!this.selectedLabel) return '#7E4C64';
      const label = this.optionsList.find(l => l.name === this.selectedLabel);
      return label?.color || '#7E4C64';
    },
    
    formatNumber(val) {
      if (val === null || val === undefined) return '-';
      return val.toFixed(4);
    },
    
    showToast(message, type = 'info') {
      this.toast = { show: true, message, type };
      setTimeout(() => { this.toast.show = false; }, 3000);
    },
    
    // Label Management Methods
    addCategory() {
      const newId = 'cat_' + Date.now();
      // Get target object reference directly
      if (!this.labels.overall_attribute) this.$set(this.labels, 'overall_attribute', {});
      if (!this.labels.local_change) this.$set(this.labels, 'local_change', {});
      
      const target = this.labelSettingsTab === 'overall' 
        ? this.labels.overall_attribute 
        : this.labels.local_change;
      
      const newCategory = {
        name: '新分类',
        labels: []
      };
      
      // Add color for local categories
      if (this.labelSettingsTab === 'local') {
        newCategory.color = '#6b7280';
      }
      
      this.$set(target, newId, newCategory);
      this.showToast('分类已添加', 'success');
    },
    
    deleteCategory(catId) {
      if (!confirm(`确定删除分类 "${this.editableCategories[catId]?.name}" 吗？`)) {
        return;
      }
      const target = this.labelSettingsTab === 'overall' 
        ? this.labels.overall_attribute 
        : this.labels.local_change;
      this.$delete(target, catId);
      this.showToast('分类已删除', 'success');
    },
    
    addLabelToCategory(catId) {
      const target = this.labelSettingsTab === 'overall' 
        ? this.labels.overall_attribute 
        : this.labels.local_change;
      const cat = target[catId];
      if (!cat) return;
      
      if (!cat.labels) {
        this.$set(cat, 'labels', []);
      }
      const newLabel = {
        id: 'label_' + Date.now(),
        text: '新标签'
      };
      
      // Add unique color for local labels
      if (this.labelSettingsTab === 'local') {
        newLabel.color = this.generateUniqueColor();
      }
      
      cat.labels.push(newLabel);
      this.showToast('标签已添加', 'success');
    },
    
    deleteLabelFromCategory(catId, idx) {
      const target = this.labelSettingsTab === 'overall' 
        ? this.labels.overall_attribute 
        : this.labels.local_change;
      if (target[catId] && target[catId].labels) {
        target[catId].labels.splice(idx, 1);
        this.showToast('标签已删除', 'success');
      }
    },
    
    // Generate a unique color that's not already used
    generateUniqueColor() {
      const palette = [
        '#ef4444', '#f97316', '#f59e0b', '#eab308', '#84cc16', 
        '#22c55e', '#10b981', '#14b8a6', '#06b6d4', '#0ea5e9',
        '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef',
        '#ec4899', '#f43f5e', '#78716c', '#64748b', '#0f172a'
      ];
      const used = this.usedColors;
      
      // Find first unused color
      for (const color of palette) {
        if (!used.has(color)) {
          return color;
        }
      }
      // If all used, generate random
      return '#' + Math.floor(Math.random()*16777215).toString(16).padStart(6, '0');
    },
    
    async loadData(filename) {
      try {
        this.loading = true;
        const res = await fetch(`${API_BASE}/data/${filename}`, {
          headers: this.getAuthHeaders()
        });
        const data = await res.json();
        
        if (data.success) {
          console.log('Loaded data:', data.data.length, 'rows');
          
          // Pass data to D3
          if (plottingApp) {
            plottingApp.updateData(data.data, data.seriesList);
          }
          
          this.isChartMode = true;
        } else if (res.status === 401) {
          this.$router.push('/login');
        } else {
          this.showToast('加载失败: ' + data.error, 'error');
        }
      } catch (e) {
        console.error('Save labels error:', e);
        this.showToast('保存失败: ' + e.message, 'error');
      }
    },
    
    async saveLabelsToServer() {
      try {
        const res = await fetch(`${API_BASE}/labels`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.labels)
        });
        const data = await res.json();
        if (data.success) {
          this.showToast('标签配置保存成功', 'success');
          this.showLabelSettings = false;
          // Update categoryColors from saved labels
          this.updateCategoryColors();
        } else {
          this.showToast('保存失败: ' + data.error, 'error');
        }
      } catch (e) {
        console.error('Save labels error:', e);
        this.showToast('保存失败: ' + e.message, 'error');
      }
    },
    
    updateCategoryColors() {
      // Sync categoryColors from local_change labels
      const localCats = this.labels.local_change || {};
      Object.keys(localCats).forEach(catId => {
        const cat = localCats[catId];
        if (cat.color) {
          this.$set(this.categoryColors, catId, cat.color);
        }
      });
    }
  }
};
</script>

<style>
/* Global D3 Styles */
svg { font: 10px sans-serif; display: block; margin: auto; overflow: visible; user-select: none; -webkit-user-select: none; }
#maindiv { width: 100%; text-align: left; user-select: none; -webkit-user-select: none; }
.line { fill: none; stroke: black; stroke-width: 1.5px; clip-path: url(#clip); pointer-events: none; }
.point { fill: black; stroke: none; clip-path: url(#clip); }
.axis path, .axis line { fill: none; stroke: #000; shape-rendering: crispEdges; }
.loader { position: fixed; left: 50%; top: 30%; transform: translateX(-50%); border: 8px solid #f3f3f3; border-top: 8px solid #7E4C64; border-radius: 50%; width: 60px; height: 60px; animation: spin 1s linear infinite; }
@keyframes spin { 0% { transform: translateX(-50%) rotate(0deg); } 100% { transform: translateX(-50%) rotate(360deg); } }
kbd { display: inline-block; border: 1px solid #ccc; border-radius: 4px; padding: 0.1em 0.4em; background: #f7f7f7; font-size: 0.75em; }
</style>

<style scoped>
/* App Container */
.app-container { min-height: 100vh; background: #f5f5f5; }

/* Navbar */
.navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 20px;
  background: #fff;
  border-bottom: 1px solid #eee;
  position: sticky;
  top: 0;
  z-index: 100;
}
.navbar-brand { margin: 0; font-size: 1.125rem; font-weight: 600; color: #333; }
.navbar-file { color: #666; font-size: 0.875rem; margin-left: auto; margin-right: 16px; }
.navbar-user { display: flex; align-items: center; gap: 12px; }
.user-name { font-size: 0.875rem; color: #555; }
.btn-logout {
  padding: 6px 16px;
  background: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 0.8125rem;
  cursor: pointer;
  transition: all 0.2s;
}
.btn-logout:hover { background: #e8e8e8; border-color: #ccc; }

.nav-btn { background: white; border: none; color: #7E4C64; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-weight: 500; font-size: 0.875rem; }
.nav-btn:hover { background: #f0f0f0; }

/* Layout */
.main-layout { display: grid; grid-template-columns: 280px 1fr 300px; gap: 16px; padding: 16px; min-height: calc(100vh - 60px); }
.main-layout.no-file { grid-template-columns: 280px 1fr; }

/* Sidebar */
.sidebar { background: white; border-radius: 8px; padding: 10px; overflow-y: auto; max-height: calc(100vh - 80px); box-shadow: 0 1px 3px rgba(0,0,0,0.1); }

/* Panel Cards */
.panel-card { border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px; margin-bottom: 10px; background: #fafafa; }
.panel-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.panel-card-title { font-size: 0.8125rem; font-weight: 600; color: #333; }
.btn-icon-sm { background: none; border: none; cursor: pointer; font-size: 0.875rem; padding: 2px; opacity: 0.7; }
.btn-icon-sm:hover { opacity: 1; }

/* Legacy section styles */
.panel-section { margin-bottom: 12px; }
.section-header { display: flex; justify-content: space-between; align-items: center; }
.section-title { margin: 0 0 8px; font-size: 0.8125rem; font-weight: 600; color: #333; }
.subsection-title { font-size: 0.8125rem; font-weight: 600; color: #666; margin: 12px 0 8px; }

/* Inputs */
.input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; resize: vertical; font-family: inherit; }
textarea:disabled { background-color: #f5f5f5; color: #999; cursor: not-allowed; }
.input:focus, textarea:focus { outline: none; border-color: #7E4C64; }
.path-input-group { display: flex; gap: 6px; }
.path-input-group .input { flex: 1; min-width: 0; }
.current-path { font-size: 0.75rem; color: #888; margin: 4px 0 0; word-break: break-all; }

/* Buttons */
.btn { padding: 8px 16px; border: none; border-radius: 6px; font-size: 0.875rem; cursor: pointer; transition: all 0.2s; }
.btn-primary { background: #7E4C64; color: white; }
.btn-primary:hover { background: #6a3f54; }
.btn-success { background: #22c55e; color: white; }
.btn-sm { padding: 6px 12px; font-size: 0.8125rem; }
.btn-lg { padding: 12px 24px; font-size: 1rem; }
.btn-icon { background: none; border: none; font-size: 1rem; cursor: pointer; padding: 4px; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-delete { background: none; border: none; color: #ef4444; font-size: 1.2rem; cursor: pointer; }
.add-btn, .delete-btn { background: #eee; border: 1px solid #ddd; width: 28px; height: 28px; border-radius: 4px; font-size: 1.2rem; cursor: pointer; }

/* File List */
/* File List - Expanded and bordered */
.file-list { max-height: 300px; overflow-y: auto; margin-top: 8px; }
.file-item { 
  padding: 10px 14px; 
  margin-bottom: 6px;
  border: 1px solid #e0e0e0; 
  border-radius: 6px; 
  cursor: pointer; 
  font-size: 0.875rem;
  transition: all 0.2s;
}
.file-item:hover { background: #f8f8f8; border-color: #c0c0c0; }
.file-item.active { background: #d4edda; border-color: #28a745; font-weight: 500; }

/* Labels */
.label-section { margin-bottom: 8px; }
.label-section summary { font-weight: 600; cursor: pointer; padding: 4px 0; font-size: 0.8125rem; }

/* Label Categories - Optimized for compact layout */
.label-categories { display: flex; flex-direction: column; gap: 1px; }
.label-category { margin: 1px 0; padding-left: 0; }
.category-name { display: block; font-size: 0.75rem; font-weight: 600; color: #666; margin-bottom: 1px; }

/* Overall attributes - more compact horizontal layout */
.label-options { display: flex; flex-wrap: wrap; gap: 4px 8px; padding-left: 6px; }
.label-option { display: inline-flex; align-items: center; gap: 3px; font-size: 0.75rem; cursor: pointer; white-space: nowrap; }
.label-option input[type="radio"] { margin: 0; cursor: pointer; }
.label-option span { cursor: pointer; }

/* Local labels - compact vertical layout */
.local-category { margin: 2px 0; }
.local-label-options { display: flex; flex-direction: column; gap: 2px; padding-left: 8px; }
.local-label-item { display: flex; align-items: center; gap: 4px; padding: 2px 6px; border-radius: 4px; cursor: pointer; font-size: 0.8125rem; border: 1px solid transparent; transition: all 0.2s; }
.local-label-item:hover { background-color: #f5f5f5; }
.local-label-item.active { font-weight: 500; }
.label-color-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.label-color { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
.label-select-group { display: flex; align-items: center; gap: 4px; }
.label-select { flex: 1; min-width: 0; }

/* Main Content */
.main-content { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-height: 400px; display: flex; flex-direction: column; }
.welcome-section { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
.title { font-size: 2rem; color: #333; margin-bottom: 8px; }
.subtitle { color: #888; margin-bottom: 24px; }
.hint { color: #888; font-size: 0.875rem; margin-top: 16px; }
.chart-area { flex: 1; }

/* Toolbar (instructions + actions) */
.toolbar { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; font-size: 0.8125rem; }
.toolbar-row { display: flex; gap: 12px; align-items: center; flex-wrap: nowrap; }
.toolbar-section { padding: 8px 12px; background: #f8f8f8; border-radius: 6px; flex-shrink: 0; }
.toolbar-section.instr { line-height: 1.4; }
.toolbar-section.instr.compact { padding: 6px 10px; }
.toolbar-section.selectors { display: flex; flex-direction: column; gap: 6px; flex-shrink: 0; }
.toolbar-section.selectors select { margin-left: 8px; min-width: 100px; }
.toolbar-section.actions-inline { display: flex; gap: 8px; margin-left: auto; flex-shrink: 0; }
.selector-item { white-space: nowrap; display: flex; align-items: center; gap: 8px; }
.selector-item label { min-width: 60px; font-weight: 500; }

/* File Tabs */
.file-tabs { display: flex; gap: 0; margin-bottom: 8px; border-bottom: 1px solid #eee; }
.file-tab { flex: 1; padding: 8px 4px; background: transparent; border: none; border-bottom: 2px solid transparent; cursor: pointer; font-size: 0.8125rem; color: #666; transition: all 0.2s; }
.file-tab:hover { color: #7E4C64; background: #f8f4f6; }
.file-tab.active { color: #7E4C64; border-bottom-color: #7E4C64; font-weight: 600; }

/* Sort Control */
.sort-control { display: flex; align-items: center; gap: 6px; margin: 8px 0; font-size: 0.75rem; }
.sort-control label { color: #666; font-weight: 500; }
.sort-select { padding: 4px 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 0.75rem; cursor: pointer; }

/* File Badge */
.file-badge { color: #22c55e; font-weight: bold; margin-left: 4px; }

/* Navbar File Name */
.navbar-file { color: rgba(255,255,255,0.8); font-size: 0.875rem; margin-left: auto; }

/* Legacy compatibility */
.instructions { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-top: 16px; font-size: 0.8125rem; }
.instr-col { padding: 12px; background: #f8f8f8; border-radius: 6px; }
.selectors select { margin-left: 8px; }

/* Hover Card */
#hoverbox { position: relative; float: right; z-index: 5; }
.hover-card { position: absolute; right: 20px; top: 10px; background: white; border: 1px solid #ddd; border-radius: 6px; padding: 10px; font-size: 0.8125rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }

/* Annotations */
.annotation-list { max-height: 250px; overflow-y: auto; margin-bottom: 16px; }
.annotation-item { padding: 10px; border: 1px solid #eee; border-radius: 6px; margin-bottom: 8px; background: #fafafa; transition: all 0.2s; }
.annotation-item:hover { border-color: #ddd; background: #f5f5f5; }
.annotation-item.editing { border-color: #7E4C64; background: #f8f4f6; box-shadow: 0 0 0 2px rgba(126,76,100,0.1); }
.annotation-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.annotation-actions { display: flex; gap: 4px; margin-left: auto; }
.annotation-segments { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }
.annotation-text { color: #666; font-size: 0.75rem; line-height: 1.3; }
.segment-summary { font-size: 0.75rem; color: #888; }
.segment-badge { font-size: 0.7rem; color: #666; padding: 2px 6px; border-radius: 4px; background: #e5e7eb; }
.label-tag { font-size: 0.7rem; color: white; padding: 2px 6px; border-radius: 4px; }
.label-count { font-size: 0.65rem; opacity: 0.9; margin-left: 2px; }
.selected-labels { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; min-height: 28px; padding: 6px 8px; border: 1px solid #eee; border-radius: 6px; }
.no-label { color: #aaa; font-size: 0.8125rem; }
.annotation-form { border-top: 1px solid #eee; padding-top: 16px; }
.form-group { margin-bottom: 12px; }
.form-group label { display: block; font-size: 0.75rem; font-weight: 600; color: #888; margin-bottom: 4px; }
.selection-display { padding: 8px; background: #f8f8f8; border-radius: 6px; font-family: monospace; font-size: 0.875rem; }
.form-actions { display: flex; gap: 8px; }
.form-actions .btn { flex: 1; }

/* Segments List */

/* Segment index area - fixed height to prevent layout shift */
.segment-index-area {
  min-height: 120px;
  max-height: 150px;
  overflow-y: auto;
  border: 1px solid #e8e8e8;
  border-radius: 6px;
  background: #fafafa;
}

.empty-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 120px;
  color: #999;
  font-size: 0.875rem;
  text-align: center;
}

.segments-list { display: flex; flex-direction: column; gap: 4px; padding: 8px; background: #f8f8f8; border-radius: 6px; max-height: 150px; overflow-y: auto; }
.segment-item { display: flex; align-items: center; gap: 6px; padding: 4px 8px; background: white; border-radius: 4px; font-size: 0.8125rem; }
.segment-range { font-family: monospace; color: #7E4C64; font-weight: 500; }
.segment-count { color: #888; font-size: 0.75rem; }

/* Selection Stats */
.selection-stats { background: #f8f8f8; border-radius: 6px; padding: 10px; font-size: 0.8125rem; }
.stat-row { display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #eee; }
.stat-row:last-child { border-bottom: none; }
.stat-label { color: #666; }
.stat-value { font-family: monospace; color: #7E4C64; font-weight: 500; }

/* Toast */
.toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); padding: 12px 24px; background: #333; color: white; border-radius: 8px; z-index: 9999; }
.toast.success { background: #22c55e; }
.toast.error { background: #ef4444; }

/* Modal */
.modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000; }
.modal-box { background: white; border-radius: 12px; box-shadow: 0 20px 40px rgba(0,0,0,0.2); max-width: 600px; width: 90%; max-height: 80vh; display: flex; flex-direction: column; }
.modal-sm { max-width: 360px; }
.modal-header { display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; border-bottom: 1px solid #eee; }
.modal-header h3 { margin: 0; font-size: 1.1rem; }
.close-btn { background: none; border: none; font-size: 1.5rem; cursor: pointer; color: #888; }
.modal-body { padding: 16px 20px; overflow-y: auto; flex: 1; }
.modal-footer { display: flex; justify-content: flex-end; gap: 12px; padding: 16px 20px; border-top: 1px solid #eee; }
.browser-toolbar { display: flex; gap: 8px; margin-bottom: 16px; }
.browser-toolbar .input { flex: 1; }
.dir-list { border: 1px solid #eee; border-radius: 6px; min-height: 200px; max-height: 300px; overflow-y: auto; }
.dir-item { display: flex; justify-content: space-between; padding: 10px 14px; cursor: pointer; border-bottom: 1px solid #f0f0f0; }
.dir-item:hover { background: #f8f8f8; }
.dir-item.has-data { background: #f0fff4; }
.data-badge { background: #22c55e; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; }
.empty-message { text-align: center; color: #888; padding: 12px; font-size: 0.875rem; }

/* Label Settings Modal */
.modal-lg { max-width: 700px; }
.label-settings-tabs { display: flex; gap: 0; margin-bottom: 16px; border-bottom: 2px solid #eee; }
.settings-tab { flex: 1; padding: 10px 16px; background: transparent; border: none; border-bottom: 2px solid transparent; cursor: pointer; font-size: 0.875rem; font-weight: 500; color: #666; transition: all 0.2s; margin-bottom: -2px; }
.settings-tab:hover { color: #7E4C64; background: #f8f4f6; }
.settings-tab.active { color: #7E4C64; border-bottom-color: #7E4C64; font-weight: 600; }

.category-editor-list { display: flex; flex-direction: column; gap: 12px; }
.category-editor-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; background: #fafafa; }
.category-editor-header { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }
.category-name-input { flex: 1; font-weight: 600; }
.category-actions { display: flex; gap: 6px; align-items: center; }

.label-editor-list { display: flex; flex-direction: column; gap: 6px; padding-left: 12px; border-left: 3px solid #e0e0e0; }
.label-editor-item { display: flex; gap: 6px; align-items: center; }
.label-name-input { flex: 1; }

.color-picker { width: 32px; height: 32px; padding: 0; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; }
.color-picker-sm { width: 24px; height: 24px; padding: 0; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; }

.btn-icon-danger { background: none; border: none; cursor: pointer; font-size: 1rem; padding: 4px; }
.btn-icon-danger:hover { opacity: 0.7; }
.btn-outline { background: transparent; border: 1px dashed #ccc; color: #666; }
.btn-outline:hover { border-color: #7E4C64; color: #7E4C64; }
.btn-xs { padding: 4px 8px; font-size: 0.75rem; }
.input-xs { padding: 4px 8px; font-size: 0.8125rem; }

.add-category-btn { margin-top: 8px; }

/* Floating Selection Stats Panel */
.selection-stats-panel {
  position: absolute;
  right: 20px;
  top: 50px;
  z-index: 10;
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 10px 14px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  min-width: 200px;
}
.stats-title {
  font-size: 0.8125rem;
  font-weight: 600;
  color: #7E4C64;
  margin-bottom: 8px;
  border-bottom: 1px solid #eee;
  padding-bottom: 4px;
}
.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px 12px;
  font-size: 0.75rem;
}
.stats-item {
  color: #555;
}
.stats-item b {
  color: #888;
  font-weight: 500;
}

/* Clickable elements */
.clickable {
  cursor: pointer;
  transition: all 0.15s ease;
}
.clickable:hover {
  opacity: 0.85;
  transform: translateX(2px);
}
.segment-badge.clickable:hover {
  background: #d1d5db;
}
.label-tag.clickable:hover {
  filter: brightness(1.1);
}

/* Improved annotation list - larger area */
.annotation-list { 
  max-height: 400px; 
  overflow-y: auto; 
  margin-bottom: 8px; 
}

/* Fixed Stats Section in Sidebar */
.stats-section {
  background: #f0f8ff;
  border: 1px solid #e0e8f0;
  border-radius: 8px;
  padding: 10px;
  margin-bottom: 12px;
}
.stats-grid-fixed {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.8125rem;
}
.stat-row-inline {
  display: flex;
  justify-content: space-between;
  padding: 3px 0;
  border-bottom: 1px solid #e0e8f0;
}
.stat-row-inline:last-child {
  border-bottom: none;
}

/* Interactive Label Tags (with remove button) */
.label-tag-interactive {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  font-size: 0.7rem;
  color: white;
  padding: 2px 4px 2px 8px;
  border-radius: 4px;
  margin: 2px;
}
.tag-content {
  cursor: pointer;
}
.tag-content:hover {
  opacity: 0.9;
}
.tag-remove {
  background: rgba(255,255,255,0.3);
  border: none;
  color: white;
  font-size: 0.85rem;
  cursor: pointer;
  padding: 0 4px;
  margin-left: 2px;
  border-radius: 2px;
  line-height: 1;
}
.tag-remove:hover {
  background: rgba(255,255,255,0.5);
}

/* Selection Stats Box (clean grid layout) */
.selection-stats-box {
  background: #f8f4f6;
  border: 1px solid #e8dce3;
  border-radius: 6px;
  padding: 8px 14px;
  margin-left: auto;
  font-size: 0.75rem;
}
.selection-stats-box .stats-header {
  font-weight: 600;
  color: #7E4C64;
  margin-bottom: 6px;
  font-size: 0.8rem;
}
.selection-stats-box .stats-grid {
  display: grid;
  grid-template-columns: auto 1fr auto 1fr;
  gap: 4px 12px;
  align-items: center;
}
.selection-stats-box .stat-label {
  color: #888;
  text-align: right;
}
.selection-stats-box .stat-value {
  color: #7E4C64;
  font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
  font-weight: 500;
  white-space: nowrap;
}

/* Chart Labels Container in Workspace */
.chart-labels-container {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.chart-label-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  border-radius: 4px;
  color: white;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s ease;
  border: 2px solid transparent;
}

.chart-label-tag:hover {
  opacity: 0.9;
  transform: translateY(-1px);
}

.chart-label-tag.active {
  border-color: #333;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.chart-label-tag .label-text {
  font-weight: 500;
}

.chart-label-tag .label-count {
  opacity: 0.8;
  font-size: 0.75rem;
}

.chart-label-tag .label-remove {
  background: rgba(255,255,255,0.3);
  border: none;
  color: white;
  font-size: 0.9rem;
  cursor: pointer;
  padding: 0 3px;
  margin-left: 2px;
  border-radius: 2px;
  line-height: 1;
}

.chart-label-tag .label-remove:hover {
  background: rgba(255,255,255,0.5);
}

/* Workspace Section */
.workspace-section {
  border: 1px solid #e8dce3;
  background: #fdfbfc;
}
</style>