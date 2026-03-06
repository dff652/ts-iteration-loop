<template>
  <div class="annotation-workbench">
    <div class="top-bar">
      <div class="top-bar-left">
        <el-button size="small" @click="$router.push('/annotation/review')" text>← 返回平台</el-button>
        <span class="page-title">📊 标注工作台</span>
        <span v-if="currentFile" class="top-filename">{{ currentFile }}</span>
      </div>
      <span v-if="statusMsg" class="status-msg" :class="statusType">{{ statusMsg }}</span>
      <div class="top-actions">
        <el-button size="small" @click="refreshFiles">刷新</el-button>
      </div>
    </div>

    <div class="workbench-layout">
      <!-- ═══════ Left Sidebar ═══════ -->
      <aside class="left-sidebar">
        <!-- 数据目录 -->
        <el-collapse v-model="leftPanels">
          <el-collapse-item name="files" title="📁 数据目录">
            <el-tabs v-model="activeFileTab" class="file-tabs" stretch>
              <!-- 数据文件 Tab -->
              <el-tab-pane label="数据文件" name="raw">
                <el-input v-model="searchRaw" placeholder="搜索文件..." size="small" clearable class="file-search" />
                <div class="file-list tab-file-list">
                  <div
                    v-for="file in filteredRawFiles"
                    :key="file.filename"
                    class="file-item"
                    :class="{ active: file.filename === currentFile }"
                    @click="loadRawFile(file)"
                  >
                    <span class="file-name">{{ file.filename }}</span>
                    <span v-if="file.size_bytes" class="file-size">{{ formatSize(file.size_bytes) }}</span>
                  </div>
                  <el-empty v-if="filteredRawFiles.length === 0" description="暂无文件" :image-size="40" />
                </div>
              </el-tab-pane>

              <!-- 已标注数据 Tab -->
              <el-tab-pane label="标注数据" name="assets">
                <el-input v-model="searchAssets" placeholder="搜索标注数据..." size="small" clearable class="file-search" />
                <div class="dataset-list tab-file-list">
                  <div
                    v-for="file in filteredAssets"
                    :key="file.filename"
                    class="file-item"
                    :class="{ active: file.filename === currentFile }"
                    @click="loadRawFile(file)"
                  >
                    <span class="file-name">{{ file.filename || file.name }}</span>
                    <el-tag v-if="file.annotation_count" size="small" type="success" effect="plain" class="count-tag">{{ file.annotation_count }} 条</el-tag>
                  </div>
                  <el-empty v-if="filteredAssets.length === 0" description="暂无标注数据" :image-size="40" />
                </div>
              </el-tab-pane>
            </el-tabs>
          </el-collapse-item>

          <!-- 索引数据段 -->
          <el-collapse-item v-if="inferenceSegments.length > 0" name="segments" title="📊 索引数据段">
            <div class="segment-filter">
              <el-select v-model="segmentFilterRule" size="small" style="width: 60px">
                <el-option value="gt" label=">" />
                <el-option value="gte" label="≥" />
                <el-option value="lt" label="<" />
                <el-option value="lte" label="≤" />
              </el-select>
              <el-input-number v-model="segmentScoreThreshold" :min="0" :max="1" :step="0.01" size="small" style="width: 90px" placeholder="阈值" />
              <span class="seg-filter-count">{{ filteredSegments.length }}/{{ inferenceSegments.length }}</span>
            </div>
            <div class="segment-nav-list">
              <div
                v-for="(seg, idx) in filteredSegments"
                :key="idx"
                class="segment-nav-item"
                @click="navigateToInferenceSegment(seg)"
              >
                <span class="segment-range">{{ seg.start }} - {{ seg.end }}</span>
                <span class="segment-score">{{ seg.score?.toFixed(2) }}</span>
              </div>
            </div>
          </el-collapse-item>

          <!-- 标签列表 -->
          <el-collapse-item name="labels">
            <template #title>
              <div class="collapse-title-with-btn">
                <span>🏷️ 标签列表</span>
                <el-button size="small" text class="settings-btn" @click.stop="showLabelDialog = true">
                  ⚙️ 设置
                </el-button>
              </div>
            </template>
            
            <el-collapse v-model="labelPanels" class="inner-collapse">
              <!-- 整体属性 -->
              <el-collapse-item name="overall" title="整体属性">
                <div v-for="(cat, catId) in overallCategories" :key="catId" class="label-category">
                  <span class="cat-name">{{ cat.name }}</span>
                  <el-radio-group v-model="selectedOverall[catId]" size="small">
                    <el-radio v-for="label in cat.labels" :key="label.id" :value="label.id">{{ label.text }}</el-radio>
                  </el-radio-group>
                </div>
              </el-collapse-item>

              <!-- 局部变化 -->
              <el-collapse-item name="local" title="局部变化">
                <div v-for="(cat, catId) in localCategories" :key="catId" class="label-category">
                  <span class="cat-name" :style="{ color: cat.color || '#666' }">■ {{ cat.name }}</span>
                  <div class="local-labels">
                    <span
                      v-for="label in cat.labels"
                      :key="label.id"
                      class="local-label-chip"
                      :class="{ active: activeLocalLabel?.id === label.id }"
                      :style="activeLocalLabel?.id === label.id ? { borderColor: label.color || cat.color, backgroundColor: (label.color || cat.color) + '22' } : {}"
                      @click="selectLocalLabel(label, catId)"
                    >
                      <span class="label-dot" :style="{ backgroundColor: label.color || cat.color }"></span>
                      {{ label.text }}
                    </span>
                  </div>
                </div>
                <el-empty v-if="Object.keys(localCategories).length === 0" description="暂无标签" :image-size="30" />
              </el-collapse-item>
            </el-collapse>

          </el-collapse-item>
        </el-collapse>
      </aside>

      <!-- ═══════ Main Content ═══════ -->
      <main class="main-content">
        <!-- Toolbar -->
        <div v-if="chartReady" class="chart-toolbar">
          <div class="toolbar-left">
            <span class="instr"><strong>标注:</strong> 拖拽框选 | 点击toggle | <kbd>Shift</kbd>+拖拽取消</span>
            <span class="instr"><strong>导航:</strong> <kbd>←</kbd><kbd>→</kbd>平移 | <kbd>↑</kbd><kbd>↓</kbd>缩放</span>
          </div>
          <div class="toolbar-right">
            <el-select v-if="seriesList.length > 1" v-model="selectedSeries" size="small" placeholder="主序列" style="width: 160px" @change="onSeriesChange">
              <el-option v-for="s in seriesList" :key="s" :label="s" :value="s" />
            </el-select>
            <el-select v-if="seriesList.length > 1" v-model="selectedRef" size="small" placeholder="参考序列" style="width: 160px" @change="onRefChange">
              <el-option v-for="s in seriesList" :key="s" :label="s" :value="s" />
            </el-select>
            <el-button size="small" @click="d3Ref?.resetView()">🔄 重置视图</el-button>
            <el-button size="small" type="warning" @click="d3Ref?.clearAllLabels()">清除标注</el-button>
          </div>
        </div>

        <!-- Selection Stats -->
        <div v-if="selectionStats" class="selection-stats">
          <div class="stats-row">
            <span>📊 <strong>框选范围</strong>: {{ selectionStats.start }} - {{ selectionStats.end }}</span>
            <span><strong>点数</strong>: {{ selectionStats.count }}</span>
          </div>
          <div class="stats-row" v-if="selectionStats.minVal !== undefined">
            <span><strong>范围</strong>: {{ (selectionStats.minVal ?? 0).toFixed(3) }} ~ {{ (selectionStats.maxVal ?? 0).toFixed(3) }}</span>
            <span><strong>均值</strong>: {{ (selectionStats.mean ?? 0).toFixed(3) }}</span>
            <span><strong>标准差</strong>: {{ (selectionStats.std ?? 0).toFixed(3) }}</span>
            <span v-if="selectionStats.score !== undefined"><strong>分数</strong>: {{ selectionStats.score.toFixed(2) }}</span>
          </div>
        </div>

        <!-- D3 Chart -->
        <D3Wrapper
          v-if="chartData.length > 0"
          ref="d3Ref"
          :csv-data="chartData"
          :filename="currentFile"
          :header-str="headerStr"
          :series-list="seriesList"
          :label-list="chartLabelList"
          :selected-label="activeLocalLabel?.text || ''"
          @chart-selection="onChartSelection"
          @hover-update="onHoverUpdate"
          @edit-axis="onEditAxis"
          @label-change="onLabelChange"
        />

        <el-empty v-if="chartData.length === 0" description="请在左侧选择文件" class="empty-chart" />
      </main>

      <!-- ═══════ Right Sidebar ═══════ -->
      <aside v-if="chartReady" class="right-sidebar">
        <!-- 标注工作区 -->
        <div class="panel-section">
          <div class="panel-header">
            <span>📝 标注工作区</span>
            <div v-if="workspaceData">
              <el-button v-if="workspaceMode === 'view'" size="small" @click="workspaceMode = 'edit'">✏️ 编辑</el-button>
              <el-button size="small" @click="clearWorkspace">退出</el-button>
            </div>
          </div>

          <template v-if="!workspaceData">
            <div class="empty-panel">
              <p>点击下方标注结果加载</p>
              <p class="hint">或在图上框选新建标注</p>
            </div>
          </template>

          <template v-else>
            <!-- Label -->
            <div class="form-group">
              <label>标签类型 <span v-if="workspaceMode === 'edit'" class="edit-badge">编辑中</span></label>
              <el-select v-if="workspaceMode === 'edit'" v-model="workspaceData.labelId" size="small" @change="onWorkspaceLabelChange">
                <el-option v-for="l in flatLabels" :key="l.id" :label="l.text" :value="l.id" />
              </el-select>
              <div v-else class="label-display">
                <span class="label-tag" :style="{ backgroundColor: workspaceLabelColor }">{{ workspaceLabelText }}</span>
              </div>
            </div>

            <!-- Segments -->
            <div class="form-group">
              <label>数据段 ({{ workspaceData.segments.length }})</label>
              <div class="segments-list">
                <div
                  v-for="(seg, idx) in workspaceData.segments"
                  :key="idx"
                  class="segment-item"
                  :style="{ borderLeft: '3px solid ' + workspaceLabelColor }"
                >
                  <span class="segment-range" @click="navigateToSegment(seg)">{{ seg.start }} - {{ seg.end }}</span>
                  <span class="segment-count">({{ seg.end - seg.start + 1 }}点)</span>
                  <el-button v-if="workspaceMode === 'edit'" size="small" text type="danger" @click="removeSegment(idx)">×</el-button>
                </div>
              </div>
            </div>

            <!-- Prompt / Expert -->
            <div v-if="workspaceMode === 'edit'" class="form-group">
              <label>问题</label>
              <el-input v-model="workspaceData.prompt" type="textarea" :rows="2" placeholder="描述发现的问题..." />
            </div>
            <div v-if="workspaceMode === 'edit'" class="form-group">
              <label>评价</label>
              <el-input v-model="workspaceData.expertOutput" type="textarea" :rows="2" placeholder="评价..." />
            </div>

            <!-- Save workspace -->
            <div v-if="workspaceMode === 'edit'" class="ws-actions">
              <el-button type="primary" size="small" @click="saveWorkspaceToAnnotations">确认保存</el-button>
            </div>
          </template>
        </div>

        <!-- 标注结果 -->
        <div class="panel-section">
          <div class="panel-header">
            <span>📋 标注结果 ({{ annotations.length }})</span>
            <div>
              <el-button size="small" :disabled="undoStack.length === 0" @click="undoAction" title="撤回">↶</el-button>
              <el-button size="small" :disabled="redoStack.length === 0" @click="redoAction" title="重做">↷</el-button>
              <el-button size="small" type="primary" @click="saveAnnotations" :disabled="annotations.length === 0">💾 保存</el-button>
              <el-button size="small" @click="downloadAnnotations" :disabled="annotations.length === 0">📥 导出</el-button>
            </div>
          </div>
          <div class="annotation-list">
            <div
              v-for="(ann, idx) in annotations"
              :key="idx"
              class="annotation-card"
              :class="{ active: workspaceIndex === idx }"
              @click="loadToWorkspace(idx)"
            >
              <div class="ann-header">
                <span class="label-tag" :style="{ backgroundColor: getLabelColor(ann.labelId) }">{{ getLabelText(ann.labelId) }}</span>
                <span class="seg-count">({{ ann.segments.length }}段)</span>
                <el-button size="small" text type="danger" @click.stop="deleteAnnotation(idx)">×</el-button>
              </div>
              <div class="ann-segments">
                <span v-for="(seg, si) in ann.segments.slice(0, 5)" :key="si" class="seg-badge">
                  {{ seg.start }}-{{ seg.end }}
                </span>
                <span v-if="ann.segments.length > 5" class="seg-more">+{{ ann.segments.length - 5 }}</span>
              </div>
              <div v-if="ann.prompt" class="ann-prompt-preview">
                <small>Q: {{ ann.prompt.substring(0, 50) }}{{ ann.prompt.length > 50 ? '...' : '' }}</small>
              </div>
            </div>
            <el-empty v-if="annotations.length === 0" description="暂无标注" :image-size="30" />
          </div>
        </div>
      </aside>
    </div>

    <!-- Label Settings Dialog -->
    <LabelSettingsDialog
      v-model="showLabelDialog"
      :initial-labels="{ overall_attribute: overallCategories, local_change: localCategories } as any"
      @saved="onLabelsSaved"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import D3Wrapper from '../components/D3Wrapper.vue'
import LabelSettingsDialog, { type LabelsConfig } from '../components/LabelSettingsDialog.vue'
import {
  listAnnotatorFiles,
  getAnnotatorData,
  getAnnotations,
  saveAnnotations as saveAnnotationsApi,
  getLabels
} from '../api/annotation'
import { listRawDatasets, type RawDatasetFile } from '../api/data'

// ─── State ────────────────────────────────────────────────
const leftPanels = ref(['files', 'labels', 'segments'])

// File management / Data Catalog
const dataPath = ref('/home/share/data/downsampled') // default fallback
const currentFile = ref('')
const headerStr = ref('')

const activeFileTab = ref('raw')

// Tab 1: Raw Datasets
const rawFiles = ref<RawDatasetFile[]>([])
const searchRaw = ref('')
const filteredRawFiles = computed(() => {
  if (!searchRaw.value) return rawFiles.value
  const kw = searchRaw.value.toLowerCase()
  return rawFiles.value.filter((f: RawDatasetFile) => (f.filename || f.name).toLowerCase().includes(kw))
})

// Tab 2: Annotated Files
const annotatedFiles = ref<RawDatasetFile[]>([])
const searchAssets = ref('')

const filteredAssets = computed(() => {
  if (!searchAssets.value) return annotatedFiles.value
  const kw = searchAssets.value.toLowerCase()
  return annotatedFiles.value.filter((d: RawDatasetFile) => (d.filename || d.name).toLowerCase().includes(kw))
})

// Chart data
const chartData = ref<any[]>([])
const seriesList = ref<string[]>([])
const chartReady = computed(() => chartData.value.length > 0)

// D3 ref
const d3Ref = ref<InstanceType<typeof D3Wrapper>>()

// Series selection
const selectedSeries = ref('')
const selectedRef = ref('')

// Selection stats
const selectionStats = ref<any>(null)

// Labels
interface LabelItem { id: string; text: string; color: string; categoryId?: string }
interface Category { name: string; labels: LabelItem[]; color?: string }
const overallCategories = ref<Record<string, Category>>({})
const localCategories = ref<Record<string, Category>>({})
const selectedOverall = ref<Record<string, string>>({})
const activeLocalLabel = ref<LabelItem | null>(null)
const showLabelDialog = ref(false)
const labelPanels = ref(['overall', 'local']) // Default expanded panels

// Chart label list (for D3)
const chartLabelList = computed(() => {
  const labels: { name: string; color: string }[] = []
  Object.values(localCategories.value).forEach(cat => {
    cat.labels.forEach(l => {
      labels.push({ name: l.text, color: l.color || cat.color || '#7E4C64' })
    })
  })
  if (labels.length === 0) {
    labels.push({ name: 'label_1', color: '#ef4444' })
  }
  return labels
})

const flatLabels = computed(() => {
  const result: LabelItem[] = []
  Object.entries(localCategories.value).forEach(([catId, cat]) => {
    cat.labels.forEach(l => {
      result.push({ ...l, categoryId: catId })
    })
  })
  return result
})

// Annotations
interface Segment { start: number; end: number }
interface AnnotationRecord {
  labelId: string
  segments: Segment[]
  prompt: string
  expertOutput: string
}

const annotations = ref<AnnotationRecord[]>([])

// Workspace
const workspaceData = ref<{ labelId: string; segments: Segment[]; prompt: string; expertOutput: string } | null>(null)
const workspaceMode = ref<'view' | 'edit'>('view')
const workspaceIndex = ref<number | null>(null)

// Status
const statusMsg = ref('')
const statusType = ref<'success' | 'error' | 'info'>('info')

function showStatus(msg: string, type: 'success' | 'error' | 'info' = 'info') {
  statusMsg.value = msg
  statusType.value = type
  setTimeout(() => { statusMsg.value = '' }, 3000)
}

// ─── File Management & Data Catalog ─────────────────────────────

async function refreshFiles() {
  try {
    // Fetch both lists in parallel; use safe fallbacks
    const rawRes = await listRawDatasets().catch(() => null)
    const assetsRes = await listAnnotatorFiles(dataPath.value).catch(() => null)
    
    // Parse raw datasets — interceptor already unwraps response.data
    if (rawRes && typeof rawRes === 'object') {
      // After interceptor: rawRes = { success, data: { datasets: [...] }, message }
      const inner = (rawRes as any).data || rawRes
      const datasets = inner.datasets || inner
      rawFiles.value = Array.isArray(datasets) ? datasets : []
    } else {
      rawFiles.value = []
    }
    
    // Parse annotated files
    if (assetsRes && typeof assetsRes === 'object') {
      const inner = (assetsRes as any).data || assetsRes
      const files = inner.files || inner
      if (Array.isArray(files)) {
        annotatedFiles.value = files.filter((f: any) => f.has_annotations)
      } else {
        annotatedFiles.value = []
      }
    } else {
      annotatedFiles.value = []
    }
    
    showStatus('目录刷新成功', 'success')
  } catch (e: any) {
    showStatus('目录加载失败: ' + e.message, 'error')
  }
}

async function loadFileCore(filename: string, dirPath?: string) {
  try {
    currentFile.value = filename
    // Clear old state before setting new state
    annotations.value = []
    clearWorkspace()

    const path = dirPath || dataPath.value || '/home/share/data/downsampled'
    const resp: any = await getAnnotatorData(filename, path)

    // The backend now returns { idx, val, label } points and a single seriesName.
    // LabelerD3 expects { idx, val, label, series } mapped to it.
    const sName = resp.seriesName || 'value'
    chartData.value = (resp.data || []).map((d: any) => ({
      ...d,
      series: sName,
    }))
    seriesList.value = [sName]
    headerStr.value = resp.header || ''

    selectedSeries.value = seriesList.value[0] || ''
    selectedRef.value = seriesList.value[0] || ''

    // Load existing annotations
    await loadAnnotations()
    // Load labels
    await loadLabels()

    showStatus(`已加载 ${currentFile.value}`, 'success')
  } catch (e: any) {
    showStatus('文件加载失败: ' + e.message, 'error')
  }
}

async function loadRawFile(file: RawDatasetFile) {
  const dirPath = file.path ? file.path.replace(/\\/g, '/').split('/').slice(0, -1).join('/') : ''
  dataPath.value = dirPath // Cache path for annotations saving logic
  await loadFileCore(file.filename || file.name, dirPath)
}


async function loadAnnotations() {
  try {
    const resp: any = await getAnnotations(currentFile.value, dataPath.value)
    if (resp && Array.isArray(resp.annotations)) {
      annotations.value = resp.annotations.map((a: any) => ({
        labelId: a.label?.id || a.labelId || '',
        segments: a.segments || [],
        prompt: a.prompt || '',
        expertOutput: a.expertOutput || a.expert_output || '',
      }))
    }
  } catch {
    // no annotations yet
  }
}

async function loadLabels() {
  try {
    const resp: any = await getLabels()
    const data = resp.labels
    overallCategories.value = data?.overall_attribute || {}
    localCategories.value = data?.local_change || {}
  } catch {
    // use defaults
  }
}

// ─── Series ──────────────────────────────────────────────
function onSeriesChange(val: string) {
  d3Ref.value?.changeSeries(val)
}

function onRefChange(val: string) {
  d3Ref.value?.changeReference(val)
}

// ─── Labels ──────────────────────────────────────────────
function selectLocalLabel(label: LabelItem, catId: string) {
  // If workspace has unsaved changes for a DIFFERENT label, auto-commit first
  if (workspaceData.value && workspaceData.value.segments.length > 0) {
    const currentLabelId = workspaceData.value.labelId
    if (currentLabelId !== label.id) {
      // Auto-save the current workspace to annotations list
      saveWorkspaceToAnnotations()
      // Clear workspace for the new label
      workspaceData.value = null
      workspaceMode.value = 'edit'
      workspaceIndex.value = null
    }
  }

  activeLocalLabel.value = { ...label, categoryId: catId }
  d3Ref.value?.changeLabel(label.text)
  d3Ref.value?.setLabelColor(label.color || '#7E4C64')

  // If there's an existing annotation for this label, load it to workspace
  const existingIdx = annotations.value.findIndex(a => a.labelId === label.id)
  if (existingIdx >= 0) {
    loadToWorkspace(existingIdx)
    workspaceMode.value = 'edit'
  } else {
    // No existing annotation; prepare empty workspace for this label
    if (!workspaceData.value || workspaceData.value.labelId !== label.id) {
      workspaceData.value = {
        labelId: label.id,
        segments: [],
        prompt: '',
        expertOutput: '',
      }
      workspaceMode.value = 'edit'
      workspaceIndex.value = null
    }
  }
}

function onLabelsSaved(newLabels: LabelsConfig) {
  overallCategories.value = (newLabels.overall_attribute as any) || {}
  localCategories.value = (newLabels.local_change as any) || {}
  
  if (activeLocalLabel.value) {
    const newColor = getLabelColor(activeLocalLabel.value.id)
    if (newColor) {
      d3Ref.value?.setLabelColor(newColor)
      activeLocalLabel.value.color = newColor
    }
  }
}

function getLabelColor(labelId: string): string {
  for (const cat of Object.values(localCategories.value)) {
    const found = cat.labels.find(l => l.id === labelId)
    if (found) return found.color || cat.color || '#7E4C64'
  }
  return '#7E4C64'
}

function getLabelText(labelId: string): string {
  for (const cat of Object.values(localCategories.value)) {
    const found = cat.labels.find(l => l.id === labelId)
    if (found) return found.text
  }
  return labelId
}

const workspaceLabelColor = computed(() => workspaceData.value ? getLabelColor(workspaceData.value.labelId) : '#7E4C64')
const workspaceLabelText = computed(() => workspaceData.value ? getLabelText(workspaceData.value.labelId) : '')

// ─── Chart Events ────────────────────────────────────────
function onChartSelection(start: number, end: number) {
  const baseStats = d3Ref.value?.getSelection() || { start, end, count: end - start + 1 }
  selectionStats.value = calculateSelectionStats(start, end, baseStats)

  // Auto-create workspace entry if label is selected
  if (activeLocalLabel.value) {
    if (!workspaceData.value) {
      workspaceData.value = {
        labelId: activeLocalLabel.value.id,
        segments: [{ start, end }],
        prompt: '',
        expertOutput: '',
      }
      workspaceMode.value = 'edit'
      workspaceIndex.value = null
    } else if (workspaceMode.value === 'edit') {
      workspaceData.value.segments.push({ start, end })
    }
  }
}

function onHoverUpdate(_info: { time: string; val: string; label: string }) {
  // Could display in a tooltip if needed
}

function onEditAxis() {
  // Could open axis edit dialog
  ElMessage.info('Y轴编辑功能 (待实现)')
}

function onLabelChange(_label: string) {
  // D3 changed label via keyboard shortcut
}

// ─── Workspace ───────────────────────────────────────────
function loadToWorkspace(idx: number) {
  const ann = annotations.value[idx]
  if (!ann) return
  workspaceData.value = JSON.parse(JSON.stringify(ann))
  workspaceMode.value = 'view'
  workspaceIndex.value = idx

  // Navigate to first segment
  if (ann.segments.length > 0 && ann.segments[0]) {
    navigateToSegment(ann.segments[0])
  }
}

function clearWorkspace() {
  workspaceData.value = null
  workspaceMode.value = 'view'
  workspaceIndex.value = null
  selectionStats.value = null
  d3Ref.value?.clearSelection()
}

function applyAnnotationsToChart() {
  const data = chartData.value
  if (!data || data.length === 0) return

  // 1. Clear existing labels on all points
  for (let i = 0; i < data.length; i++) {
    data[i].label = ''
  }

  // Helper to apply labels to segments
  const applySegs = (segments: Segment[], text: string) => {
    for (const seg of segments) {
      let started = false
      for (let i = 0; i < data.length; i++) {
        const idx = data[i].idx
        if (idx >= seg.start && idx <= seg.end) {
          data[i].label = text
          started = true
        } else if (started && idx > seg.end) {
          break // early exit since data is sorted by idx
        }
      }
    }
  }

  // 2. Apply existing annotations (least priority)
  annotations.value.forEach((ann, idx) => {
    if (workspaceMode.value === 'edit' && workspaceIndex.value === idx) return
    applySegs(ann.segments, getLabelText(ann.labelId))
  })

  // 3. Apply workspace Data (highest priority)
  if (workspaceData.value) {
    applySegs(workspaceData.value.segments, getLabelText(workspaceData.value.labelId))
  }

  // 4. Update D3
  d3Ref.value?.triggerRecolor()
}

watch(
  [() => chartData.value.length, annotations, workspaceData, workspaceMode, workspaceIndex],
  () => {
    if (chartData.value.length > 0) {
      applyAnnotationsToChart()
    }
  },
  { deep: true, immediate: true }
)

function calculateSelectionStats(start: number, end: number, baseStats: any) {
  const result = { ...baseStats, start, end, count: end - start + 1 }
  const data = chartData.value
  if (!data || data.length === 0) return result

  // Find boundaries
  let startIndex = 0
  let endIndex = data.length - 1
  for (let i = 0; i < data.length; i++) {
    if (data[i].idx >= start) {
      startIndex = i
      break
    }
  }
  for (let i = data.length - 1; i >= 0; i--) {
    if (data[i].idx <= end) {
      endIndex = i
      break
    }
  }

  if (startIndex <= endIndex) {
    const slice = data.slice(startIndex, endIndex + 1)
    const vals = slice.map((d: any) => d.val).filter((v: number) => typeof v === 'number' && !isNaN(v))
    if (vals.length > 0) {
      result.minVal = Math.min(...vals)
      result.maxVal = Math.max(...vals)
      
      const sum = vals.reduce((a: number, b: number) => a + b, 0)
      result.mean = sum / vals.length
      
      const squareDiffs = vals.map((v: number) => {
        const diff = v - result.mean
        return diff * diff
      })
      const avgSquareDiff = squareDiffs.reduce((a: number, b: number) => a + b, 0) / vals.length
      result.std = Math.sqrt(avgSquareDiff)

      // Calculate score based on Qwen metric mapping if matching
      if (slice[0] && typeof slice[0].score !== 'undefined') {
        const scores = slice.map((d: any) => d.score).filter((v: number) => typeof v === 'number' && !isNaN(v))
        if(scores.length > 0) result.score = Math.max(...scores)
      } else {
        // Fallback or generic score interpretation
        result.score = 0
      }
    }
  }
  return result
}
function onWorkspaceLabelChange() {
  // Label changed in workspace
}

function removeSegment(idx: number) {
  workspaceData.value?.segments.splice(idx, 1)
}

function navigateToSegment(seg: Segment) {
  d3Ref.value?.setPreventBrushSearch(true)
  d3Ref.value?.setSelection(seg.start, seg.end)
}

function saveWorkspaceToAnnotations() {
  if (!workspaceData.value || workspaceData.value.segments.length === 0) {
    ElMessage.warning('请至少添加一个数据段')
    return
  }

  const record: AnnotationRecord = JSON.parse(JSON.stringify(workspaceData.value))

  pushHistory()
  if (workspaceIndex.value !== null) {
    annotations.value[workspaceIndex.value] = record
  } else {
    annotations.value.push(record)
    workspaceIndex.value = annotations.value.length - 1
  }

  workspaceMode.value = 'view'
  showStatus('标注已保存到列表', 'success')
}

function deleteAnnotation(idx: number) {
  pushHistory()
  annotations.value.splice(idx, 1)
  if (workspaceIndex.value === idx) {
    clearWorkspace()
  }
}

// ─── Save / Export ───────────────────────────────────────
async function saveAnnotations() {
  try {
    const payload = {
      filename: currentFile.value,
      path: dataPath.value,
      annotations: annotations.value.map(a => ({
        label: { id: a.labelId, text: getLabelText(a.labelId), color: getLabelColor(a.labelId) },
        segments: a.segments,
        prompt: a.prompt,
        expertOutput: a.expertOutput,
      })),
      overall_attribute: selectedOverall.value,
    }
    await saveAnnotationsApi(currentFile.value, payload)
    showStatus('保存成功', 'success')
    ElMessage.success('标注已保存到服务器')
    // Refresh to ensure new dataset entries show up in assets list
    await refreshFiles()
  } catch (e: any) {
    showStatus('保存失败: ' + e.message, 'error')
    ElMessage.error('保存失败')
  }
}

function downloadAnnotations() {
  const data = {
    filename: currentFile.value,
    annotations: annotations.value,
    overall_attributes: selectedOverall.value,
  }
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = currentFile.value.replace(/\.\w+$/, '') + '_annotations.json'
  a.click()
  URL.revokeObjectURL(url)
}

// ─── Inference Segments (A6) ─────────────────────────────
interface InferenceSegment { start: number; end: number; count?: number; score?: number }
const inferenceSegments = ref<InferenceSegment[]>([])
const segmentFilterRule = ref('gt')
const segmentScoreThreshold = ref<number>(0.7)

const filteredSegments = computed(() => {
  const threshold = segmentScoreThreshold.value
  if (threshold === null || threshold === undefined) return inferenceSegments.value
  return inferenceSegments.value.filter(seg => {
    const score = seg.score ?? 0
    switch (segmentFilterRule.value) {
      case 'gt': return score > threshold
      case 'gte': return score >= threshold
      case 'lt': return score < threshold
      case 'lte': return score <= threshold
      default: return true
    }
  })
})

function navigateToInferenceSegment(seg: InferenceSegment) {
  d3Ref.value?.setSelection(seg.start, seg.end)
  selectionStats.value = { start: seg.start, end: seg.end, count: seg.count || (seg.end - seg.start + 1), score: seg.score }
}

// ─── Undo / Redo (F3) ───────────────────────────────────
const undoStack = ref<string[]>([])
const redoStack = ref<string[]>([])
const HISTORY_LIMIT = 20

function pushHistory() {
  undoStack.value.push(JSON.stringify(annotations.value))
  if (undoStack.value.length > HISTORY_LIMIT) undoStack.value.shift()
  redoStack.value = []
}

function undoAction() {
  if (undoStack.value.length === 0) return
  redoStack.value.push(JSON.stringify(annotations.value))
  const prev = undoStack.value.pop()!
  annotations.value = JSON.parse(prev)
  d3Ref.value?.triggerRecolor()
}

function redoAction() {
  if (redoStack.value.length === 0) return
  undoStack.value.push(JSON.stringify(annotations.value))
  const next = redoStack.value.pop()!
  annotations.value = JSON.parse(next)
  d3Ref.value?.triggerRecolor()
}

// ─── Helpers ─────────────────────────────────────────────

function formatSize(bytes: number): string {
  if (bytes < 1024) return bytes + 'B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + 'K'
  return (bytes / (1024 * 1024)).toFixed(1) + 'M'
}

// ─── Lifecycle ───────────────────────────────────────────
onMounted(() => {
  refreshFiles()
})
</script>

<style scoped>
.annotation-workbench {
  height: calc(100vh - 10px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.top-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 16px;
  border-bottom: 1px solid var(--el-border-color-lighter);
  background: var(--el-bg-color);
  flex-shrink: 0;
}

.top-bar-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.page-title {
  font-weight: 600;
  font-size: 16px;
}

.status-msg {
  font-size: 13px;
  padding: 2px 8px;
  border-radius: 4px;
}
.status-msg.success { color: #67c23a; }
.status-msg.error { color: #f56c6c; }
.status-msg.info { color: #909399; }

.top-actions {
  margin-left: auto;
  display: flex;
  gap: 6px;
}

.workbench-layout {
  flex: 1;
  display: flex;
  overflow: hidden;
}

/* ─── Left Sidebar ─── */
.left-sidebar {
  width: 280px;
  min-width: 220px;
  border-right: 1px solid var(--el-border-color-lighter);
  overflow-y: auto;
  padding: 8px;
}

.path-row { margin-bottom: 8px; }

.file-list {
  max-height: 300px;
  overflow-y: auto;
}

.file-item {
  display: flex;
  justify-content: space-between;
  padding: 6px 8px;
  cursor: pointer;
  border-radius: 4px;
  font-size: 13px;
  transition: background 0.15s;
}

.file-item:hover { background: var(--el-fill-color-light); }
.file-item.active { background: var(--el-color-primary-light-9); color: var(--el-color-primary); font-weight: 500; }

.file-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.file-size { color: #999; font-size: 12px; margin-left: 8px; }

.label-section { margin-top: 8px; }
.section-label { font-weight: 600; font-size: 13px; margin-bottom: 4px; }

.collapse-title-with-btn {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.settings-btn {
  margin-right: 8px;
  font-size: 12px;
  padding: 4px 8px;
}

.inner-collapse {
  border-top: none;
  border-bottom: none;
  margin-left: -4px;
}

.inner-collapse :deep(.el-collapse-item__header) {
  height: 32px;
  line-height: 32px;
  font-size: 13px;
  font-weight: 500;
  background-color: transparent;
  color: #555;
  border-bottom: none;
  padding-left: 8px;
}

.inner-collapse :deep(.el-collapse-item__wrap) {
  border-bottom: none;
  background-color: transparent;
}

.inner-collapse :deep(.el-collapse-item__content) {
  padding-bottom: 8px;
}

.label-category { margin-bottom: 8px; padding-left: 8px; }
.cat-name { font-size: 13px; font-weight: 500; display: block; margin-bottom: 4px; }

.local-labels { display: flex; flex-wrap: wrap; gap: 4px; }

.local-label-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border: 1px solid #ddd;
  border-radius: 12px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}
.local-label-chip:hover { border-color: #aaa; }
.local-label-chip.active { font-weight: 600; }

.label-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

/* ─── Main Content ─── */
.main-content {
  flex: 1;
  overflow: auto;
  padding: 8px;
  min-width: 0;
}

.chart-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 6px 8px;
  background: var(--el-fill-color-lighter);
  border-radius: 4px;
  margin-bottom: 8px;
}

.toolbar-left { display: flex; gap: 16px; }
.toolbar-right { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }

.instr {
  font-size: 12px;
  color: #666;
}

kbd {
  display: inline-block;
  border: 1px solid #ccc;
  border-radius: 3px;
  padding: 0 4px;
  margin: 0 2px;
  background: #f7f7f7;
  font-size: 11px;
  line-height: 1.4;
}

.selection-stats {
  display: flex;
  gap: 16px;
  padding: 4px 8px;
  background: #fef3c7;
  border-radius: 4px;
  font-size: 13px;
  margin-bottom: 8px;
}

.empty-chart {
  margin-top: 100px;
}

/* ─── Right Sidebar ─── */
.right-sidebar {
  width: 300px;
  min-width: 240px;
  border-left: 1px solid var(--el-border-color-lighter);
  overflow-y: auto;
  padding: 8px;
}

.panel-section {
  margin-bottom: 12px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 8px;
}

.empty-panel {
  text-align: center;
  padding: 20px;
  color: #999;
  font-size: 13px;
}

.empty-panel .hint { font-size: 12px; color: #bbb; }

.form-group { margin-bottom: 10px; }
.form-group label { display: block; font-size: 13px; font-weight: 500; margin-bottom: 4px; }

.edit-badge {
  font-size: 11px;
  padding: 1px 6px;
  background: #fef3c7;
  border-radius: 4px;
  color: #d97706;
}

.label-display { margin-top: 4px; }

.label-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  color: #fff;
  font-size: 12px;
  font-weight: 500;
}

.segments-list {
  max-height: 200px;
  overflow-y: auto;
}

.segment-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  margin-bottom: 4px;
  background: var(--el-fill-color-lighter);
  border-radius: 4px;
  font-size: 13px;
}

.segment-range {
  cursor: pointer;
  color: var(--el-color-primary);
}
.segment-range:hover { text-decoration: underline; }

.segment-count { color: #999; font-size: 12px; }

.ws-actions {
  margin-top: 8px;
  text-align: right;
}

/* Annotation list */
.annotation-list {
  max-height: 400px;
  overflow-y: auto;
}

.annotation-card {
  padding: 8px;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;
  margin-bottom: 6px;
  cursor: pointer;
  transition: border-color 0.15s;
}

.annotation-card:hover { border-color: var(--el-color-primary-light-5); }
.annotation-card.active { border-color: var(--el-color-primary); background: var(--el-color-primary-light-9); }

.ann-header {
  display: flex;
  align-items: center;
  gap: 6px;
}

.seg-count { color: #999; font-size: 12px; }

.ann-segments {
  margin-top: 4px;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.seg-badge {
  font-size: 11px;
  padding: 1px 6px;
  background: var(--el-fill-color);
  border-radius: 4px;
}

.seg-more { font-size: 11px; color: #999; }

.top-filename {
  font-size: 13px;
  color: var(--el-text-color-secondary);
  font-family: monospace;
  background: var(--el-fill-color-light);
  padding: 2px 8px;
  border-radius: 4px;
}

/* Inference Segments */
.segment-filter {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 8px;
}
.seg-filter-count { font-size: 11px; color: #999; }
.segment-nav-list { max-height: 200px; overflow-y: auto; }
.segment-nav-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 8px;
  cursor: pointer;
  font-size: 12px;
  border-radius: 4px;
}
.segment-nav-item:hover { background: var(--el-fill-color); }
.segment-range { color: var(--el-text-color-regular); }
.segment-score { color: var(--el-color-warning); font-weight: 600; }

/* Annotation prompt preview */
.ann-prompt-preview {
  font-size: 11px;
  color: #999;
  margin-top: 2px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* File & Dataset Tabs */
.file-tabs :deep(.el-tabs__header) {
  margin-bottom: 8px;
}
.file-search {
  margin-bottom: 8px;
}
.tab-file-list {
  max-height: 400px;
  overflow-y: auto;
}
.dataset-group {
  margin-bottom: 4px;
}
.dataset-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  cursor: pointer;
  border-radius: 4px;
  background-color: var(--el-fill-color-light);
  font-size: 13px;
  transition: background-color 0.2s;
}
.dataset-header:hover {
  background-color: var(--el-fill-color);
}
.ds-icon {
  display: flex;
  align-items: center;
  color: var(--el-text-color-secondary);
}
.ds-name {
  flex: 1;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.dataset-items {
  padding-left: 14px;
  margin-top: 2px;
  border-left: 1px dashed var(--el-border-color);
  margin-left: 12px;
}
.sub-item {
  padding: 4px 8px !important;
  font-size: 12px !important;
  border-bottom: none !important;
}
.ds-loading, .ds-empty {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  padding: 4px 8px;
}
</style>
