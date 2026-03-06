<template>
  <div class="task-center-page">
    <PageHeader title="工作台" subtitle="统一管理所有采集、推理、训练任务" />

    <!-- 统计卡片 -->
    <el-row :gutter="12" class="summary-row">
      <el-col :xs="12" :sm="6" :md="5">
        <StatCard label="总运行" :value="runTotal" icon="📊" color="primary" />
      </el-col>
      <el-col :xs="12" :sm="6" :md="5">
        <StatCard label="运行中" :value="runningCount" icon="🔄" color="warning" />
      </el-col>
      <el-col :xs="12" :sm="6" :md="5">
        <StatCard label="已完成" :value="completedCount" icon="✅" color="success" />
      </el-col>
      <el-col :xs="12" :sm="6" :md="5">
        <StatCard label="失败/超时" :value="failedCount" icon="❌" color="danger" />
      </el-col>
      <el-col :xs="12" :sm="6" :md="4">
        <StatCard label="任务定义" :value="definitions.length" icon="📋" color="primary" />
      </el-col>
    </el-row>

    <!-- 主面板: 4 Tab -->
    <el-card shadow="never" class="panel-card">
      <template #header>
        <div class="panel-header">
          <el-tabs v-model="activeTab" class="header-tabs" @tab-change="onTabChange">
            <el-tab-pane label="📊 概览" name="overview" />
            <el-tab-pane label="🔄 运行记录" name="runs" />
            <el-tab-pane label="📋 任务定义" name="definitions" />
            <el-tab-pane label="⚡ 事件触发" name="events" />
          </el-tabs>
          <el-button :icon="Refresh" @click="loadDashboard">刷新</el-button>
        </div>
      </template>

      <!-- ====== 概览 Tab ====== -->
      <div v-show="activeTab === 'overview'" class="overview-tab">
        <!-- 按类型分布 -->
        <el-row :gutter="16">
          <el-col :xs="24" :md="12">
            <el-card shadow="never">
              <template #header><span>按类型分布</span></template>
              <div class="type-dist">
                <div v-for="item in typeDistribution" :key="item.type" class="dist-item">
                  <div class="dist-label">{{ item.type }}</div>
                  <el-progress
                    :percentage="runTotal > 0 ? Math.round(item.count / runTotal * 100) : 0"
                    :stroke-width="18"
                    :color="item.color"
                    :format="() => `${item.count}`"
                  />
                </div>
                <div v-if="typeDistribution.length === 0" class="empty-text">暂无运行数据</div>
              </div>
            </el-card>
          </el-col>
          <el-col :xs="24" :md="12">
            <el-card shadow="never">
              <template #header><span>按状态分布</span></template>
              <div class="type-dist">
                <div v-for="item in statusDistribution" :key="item.status" class="dist-item">
                  <div class="dist-label">
                    <el-tag :type="statusTagType(item.status)" size="small">{{ item.status }}</el-tag>
                  </div>
                  <el-progress
                    :percentage="runTotal > 0 ? Math.round(item.count / runTotal * 100) : 0"
                    :stroke-width="18"
                    :color="item.color"
                    :format="() => `${item.count}`"
                  />
                </div>
                <div v-if="statusDistribution.length === 0" class="empty-text">暂无运行数据</div>
              </div>
            </el-card>
          </el-col>
        </el-row>

        <!-- 最近活动 -->
        <el-card shadow="never" style="margin-top: 12px">
          <template #header><span>最近活动</span></template>
          <el-timeline>
            <el-timeline-item
              v-for="run in recentRuns"
              :key="run.run_id"
              :timestamp="formatTime(run.created_at)"
              :type="timelineType(run.status)"
              placement="top"
            >
              <span class="activity-text">
                <el-tag :type="statusTagType(run.status)" size="small">{{ run.status }}</el-tag>
                {{ run.task_type }} · {{ (run.run_id || '').substring(0, 8) }}...
                <el-button type="primary" link size="small" @click="openRunDetail(run.run_id)">详情</el-button>
              </span>
            </el-timeline-item>
            <el-timeline-item v-if="recentRuns.length === 0" timestamp="">暂无活动</el-timeline-item>
          </el-timeline>
        </el-card>
      </div>

      <!-- ====== 运行记录 Tab ====== -->
      <div v-show="activeTab === 'runs'">
        <el-form :inline="true" class="filter-form">
          <el-form-item label="状态">
            <el-select v-model="filters.status" placeholder="全部" clearable style="width: 140px">
              <el-option v-for="item in statusOptions" :key="item" :label="item" :value="item" />
            </el-select>
          </el-form-item>
          <el-form-item label="类型">
            <el-select v-model="filters.task_type" placeholder="全部" clearable style="width: 140px">
              <el-option label="inference" value="inference" />
              <el-option label="training" value="training" />
              <el-option label="acquire" value="acquire" />
            </el-select>
          </el-form-item>
          <el-form-item label="触发">
            <el-select v-model="filters.trigger_mode" placeholder="全部" clearable style="width: 120px">
              <el-option v-for="item in triggerModeOptions" :key="item" :label="item" :value="item" />
            </el-select>
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="onSearch">查询</el-button>
            <el-button @click="onReset">重置</el-button>
          </el-form-item>
        </el-form>

        <el-table :data="runRows" v-loading="loadingRuns" border stripe>
          <el-table-column prop="run_id" label="Run ID" width="150" show-overflow-tooltip>
            <template #default="{ row }">{{ (row.run_id || '').substring(0, 10) }}...</template>
          </el-table-column>
          <el-table-column prop="task_type" label="类型" width="110" />
          <el-table-column prop="trigger_mode" label="触发" width="90" />
          <el-table-column label="状态" width="110">
            <template #default="{ row }">
              <el-tag :type="statusTagType(row.status)" size="small">{{ row.status }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="step_count" label="步骤" width="70" />
          <el-table-column label="创建时间" width="170">
            <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
          </el-table-column>
          <el-table-column label="结束时间" width="170">
            <template #default="{ row }">{{ formatTime(row.completed_at) }}</template>
          </el-table-column>
          <el-table-column label="操作" width="220" fixed="right">
            <template #default="{ row }">
              <el-button type="primary" link @click="openRunDetail(row.run_id)">详情</el-button>
              <el-button type="success" link @click="executeRun(row.run_id, false)">执行</el-button>
              <el-button v-if="row.status === 'failed' || row.status === 'timeout'" type="warning" link @click="retryRun(row.run_id)">重试</el-button>
              <el-button v-if="row.status === 'running' || row.status === 'pending'" type="danger" link @click="cancelRun(row.run_id)">取消</el-button>
            </template>
          </el-table-column>
        </el-table>

        <div class="pager-wrap">
          <el-pagination layout="total, prev, pager, next" :current-page="page" :page-size="pageSize" :total="runTotal" @current-change="onPageChange" />
        </div>
      </div>

      <!-- ====== 任务定义 Tab ====== -->
      <div v-show="activeTab === 'definitions'">
        <el-table :data="definitions" v-loading="loadingDefinitions" border stripe>
          <el-table-column prop="name" label="名称" min-width="200" show-overflow-tooltip />
          <el-table-column prop="task_type" label="类型" width="110">
            <template #default="{ row }"><el-tag size="small">{{ row.task_type }}</el-tag></template>
          </el-table-column>
          <el-table-column prop="trigger_mode" label="触发" width="90" />
          <el-table-column label="配置摘要" min-width="200" show-overflow-tooltip>
            <template #default="{ row }">{{ configSummary(row.config) }}</template>
          </el-table-column>
          <el-table-column label="创建时间" width="170">
            <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
          </el-table-column>
          <el-table-column label="操作" width="120" fixed="right">
            <template #default="{ row }">
              <el-button type="success" link @click="executeDefinition(row)">▶ 执行</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <!-- ====== 事件触发 Tab ====== -->
      <div v-show="activeTab === 'events'">
        <el-row :gutter="12" style="margin-bottom: 12px">
          <el-col :span="8">
            <el-statistic title="死信总数" :value="deadLetterTotal" />
          </el-col>
          <el-col :span="16" style="display: flex; align-items: flex-end; justify-content: flex-end">
            <el-button @click="loadDeadLetters">刷新死信</el-button>
          </el-col>
        </el-row>

        <el-table :data="deadLetters" v-loading="loadingDeadLetters" border stripe>
          <el-table-column prop="id" label="Event ID" width="140" show-overflow-tooltip>
            <template #default="{ row }">{{ (row.id || '').substring(0, 10) }}...</template>
          </el-table-column>
          <el-table-column prop="source" label="来源" width="120" />
          <el-table-column prop="event_key" label="事件键" width="150" />
          <el-table-column prop="error" label="错误" min-width="200" show-overflow-tooltip />
          <el-table-column label="时间" width="170">
            <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
          </el-table-column>
          <el-table-column label="操作" width="170">
            <template #default="{ row }">
              <el-button type="primary" link @click="replayDeadLetter(row.id, 'dispatch')">重放执行</el-button>
              <el-button type="info" link @click="replayDeadLetter(row.id, 'none')">仅重建</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-card>

    <!-- 运行详情 Drawer -->
    <el-drawer v-model="detailVisible" title="运行详情" size="50%">
      <el-skeleton :rows="4" animated v-if="loadingDetail" />
      <template v-else>
        <el-descriptions :column="2" border>
          <el-descriptions-item label="Run ID">{{ runDetail?.run_id || '-' }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="statusTagType(runDetail?.status || '')">{{ runDetail?.status || '-' }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="类型">{{ runDetail?.task_type || '-' }}</el-descriptions-item>
          <el-descriptions-item label="触发">{{ runDetail?.trigger_mode || '-' }}</el-descriptions-item>
          <el-descriptions-item label="创建">{{ formatTime(runDetail?.created_at || null) }}</el-descriptions-item>
          <el-descriptions-item label="结束">{{ formatTime(runDetail?.completed_at || null) }}</el-descriptions-item>
          <el-descriptions-item label="错误" :span="2">{{ runDetail?.error || '-' }}</el-descriptions-item>
        </el-descriptions>
        <el-divider content-position="left">步骤状态</el-divider>
        <el-table :data="runDetail?.steps || []" border stripe>
          <el-table-column prop="step_name" label="Step" width="150" />
          <el-table-column label="状态" width="110">
            <template #default="{ row }"><el-tag :type="statusTagType(row.status)" size="small">{{ row.status }}</el-tag></template>
          </el-table-column>
          <el-table-column label="依赖" min-width="160">
            <template #default="{ row }">{{ Array.isArray(row.depends_on) && row.depends_on.length ? row.depends_on.join(', ') : '-' }}</template>
          </el-table-column>
          <el-table-column prop="message" label="消息" min-width="200" show-overflow-tooltip />
          <el-table-column label="开始" width="170"><template #default="{ row }">{{ formatTime(row.started_at) }}</template></el-table-column>
          <el-table-column label="结束" width="170"><template #default="{ row }">{{ formatTime(row.completed_at) }}</template></el-table-column>
        </el-table>
      </template>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { formatTime, statusTagType } from '../utils/format'
import PageHeader from '../components/PageHeader.vue'
import StatCard from '../components/StatCard.vue'
import {
  cancelTaskCenterRun,
  createTaskCenterRun,
  executeTaskCenterRun,
  fetchTaskCenterDefinitions,
  fetchTaskCenterDeadLetters,
  fetchTaskCenterEventMetrics,
  fetchTaskCenterRuns,
  fetchTaskCenterRunStatus,
  replayTaskCenterDeadLetter,
  retryTaskCenterRun,
  type TaskCenterDeadLetter,
  type TaskCenterDefinitionRow,
  type TaskCenterRunRow,
  type TaskCenterRunStatusData,
} from '../api/taskCenter'

const activeTab = ref('overview')
const loadingRuns = ref(false)
const loadingDetail = ref(false)
const loadingDeadLetters = ref(false)
const loadingDefinitions = ref(false)
const page = ref(1)
const pageSize = 20
const runTotal = ref(0)
const runRows = ref<TaskCenterRunRow[]>([])
const allRuns = ref<TaskCenterRunRow[]>([])
const deadLetters = ref<TaskCenterDeadLetter[]>([])
const definitions = ref<TaskCenterDefinitionRow[]>([])
const detailVisible = ref(false)
const runDetail = ref<TaskCenterRunStatusData | null>(null)
const deadLetterTotal = ref(0)

const filters = reactive({ status: '', task_type: '', trigger_mode: '' })
const statusOptions = ['pending', 'running', 'completed', 'failed', 'cancelled', 'timeout']
const triggerModeOptions = ['manual', 'auto', 'schedule', 'event']

// Computed stats
const runningCount = computed(() => allRuns.value.filter(r => r.status === 'running' || r.status === 'pending').length)
const completedCount = computed(() => allRuns.value.filter(r => r.status === 'completed').length)
const failedCount = computed(() => allRuns.value.filter(r => r.status === 'failed' || r.status === 'timeout').length)

const recentRuns = computed(() => allRuns.value.slice(0, 10))

const typeDistribution = computed(() => {
  const map: Record<string, number> = {}
  allRuns.value.forEach(r => { map[r.task_type] = (map[r.task_type] || 0) + 1 })
  const colors: Record<string, string> = { inference: '#409EFF', training: '#E6A23C', acquire: '#67C23A' }
  return Object.entries(map).map(([type, count]) => ({ type, count, color: colors[type] || '#909399' }))
})

const statusDistribution = computed(() => {
  const map: Record<string, number> = {}
  allRuns.value.forEach(r => { map[r.status] = (map[r.status] || 0) + 1 })
  const colors: Record<string, string> = { completed: '#67C23A', running: '#E6A23C', pending: '#E6A23C', failed: '#F56C6C', timeout: '#F56C6C', cancelled: '#909399' }
  return Object.entries(map).map(([status, count]) => ({ status, count, color: colors[status] || '#909399' }))
})

function timelineType(status: string): 'primary' | 'success' | 'warning' | 'danger' | 'info' {
  if (status === 'completed') return 'success'
  if (status === 'running' || status === 'pending') return 'warning'
  if (status === 'failed' || status === 'timeout') return 'danger'
  return 'info'
}

function configSummary(config: Record<string, unknown> | null): string {
  if (!config) return '-'
  const parts: string[] = []
  if (config.algorithm) parts.push(`算法:${config.algorithm}`)
  if (config.config_name) parts.push(`配置:${config.config_name}`)
  if (config.model_family) parts.push(`模型族:${config.model_family}`)
  return parts.join(' | ') || JSON.stringify(config).substring(0, 50)
}

// ==================== Data Loading ====================

async function loadRuns(): Promise<void> {
  loadingRuns.value = true
  try {
    const params: Record<string, unknown> = { limit: pageSize, offset: (page.value - 1) * pageSize }
    if (filters.status) params.status = filters.status
    if (filters.task_type) params.task_type = filters.task_type
    if (filters.trigger_mode) params.trigger_mode = filters.trigger_mode
    const res = await fetchTaskCenterRuns(params)
    const data = (res && typeof res === 'object' ? (res as { data?: Record<string, unknown> }).data : {}) || {}
    runTotal.value = Number(data.total || 0)
    runRows.value = Array.isArray(data.runs) ? (data.runs as TaskCenterRunRow[]) : []
  } finally {
    loadingRuns.value = false
  }
}

async function loadAllRuns(): Promise<void> {
  const res = await fetchTaskCenterRuns({ limit: 200 })
  const data = (res && typeof res === 'object' ? (res as { data?: Record<string, unknown> }).data : {}) || {}
  allRuns.value = Array.isArray(data.runs) ? (data.runs as TaskCenterRunRow[]) : []
  runTotal.value = Number(data.total || allRuns.value.length)
}

async function loadDefinitions(): Promise<void> {
  loadingDefinitions.value = true
  try {
    const res = await fetchTaskCenterDefinitions()
    const data = (res && typeof res === 'object' ? (res as { data?: Record<string, unknown> }).data : {}) || {}
    definitions.value = Array.isArray(data.definitions) ? (data.definitions as TaskCenterDefinitionRow[]) : []
  } finally {
    loadingDefinitions.value = false
  }
}

async function loadDeadLetters(): Promise<void> {
  loadingDeadLetters.value = true
  try {
    const res = await fetchTaskCenterDeadLetters({ limit: 50, offset: 0 })
    const data = (res && typeof res === 'object' ? (res as { data?: Record<string, unknown> }).data : {}) || {}
    deadLetters.value = Array.isArray(data.items) ? (data.items as TaskCenterDeadLetter[]) : []
  } finally {
    loadingDeadLetters.value = false
  }
}

async function loadMetrics(): Promise<void> {
  try {
    const res = await fetchTaskCenterEventMetrics()
    const data = (res && typeof res === 'object' ? (res as { data?: Record<string, unknown> }).data : {}) || {}
    deadLetterTotal.value = Number(data.dead_letter_total || 0)
  } catch { deadLetterTotal.value = 0 }
}

async function loadDashboard(): Promise<void> {
  await Promise.all([loadAllRuns(), loadRuns(), loadDefinitions(), loadMetrics()])
}

function onTabChange(): void {
  if (activeTab.value === 'events' && deadLetters.value.length === 0) loadDeadLetters()
}

// ==================== Actions ====================

async function openRunDetail(runId: string): Promise<void> {
  loadingDetail.value = true
  detailVisible.value = true
  try {
    const res = await fetchTaskCenterRunStatus(runId)
    const data = (res && typeof res === 'object' ? (res as { data?: Record<string, unknown> }).data : {}) || {}
    runDetail.value = data as unknown as TaskCenterRunStatusData
  } finally {
    loadingDetail.value = false
  }
}

async function executeRun(runId: string, simulate: boolean): Promise<void> {
  await executeTaskCenterRun(runId, simulate)
  ElMessage.success(simulate ? '模拟执行已触发' : '执行已触发')
  await loadDashboard()
}

async function executeDefinition(row: TaskCenterDefinitionRow): Promise<void> {
  const resp = await createTaskCenterRun({
    definition_id: row.id, task_type: row.task_type,
    trigger_mode: 'manual', input_payload: row.config, auto_execute: true,
  })
  const data = (resp && typeof resp === 'object' ? (resp as { data?: Record<string, unknown> }).data : {}) || {}
  const runId = String(data.run_id || '')
  if (runId) ElMessage.success(`任务已执行: ${runId.substring(0, 8)}...`)
  await loadDashboard()
}

async function retryRun(runId: string): Promise<void> {
  await retryTaskCenterRun(runId)
  ElMessage.success('重试任务已创建')
  await loadDashboard()
}

async function cancelRun(runId: string): Promise<void> {
  await cancelTaskCenterRun(runId)
  ElMessage.success('取消请求已提交')
  await loadDashboard()
}

async function replayDeadLetter(eventId: string, mode: 'dispatch' | 'simulate' | 'none'): Promise<void> {
  await replayTaskCenterDeadLetter(eventId, mode)
  ElMessage.success('死信重放已提交')
  await loadDashboard()
}

function onSearch(): void { page.value = 1; loadRuns() }
function onReset(): void { filters.status = ''; filters.task_type = ''; filters.trigger_mode = ''; page.value = 1; loadRuns() }
function onPageChange(p: number): void { page.value = p; loadRuns() }

onMounted(() => { loadDashboard() })
</script>

<style scoped>
.task-center-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.summary-row {
  margin-bottom: 4px;
}

.panel-card {
  min-height: 400px;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.header-tabs {
  flex: 1;
}

.header-tabs :deep(.el-tabs__header) {
  margin-bottom: 0;
}

.header-tabs :deep(.el-tabs__nav-wrap::after) {
  display: none;
}

.filter-form {
  margin-bottom: 12px;
}

.pager-wrap {
  margin-top: 12px;
  display: flex;
  justify-content: flex-end;
}

.overview-tab {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.type-dist {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.dist-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.dist-label {
  width: 100px;
  flex-shrink: 0;
  font-size: 13px;
}

.dist-item .el-progress {
  flex: 1;
}

.empty-text {
  color: var(--el-text-color-secondary);
  text-align: center;
  padding: 20px;
}

.activity-text {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
</style>
