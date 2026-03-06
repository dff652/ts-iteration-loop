<template>
  <div class="task-def-panel">
    <!-- 看板 -->
    <el-row :gutter="12" class="stats-row">
      <el-col :xs="12" :sm="6">
        <StatCard label="任务定义" :value="definitions.length" icon="📋" color="primary" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <StatCard label="运行中" :value="runStats.running" icon="🔄" color="warning" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <StatCard label="已完成" :value="runStats.completed" icon="✅" color="success" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <StatCard label="失败" :value="runStats.failed" icon="❌" color="danger" />
      </el-col>
    </el-row>

    <!-- 主卡片: Tab 切换 -->
    <el-card shadow="never">
      <template #header>
        <div class="section-header">
          <el-tabs v-model="activeTab" class="header-tabs" @tab-change="onTabChange">
            <el-tab-pane label="运行记录" name="runs" />
            <el-tab-pane label="任务定义" name="definitions" />
          </el-tabs>
          <div class="header-actions">
            <el-button type="primary" @click="$emit('create')">+ 新建</el-button>
            <el-button @click="refreshAll">刷新</el-button>
          </div>
        </div>
      </template>

      <!-- 运行记录 Tab -->
      <div v-show="activeTab === 'runs'">
        <el-table :data="runs" v-loading="loadingRuns" border stripe>
          <el-table-column prop="run_id" label="Run ID" width="140" show-overflow-tooltip>
            <template #default="{ row }">{{ (row.run_id || '').substring(0, 8) }}...</template>
          </el-table-column>
          <el-table-column label="状态" width="100">
            <template #default="{ row }">
              <el-tag :type="statusTagType(row.status)" size="small">{{ row.status }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="来源" width="100">
            <template #default="{ row }">{{ row.trigger_mode }}</template>
          </el-table-column>
          <el-table-column label="创建时间" width="170">
            <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
          </el-table-column>
          <el-table-column label="结束时间" width="170">
            <template #default="{ row }">{{ formatTime(row.completed_at) }}</template>
          </el-table-column>
          <el-table-column label="操作" width="200" fixed="right">
            <template #default="{ row }">
              <el-button type="primary" link @click="openLogDrawer(row.run_id)">📋 日志</el-button>
              <el-button type="info" link @click="openDetailDrawer(row.run_id)">详情</el-button>
              <el-button v-if="row.status === 'failed'" type="warning" link @click="handleRetry(row.run_id)">重试</el-button>
              <el-button v-if="row.status === 'running' || row.status === 'pending'" type="danger" link @click="handleCancel(row.run_id)">取消</el-button>
            </template>
          </el-table-column>
        </el-table>
        <el-pagination
          v-if="runTotal > runsPageSize"
          :current-page="runsPage"
          :page-size="runsPageSize"
          :total="runTotal"
          layout="total, prev, pager, next"
          style="margin-top: 12px; justify-content: flex-end"
          @current-change="onRunsPageChange"
        />
      </div>

      <!-- 任务定义 Tab -->
      <div v-show="activeTab === 'definitions'">
        <el-table :data="definitions" v-loading="loadingDefs" border stripe>
          <el-table-column prop="name" label="名称" min-width="180" show-overflow-tooltip />
          <el-table-column label="触发" width="80">
            <template #default="{ row }">{{ row.trigger_mode }}</template>
          </el-table-column>
          <el-table-column label="配置摘要" min-width="220" show-overflow-tooltip>
            <template #default="{ row }">{{ configSummary(row.config) }}</template>
          </el-table-column>
          <el-table-column label="创建时间" width="170">
            <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
          </el-table-column>
          <el-table-column label="操作" width="240" fixed="right">
            <template #default="{ row }">
              <el-button type="success" link @click="executeDefinition(row)">▶ 执行</el-button>
              <el-button type="primary" link @click="$emit('edit', row)">编辑</el-button>
              <el-button type="info" link @click="$emit('duplicate', row)">复制</el-button>
              <el-popconfirm title="确定删除？" @confirm="removeDefinition(row.id)">
                <template #reference>
                  <el-button type="danger" link @click.stop>删除</el-button>
                </template>
              </el-popconfirm>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-card>

    <!-- 日志 Drawer -->
    <el-drawer v-model="logDrawerVisible" title="任务日志" size="45%" destroy-on-close>
      <div class="log-drawer-content">
        <div v-if="logDrawerStatus" class="log-status-bar">
          <span>状态：</span>
          <el-tag :type="statusTagType(logDrawerStatus)" size="small">{{ logDrawerStatus }}</el-tag>
        </div>
        <el-input v-model="logDrawerText" type="textarea" :rows="28" readonly placeholder="暂无日志..." class="log-area" />
      </div>
    </el-drawer>

    <!-- 详情 Drawer -->
    <RunDetailDrawer v-model="detailDrawerVisible" title="运行详情" :loading="loadingDetail" :detail="runDetail" />
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { formatTime, getApiData } from '../utils/format'
import StatCard from './StatCard.vue'
import RunDetailDrawer from './RunDetailDrawer.vue'
import {
  fetchTaskCenterDefinitions,
  createTaskCenterRun,
  fetchTaskCenterRuns,
  fetchTaskCenterRunStatus,
  cancelTaskCenterRun,
  retryTaskCenterRun,
  type TaskCenterDefinitionRow,
  type TaskCenterRunRow,
  type TaskCenterRunStatusData,
} from '../api/taskCenter'

type DefRow = TaskCenterDefinitionRow

const props = defineProps<{
  taskType: string
  getLogFn?: (taskId: string, offset?: number) => Promise<unknown>
}>()

const emit = defineEmits<{
  (e: 'create'): void
  (e: 'edit', row: DefRow): void
  (e: 'duplicate', row: DefRow): void
  (e: 'executed', runId: string): void
}>()

const activeTab = ref('runs')

// Stats
const runStats = ref({ running: 0, completed: 0, failed: 0 })

// Definitions
const loadingDefs = ref(false)
const definitions = ref<DefRow[]>([])

// Runs
const loadingRuns = ref(false)
const runs = ref<TaskCenterRunRow[]>([])
const runsPage = ref(1)
const runsPageSize = 15
const runTotal = ref(0)

// Detail drawer
const detailDrawerVisible = ref(false)
const loadingDetail = ref(false)
const runDetail = ref<TaskCenterRunStatusData | null>(null)

// Log drawer
const logDrawerVisible = ref(false)
const logDrawerRunId = ref('')
const logDrawerText = ref('')
const logDrawerStatus = ref('')
const logDrawerOffset = ref(0)
let logPollTimer: ReturnType<typeof setInterval> | null = null

function statusTagType(status: string): '' | 'success' | 'warning' | 'danger' | 'info' {
  if (status === 'completed') return 'success'
  if (status === 'running' || status === 'pending') return 'warning'
  if (status === 'failed' || status === 'timeout') return 'danger'
  if (status === 'cancelled') return 'info'
  return ''
}

function configSummary(config: Record<string, unknown> | null): string {
  if (!config) return '-'
  const parts: string[] = []
  if (config.algorithm) parts.push(`算法:${config.algorithm}`)
  if (config.config_name) parts.push(`配置:${config.config_name}`)
  if (config.model_family) parts.push(`模型族:${config.model_family}`)
  if (config.input_files) {
    const files = config.input_files as string[]
    parts.push(`文件:${files.length}个`)
  }
  if (config.iotdb_source_name) parts.push(`源:${config.iotdb_source_name}`)
  return parts.join(' | ') || JSON.stringify(config).substring(0, 60)
}

// ==================== Data Loading ====================

async function loadDefinitions(): Promise<void> {
  loadingDefs.value = true
  try {
    const resp = await fetchTaskCenterDefinitions()
    const data = getApiData<{ definitions?: DefRow[] }>(resp)
    const allDefs = Array.isArray(data.definitions) ? data.definitions : []
    definitions.value = allDefs.filter(d => d.task_type === props.taskType)
  } finally {
    loadingDefs.value = false
  }
}

async function loadRuns(): Promise<void> {
  loadingRuns.value = true
  try {
    const resp = await fetchTaskCenterRuns({
      task_type: props.taskType,
      limit: runsPageSize,
      offset: (runsPage.value - 1) * runsPageSize,
    })
    const data = (resp && typeof resp === 'object' ? (resp as { data?: Record<string, unknown> }).data : {}) || {}
    runTotal.value = Number(data.total || 0)
    runs.value = Array.isArray(data.runs) ? (data.runs as TaskCenterRunRow[]) : []

    // Update stats
    const allRunsResp = await fetchTaskCenterRuns({ task_type: props.taskType, limit: 200 })
    const allData = (allRunsResp && typeof allRunsResp === 'object' ? (allRunsResp as { data?: Record<string, unknown> }).data : {}) || {}
    const allRuns = Array.isArray(allData.runs) ? (allData.runs as TaskCenterRunRow[]) : []
    runStats.value = {
      running: allRuns.filter(r => r.status === 'running' || r.status === 'pending').length,
      completed: allRuns.filter(r => r.status === 'completed').length,
      failed: allRuns.filter(r => r.status === 'failed' || r.status === 'timeout').length,
    }
  } finally {
    loadingRuns.value = false
  }
}

async function refreshAll(): Promise<void> {
  await Promise.all([loadDefinitions(), loadRuns()])
}

function onTabChange(): void {
  if (activeTab.value === 'definitions' && definitions.value.length === 0) loadDefinitions()
}

function onRunsPageChange(page: number): void {
  runsPage.value = page
  loadRuns()
}

// ==================== Definition Actions ====================

async function executeDefinition(row: DefRow): Promise<void> {
  try {
    const resp = await createTaskCenterRun({
      definition_id: row.id,
      task_type: row.task_type,
      trigger_mode: 'manual',
      input_payload: row.config,
      auto_execute: true,
    })
    const data = (resp && typeof resp === 'object' ? (resp as { data?: Record<string, unknown> }).data : {}) || {}
    const runId = String(data.run_id || '')
    if (runId) {
      ElMessage.success(`任务已执行: ${runId.substring(0, 8)}...`)
      activeTab.value = 'runs'
      emit('executed', runId)
      await loadRuns()
    }
  } catch {
    ElMessage.error('执行失败')
  }
}

function removeDefinition(id: string): void {
  definitions.value = definitions.value.filter(d => d.id !== id)
  ElMessage.success('定义已移除')
}

// ==================== Run Actions ====================

async function handleRetry(runId: string): Promise<void> {
  await retryTaskCenterRun(runId)
  ElMessage.success('重试已提交')
  await loadRuns()
}

async function handleCancel(runId: string): Promise<void> {
  await cancelTaskCenterRun(runId)
  ElMessage.success('取消已提交')
  await loadRuns()
}

// ==================== Detail Drawer ====================

async function openDetailDrawer(runId: string): Promise<void> {
  loadingDetail.value = true
  detailDrawerVisible.value = true
  try {
    const res = await fetchTaskCenterRunStatus(runId)
    runDetail.value = ((res as { data?: unknown })?.data ?? {}) as TaskCenterRunStatusData
  } finally {
    loadingDetail.value = false
  }
}

// ==================== Log Drawer ====================

function stopLogPolling(): void {
  if (logPollTimer) { clearInterval(logPollTimer); logPollTimer = null }
}

async function openLogDrawer(runId: string): Promise<void> {
  stopLogPolling()
  logDrawerRunId.value = runId
  logDrawerText.value = ''
  logDrawerStatus.value = ''
  logDrawerOffset.value = 0
  logDrawerVisible.value = true
  await fetchLog()
  // Auto-poll if running
  const run = runs.value.find(r => r.run_id === runId)
  if (run && (run.status === 'running' || run.status === 'pending')) {
    logPollTimer = setInterval(() => { fetchLog().catch(() => {}) }, 2000)
  }
}

async function fetchLog(): Promise<void> {
  if (!logDrawerRunId.value || !props.getLogFn) return
  try {
    const resp = await props.getLogFn(logDrawerRunId.value, logDrawerOffset.value)
    const data = getApiData<{ status?: string; log?: string; offset?: number }>(resp)
    if (data.status) logDrawerStatus.value = data.status
    const chunk = String(data.log || '')
    if (chunk) logDrawerText.value = logDrawerText.value ? `${logDrawerText.value}\n${chunk}` : chunk
    logDrawerOffset.value = Number(data.offset || logDrawerOffset.value)
    if (['completed', 'failed', 'cancelled', 'timeout', 'stopped'].includes(logDrawerStatus.value)) {
      stopLogPolling()
      await loadRuns()
    }
  } catch { /* ignore */ }
}

defineExpose({ loadDefinitions, loadRuns, refreshAll })

onMounted(() => { refreshAll() })
onUnmounted(() => { stopLogPolling() })
</script>

<style scoped>
.task-def-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.stats-row {
  margin-bottom: 4px;
}

.section-header {
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

.header-actions {
  display: flex;
  gap: 8px;
  flex-shrink: 0;
}

.log-drawer-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: 100%;
}

.log-status-bar {
  display: flex;
  align-items: center;
  gap: 8px;
}

.log-area :deep(textarea) {
  font-family: 'JetBrains Mono', 'Cascadia Code', monospace;
  font-size: 12px;
  line-height: 1.6;
}
</style>
