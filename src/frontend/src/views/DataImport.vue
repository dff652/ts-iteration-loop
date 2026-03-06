<template>
  <div class="data-import-page">
    <PageHeader title="数据导入" subtitle="通过 CSV 上传或 IoTDB 数据源采集导入时序数据" />

    <el-card shadow="never">
      <el-tabs v-model="activeTab">
        <!-- Tab 1: CSV 上传 -->
        <el-tab-pane label="CSV 上传" name="upload">
          <el-form label-width="120px" style="max-width: 640px">
            <el-form-item label="数据集名称">
              <el-input v-model="uploadDatasetName" placeholder="可选，不填使用文件名" clearable />
            </el-form-item>
            <el-form-item label="选择文件">
              <el-upload :auto-upload="false" :limit="1" accept=".csv" :on-change="onUploadChange" :on-remove="onUploadRemove">
                <el-button>选择 CSV 文件</el-button>
              </el-upload>
            </el-form-item>
            <el-form-item label="覆盖同名">
              <el-switch v-model="uploadOverwrite" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="uploading" :disabled="!selectedUploadFile" @click="submitUploadDataset">
                上传创建数据集
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <!-- Tab 2: 数据源管理 -->
        <el-tab-pane label="数据源管理" name="sources">
          <div class="toolbar">
            <el-button type="primary" @click="openCreateDialog">+ 新建数据源</el-button>
            <el-button @click="loadSources">刷新</el-button>
          </div>

          <el-table :data="sources" v-loading="loadingSources" border stripe>
            <el-table-column prop="name" label="名称" width="160" />
            <el-table-column label="地址" width="200">
              <template #default="{ row }">{{ row.host }}:{{ row.port }}</template>
            </el-table-column>
            <el-table-column prop="source_path" label="IoTDB 路径" min-width="260" show-overflow-tooltip />
            <el-table-column prop="point_name" label="点位" width="100" />
            <el-table-column prop="target_points" label="目标点数" width="100" />
            <el-table-column prop="description" label="备注" min-width="140" show-overflow-tooltip />
            <el-table-column label="操作" width="220" fixed="right">
              <template #default="{ row }">
                <el-button type="success" link @click="openAcquireDialog(row)">采集</el-button>
                <el-button type="primary" link @click="openEditDialog(row)">编辑</el-button>
                <el-popconfirm title="确定删除？" @confirm="handleDeleteSource(row.id)">
                  <template #reference>
                    <el-button type="danger" link>删除</el-button>
                  </template>
                </el-popconfirm>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <!-- 采集任务状态 -->
    <TaskLogPanel
      v-if="lastTaskId"
      title="采集日志"
      :log-text="taskLogText"
      :status="taskStatus"
      placeholder="等待任务运行..."
      @refresh="refreshAcquireRuntime"
    />

    <!-- 新建/编辑数据源弹窗 -->
    <el-dialog v-model="sourceDialogVisible" :title="editingSource ? '编辑数据源' : '新建数据源'" width="560px" destroy-on-close>
      <el-form :model="sourceForm" label-width="120px">
        <el-form-item label="名称" required>
          <el-input v-model="sourceForm.name" placeholder="例: 兆底4C-1216" />
        </el-form-item>
        <el-form-item label="Host / Port">
          <el-row :gutter="8">
            <el-col :span="14"><el-input v-model="sourceForm.host" placeholder="192.168.199.185" /></el-col>
            <el-col :span="10"><el-input v-model="sourceForm.port" placeholder="6667" /></el-col>
          </el-row>
        </el-form-item>
        <el-form-item label="用户名" required>
          <el-input v-model="sourceForm.username" placeholder="user" />
        </el-form-item>
        <el-form-item label="密码" required>
          <el-input v-model="sourceForm.password" type="password" show-password placeholder="password" />
        </el-form-item>
        <el-form-item label="IoTDB 路径" required>
          <el-input v-model="sourceForm.source_path" placeholder="root.zhlh_202307_202412.ZHLH_4C_1216" />
        </el-form-item>
        <el-form-item label="点位名称">
          <el-input v-model="sourceForm.point_name" placeholder="* 或具体点位" />
        </el-form-item>
        <el-form-item label="默认点数">
          <el-input-number v-model="sourceForm.target_points" :min="100" :max="200000" :step="500" style="width: 100%" />
        </el-form-item>
        <el-form-item label="备注">
          <el-input v-model="sourceForm.description" type="textarea" :rows="2" placeholder="可选" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="sourceDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="savingSource" @click="handleSaveSource">
          {{ editingSource ? '更新' : '创建' }}
        </el-button>
      </template>
    </el-dialog>

    <!-- 采集参数弹窗 -->
    <el-dialog v-model="acquireDialogVisible" :title="`采集: ${acquireSourceName}`" width="480px" destroy-on-close>
      <el-form :model="acquireForm" label-width="120px">
        <el-form-item label="目标点数">
          <el-input-number v-model="acquireForm.target_points" :min="100" :max="200000" :step="500" style="width: 100%" />
        </el-form-item>
        <el-form-item label="开始时间">
          <el-input v-model="acquireForm.start_time" placeholder="2023-01-01 00:00:00（可选）" />
        </el-form-item>
        <el-form-item label="结束时间">
          <el-input v-model="acquireForm.end_time" placeholder="2023-01-02 00:00:00（可选）" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="acquireDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="handleSubmitAcquire">提交采集</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { getApiData } from '../utils/format'
import PageHeader from '../components/PageHeader.vue'
import TaskLogPanel from '../components/TaskLogPanel.vue'
import { uploadDatasetFile } from '../api/data'
import { getAcquireTaskStatus, getAcquireTaskLog } from '../api/data'
import {
  listIotdbSources,
  createIotdbSource,
  updateIotdbSource,
  deleteIotdbSource,
  acquireFromSource,
  type IotdbSourceItem,
} from '../api/iotdbSource'

type AcquireStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'timeout' | ''

const activeTab = ref('sources')

// CSV upload state
const uploading = ref(false)
const uploadDatasetName = ref('')
const uploadOverwrite = ref(false)
const selectedUploadFile = ref<File | null>(null)

// Sources state
const sources = ref<IotdbSourceItem[]>([])
const loadingSources = ref(false)
const sourceDialogVisible = ref(false)
const savingSource = ref(false)
const editingSource = ref<IotdbSourceItem | null>(null)

const sourceForm = reactive({
  name: '',
  host: '192.168.199.185',
  port: '6667',
  username: '',
  password: '',
  source_path: '',
  point_name: '*',
  target_points: 5000,
  description: '',
})

// Acquire state
const acquireDialogVisible = ref(false)
const acquireSourceId = ref('')
const acquireSourceName = ref('')
const submitting = ref(false)
const acquireForm = reactive({
  target_points: 5000,
  start_time: '',
  end_time: '',
})

// Task polling state
const lastTaskId = ref('')
const taskStatus = ref<AcquireStatus>('')
const taskLogText = ref('')
const logOffset = ref(0)
let pollTimer: ReturnType<typeof setInterval> | null = null

// ==================== Sources CRUD ====================

async function loadSources(): Promise<void> {
  loadingSources.value = true
  try {
    const resp = await listIotdbSources()
    const data = getApiData<{ sources?: IotdbSourceItem[] }>(resp)
    sources.value = Array.isArray(data.sources) ? data.sources : []
  } finally {
    loadingSources.value = false
  }
}

function openCreateDialog(): void {
  editingSource.value = null
  Object.assign(sourceForm, {
    name: '', host: '192.168.199.185', port: '6667',
    username: '', password: '', source_path: '',
    point_name: '*', target_points: 5000, description: '',
  })
  sourceDialogVisible.value = true
}

function openEditDialog(row: IotdbSourceItem): void {
  editingSource.value = row
  Object.assign(sourceForm, {
    name: row.name, host: row.host, port: row.port,
    username: row.username, password: '',
    source_path: row.source_path, point_name: row.point_name,
    target_points: row.target_points, description: row.description || '',
  })
  sourceDialogVisible.value = true
}

async function handleSaveSource(): Promise<void> {
  if (!sourceForm.name.trim()) { ElMessage.warning('名称不能为空'); return }
  if (!sourceForm.source_path.trim()) { ElMessage.warning('IoTDB 路径不能为空'); return }

  savingSource.value = true
  try {
    if (editingSource.value) {
      await updateIotdbSource(editingSource.value.id, {
        name: sourceForm.name.trim(),
        host: sourceForm.host.trim(),
        port: sourceForm.port.trim(),
        username: sourceForm.username.trim() || undefined,
        password: sourceForm.password.trim() || undefined,
        source_path: sourceForm.source_path.trim(),
        point_name: sourceForm.point_name.trim(),
        target_points: sourceForm.target_points,
        description: sourceForm.description.trim() || undefined,
      })
      ElMessage.success('数据源已更新')
    } else {
      if (!sourceForm.username.trim() || !sourceForm.password.trim()) {
        ElMessage.warning('用户名和密码不能为空'); return
      }
      await createIotdbSource({
        name: sourceForm.name.trim(),
        host: sourceForm.host.trim(),
        port: sourceForm.port.trim(),
        username: sourceForm.username.trim(),
        password: sourceForm.password.trim(),
        source_path: sourceForm.source_path.trim(),
        point_name: sourceForm.point_name.trim(),
        target_points: sourceForm.target_points,
        description: sourceForm.description.trim() || undefined,
      })
      ElMessage.success('数据源已创建')
    }
    sourceDialogVisible.value = false
    await loadSources()
  } finally {
    savingSource.value = false
  }
}

async function handleDeleteSource(id: string): Promise<void> {
  await deleteIotdbSource(id)
  ElMessage.success('数据源已删除')
  await loadSources()
}

// ==================== Acquire ====================

function openAcquireDialog(row: IotdbSourceItem): void {
  acquireSourceId.value = row.id
  acquireSourceName.value = row.name
  acquireForm.target_points = row.target_points
  acquireForm.start_time = ''
  acquireForm.end_time = ''
  acquireDialogVisible.value = true
}

async function handleSubmitAcquire(): Promise<void> {
  submitting.value = true
  try {
    const resp = await acquireFromSource(acquireSourceId.value, {
      target_points: acquireForm.target_points,
      start_time: acquireForm.start_time.trim() || undefined,
      end_time: acquireForm.end_time.trim() || undefined,
    })
    const payload = (resp as unknown as { task_id?: string; status?: string }) ?? {}
    const taskId = String(payload.task_id || '')
    if (!taskId) throw new Error('未获取到任务ID')

    lastTaskId.value = taskId
    taskStatus.value = (String(payload.status || 'pending')) as AcquireStatus
    taskLogText.value = ''
    logOffset.value = 0
    acquireDialogVisible.value = false
    ElMessage.success('采集任务已提交')
    startPolling()
  } finally {
    submitting.value = false
  }
}

// ==================== Polling ====================

function stopPolling(): void {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

function startPolling(): void {
  stopPolling()
  refreshAcquireRuntime()
  pollTimer = setInterval(() => refreshAcquireRuntime().catch(() => {}), 2000)
}

async function refreshAcquireRuntime(): Promise<void> {
  if (!lastTaskId.value) return
  const [statusResp, logResp] = await Promise.all([
    getAcquireTaskStatus(lastTaskId.value),
    getAcquireTaskLog(lastTaskId.value, logOffset.value),
  ])
  const statusData = getApiData<{ status?: string; message?: string }>(statusResp)
  taskStatus.value = (String(statusData.status || '')) as AcquireStatus

  const logData = getApiData<{ log?: string; offset?: number }>(logResp)
  const chunk = String(logData.log || '')
  if (chunk) taskLogText.value = taskLogText.value ? `${taskLogText.value}\n${chunk}` : chunk
  logOffset.value = Number(logData.offset || logOffset.value)

  if (['completed', 'failed', 'cancelled', 'timeout'].includes(taskStatus.value)) stopPolling()
}

// ==================== CSV Upload ====================

function onUploadChange(file: { raw?: File }): void { selectedUploadFile.value = file?.raw || null }
function onUploadRemove(): void { selectedUploadFile.value = null }

async function submitUploadDataset(): Promise<void> {
  if (!selectedUploadFile.value) { ElMessage.warning('请先选择 CSV 文件'); return }
  uploading.value = true
  try {
    await uploadDatasetFile(selectedUploadFile.value, uploadDatasetName.value.trim() || undefined, uploadOverwrite.value)
    ElMessage.success('CSV 数据集上传成功')
    selectedUploadFile.value = null
    uploadDatasetName.value = ''
  } finally {
    uploading.value = false
  }
}

onMounted(() => { loadSources() })
onUnmounted(() => { stopPolling() })
</script>

<style scoped>
.data-import-page {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
}
</style>
