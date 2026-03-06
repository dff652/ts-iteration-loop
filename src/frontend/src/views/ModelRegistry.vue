<template>
  <div class="model-registry-page">
    <PageHeader title="模型仓库" subtitle="模型注册、版本对比与参数配置管理" />

    <el-card shadow="never">
      <el-tabs v-model="activeTab">
        <!-- Tab 1: 模型列表 -->
        <el-tab-pane label="模型列表" name="list">
          <div class="toolbar">
            <el-select v-model="filterFamily" placeholder="模型族" clearable style="width: 120px" @change="loadModels">
              <el-option label="chatts" value="chatts" />
              <el-option label="qwen" value="qwen" />
            </el-select>
            <el-select v-model="filterType" placeholder="类型" clearable style="width: 100px" @change="loadModels">
              <el-option label="lora" value="lora" />
              <el-option label="full" value="full" />
              <el-option label="base" value="base" />
            </el-select>
            <el-select v-model="filterStatus" placeholder="状态" clearable style="width: 110px" @change="loadModels">
              <el-option label="active" value="active" />
              <el-option label="archived" value="archived" />
              <el-option label="deprecated" value="deprecated" />
            </el-select>
            <el-input v-model="filterKeyword" placeholder="搜索名称/路径" clearable style="width: 200px" @keyup.enter="loadModels" />
            <el-button type="primary" @click="loadModels">查询</el-button>
            <el-button @click="openRegisterDialog">注册模型</el-button>
            <el-button type="success" :loading="scanning" @click="scanDiskModels">扫描注册</el-button>
          </div>

          <el-table :data="modelList" v-loading="loadingList" border stripe @selection-change="onSelectionChange">
            <el-table-column type="selection" width="45" />
            <el-table-column prop="name" label="名称" min-width="160" show-overflow-tooltip />
            <el-table-column prop="model_family" label="族" width="80" />
            <el-table-column prop="model_type" label="类型" width="70" />
            <el-table-column prop="version" label="版本" width="110" show-overflow-tooltip />
            <el-table-column label="Loss" width="90">
              <template #default="{ row }">{{ row.train_loss != null ? row.train_loss.toFixed(4) : '-' }}</template>
            </el-table-column>
            <el-table-column label="状态" width="100">
              <template #default="{ row }">
                <el-tag :type="statusTagType(row.status)">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="标签" min-width="140">
              <template #default="{ row }">
                <el-tag v-for="tag in (row.tags || [])" :key="tag" size="small" style="margin-right: 4px">{{ tag }}</el-tag>
                <span v-if="!row.tags || row.tags.length === 0">-</span>
              </template>
            </el-table-column>
            <el-table-column label="创建时间" width="170">
              <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
            </el-table-column>
            <el-table-column label="操作" width="220" fixed="right">
              <template #default="{ row }">
                <el-button type="primary" link @click="openDetailDrawer(row.id)">详情</el-button>
                <el-button type="warning" link @click="openEditDialog(row)">编辑</el-button>
                <el-popconfirm title="确认归档此模型？" @confirm="archiveModel(row.id)">
                  <template #reference>
                    <el-button type="danger" link>归档</el-button>
                  </template>
                </el-popconfirm>
              </template>
            </el-table-column>
          </el-table>

          <div class="pager-wrap">
            <el-pagination
              layout="total, prev, pager, next"
              :current-page="page"
              :page-size="pageSize"
              :total="total"
              @current-change="onPageChange"
            />
          </div>
        </el-tab-pane>

        <!-- Tab 2: 版本对比 -->
        <el-tab-pane label="版本对比" name="compare">
          <div class="toolbar">
            <el-select v-model="compareFamily" placeholder="模型族" style="width: 120px" @change="loadVersionHistory">
              <el-option label="chatts" value="chatts" />
              <el-option label="qwen" value="qwen" />
            </el-select>
            <el-button type="primary" @click="loadVersionHistory">刷新</el-button>
            <el-button
              type="success"
              :disabled="selectedForCompare.length < 2"
              @click="runComparison"
            >
              对比选中 ({{ selectedForCompare.length }})
            </el-button>
          </div>

          <el-table :data="versionList" v-loading="loadingVersions" border stripe @selection-change="onCompareSelectionChange">
            <el-table-column type="selection" width="45" />
            <el-table-column prop="name" label="名称" min-width="180" show-overflow-tooltip />
            <el-table-column prop="model_type" label="类型" width="80" />
            <el-table-column prop="version" label="版本" width="120" />
            <el-table-column label="Loss" width="100">
              <template #default="{ row }">{{ row.train_loss != null ? row.train_loss.toFixed(4) : '-' }}</template>
            </el-table-column>
            <el-table-column label="状态" width="100">
              <template #default="{ row }">
                <el-tag :type="statusTagType(row.status)">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="创建时间" width="170">
              <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
            </el-table-column>
          </el-table>

          <!-- 对比结果 -->
          <template v-if="comparisonResult.length > 0">
            <el-divider content-position="left">对比结果</el-divider>
            <el-table :data="comparisonResult" border stripe>
              <el-table-column prop="name" label="名称" min-width="160" show-overflow-tooltip />
              <el-table-column prop="model_type" label="类型" width="80" />
              <el-table-column label="Loss" width="100">
                <template #default="{ row }">{{ row.train_loss != null ? row.train_loss.toFixed(4) : '-' }}</template>
              </el-table-column>
              <el-table-column label="F1 Score" width="110">
                <template #default="{ row }">{{ getEvalMetric(row, 'f1_score') }}</template>
              </el-table-column>
              <el-table-column label="Precision" width="110">
                <template #default="{ row }">{{ getEvalMetric(row, 'precision') }}</template>
              </el-table-column>
              <el-table-column label="Recall" width="110">
                <template #default="{ row }">{{ getEvalMetric(row, 'recall') }}</template>
              </el-table-column>
              <el-table-column prop="model_path" label="路径" min-width="260" show-overflow-tooltip />
            </el-table>
          </template>
        </el-tab-pane>

        <!-- Tab 3: 参数配置 -->
        <el-tab-pane label="参数配置" name="config">
          <div class="toolbar">
            <el-select v-model="configModelId" placeholder="选择模型查看配置" style="width: 350px" filterable @change="loadModelConfig">
              <el-option
                v-for="m in modelList"
                :key="m.id"
                :label="`${m.name} (${m.model_family}/${m.model_type})`"
                :value="m.id"
              />
            </el-select>
            <el-button @click="loadModels">刷新列表</el-button>
          </div>

          <template v-if="configDetail">
            <el-descriptions :column="2" border style="margin-top: 16px">
              <el-descriptions-item label="名称">{{ configDetail.name }}</el-descriptions-item>
              <el-descriptions-item label="模型族">{{ configDetail.model_family }}</el-descriptions-item>
              <el-descriptions-item label="类型">{{ configDetail.model_type }}</el-descriptions-item>
              <el-descriptions-item label="版本">{{ configDetail.version || '-' }}</el-descriptions-item>
              <el-descriptions-item label="模型路径" :span="2">
                <code>{{ configDetail.model_path }}</code>
              </el-descriptions-item>
              <el-descriptions-item label="基础模型" :span="2">
                <code>{{ configDetail.base_model || '-' }}</code>
              </el-descriptions-item>
              <el-descriptions-item label="训练 Loss">{{ configDetail.train_loss != null ? configDetail.train_loss.toFixed(4) : '-' }}</el-descriptions-item>
              <el-descriptions-item label="状态">
                <el-tag :type="statusTagType(configDetail.status)">{{ configDetail.status }}</el-tag>
              </el-descriptions-item>
            </el-descriptions>

            <el-divider content-position="left">训练参数</el-divider>
            <el-input
              :model-value="formatJson(configDetail.config)"
              type="textarea"
              :rows="12"
              readonly
              style="font-family: monospace"
            />

            <template v-if="configDetail.metrics">
              <el-divider content-position="left">评估指标</el-divider>
              <el-input
                :model-value="formatJson(configDetail.metrics)"
                type="textarea"
                :rows="8"
                readonly
                style="font-family: monospace"
              />
            </template>
          </template>
          <el-empty v-else description="请选择模型查看配置" />
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <!-- 模型详情 Drawer -->
    <el-drawer v-model="detailVisible" title="模型详情" size="55%">
      <el-skeleton :rows="5" animated v-if="loadingDetail" />
      <template v-else-if="detailData">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="ID">{{ detailData.id }}</el-descriptions-item>
          <el-descriptions-item label="名称">{{ detailData.name }}</el-descriptions-item>
          <el-descriptions-item label="模型族">{{ detailData.model_family }}</el-descriptions-item>
          <el-descriptions-item label="类型">{{ detailData.model_type }}</el-descriptions-item>
          <el-descriptions-item label="版本">{{ detailData.version || '-' }}</el-descriptions-item>
          <el-descriptions-item label="训练 Loss">{{ detailData.train_loss != null ? detailData.train_loss.toFixed(6) : '-' }}</el-descriptions-item>
          <el-descriptions-item label="模型路径" :span="2">
            <code>{{ detailData.model_path }}</code>
          </el-descriptions-item>
          <el-descriptions-item label="基础模型" :span="2">
            <code>{{ detailData.base_model || '-' }}</code>
          </el-descriptions-item>
          <el-descriptions-item label="描述" :span="2">{{ detailData.description || '-' }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="statusTagType(detailData.status)">{{ detailData.status }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="创建者">{{ detailData.created_by || '-' }}</el-descriptions-item>
          <el-descriptions-item label="创建时间">{{ formatTime(detailData.created_at) }}</el-descriptions-item>
          <el-descriptions-item label="更新时间">{{ formatTime(detailData.updated_at) }}</el-descriptions-item>
        </el-descriptions>

        <template v-if="detailData.tags && detailData.tags.length > 0">
          <el-divider content-position="left">标签</el-divider>
          <el-tag v-for="tag in detailData.tags" :key="tag" style="margin-right: 6px">{{ tag }}</el-tag>
        </template>

        <template v-if="detailData.config">
          <el-divider content-position="left">训练参数</el-divider>
          <el-input :model-value="formatJson(detailData.config)" type="textarea" :rows="8" readonly style="font-family: monospace" />
        </template>

        <template v-if="detailData.evaluations && detailData.evaluations.length > 0">
          <el-divider content-position="left">评估记录</el-divider>
          <el-table :data="detailData.evaluations" border stripe>
            <el-table-column prop="dataset_name" label="评估集" width="140" />
            <el-table-column label="F1" width="100">
              <template #default="{ row }">{{ row.metrics?.summary?.f1_score?.toFixed(4) ?? '-' }}</template>
            </el-table-column>
            <el-table-column label="Precision" width="100">
              <template #default="{ row }">{{ row.metrics?.summary?.precision?.toFixed(4) ?? '-' }}</template>
            </el-table-column>
            <el-table-column label="Recall" width="100">
              <template #default="{ row }">{{ row.metrics?.summary?.recall?.toFixed(4) ?? '-' }}</template>
            </el-table-column>
            <el-table-column label="评估时间" min-width="160">
              <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
            </el-table-column>
          </el-table>
        </template>
      </template>
    </el-drawer>

    <!-- 注册/编辑对话框 -->
    <el-dialog v-model="dialogVisible" :title="editingId ? '编辑模型' : '注册模型'" width="600px">
      <el-form label-width="100px">
        <el-form-item label="名称" required>
          <el-input v-model="dialogForm.name" placeholder="模型显示名" />
        </el-form-item>
        <el-form-item label="模型族" v-if="!editingId">
          <el-select v-model="dialogForm.model_family" style="width: 100%">
            <el-option label="chatts" value="chatts" />
            <el-option label="qwen" value="qwen" />
          </el-select>
        </el-form-item>
        <el-form-item label="类型" v-if="!editingId">
          <el-select v-model="dialogForm.model_type" style="width: 100%">
            <el-option label="lora" value="lora" />
            <el-option label="full" value="full" />
            <el-option label="base" value="base" />
          </el-select>
        </el-form-item>
        <el-form-item label="版本">
          <el-input v-model="dialogForm.version" placeholder="v1.0 / v20260304" />
        </el-form-item>
        <el-form-item label="模型路径" required v-if="!editingId">
          <el-input v-model="dialogForm.model_path" placeholder="/path/to/model" />
        </el-form-item>
        <el-form-item label="基础模型" v-if="!editingId">
          <el-input v-model="dialogForm.base_model" placeholder="/path/to/base_model" />
        </el-form-item>
        <el-form-item label="状态">
          <el-select v-model="dialogForm.status" style="width: 100%">
            <el-option label="active" value="active" />
            <el-option label="archived" value="archived" />
            <el-option label="deprecated" value="deprecated" />
          </el-select>
        </el-form-item>
        <el-form-item label="标签">
          <el-input v-model="dialogForm.tags_str" placeholder="用逗号分隔，如: v1,实验" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="dialogForm.description" type="textarea" :rows="3" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="submitDialog">{{ editingId ? '保存' : '注册' }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { formatTime, formatJson, statusTagType, getApiData } from '../utils/format'
import PageHeader from '../components/PageHeader.vue'

import {
  compareModels,
  deleteModel,
  fetchModelDetail,
  fetchModels,
  fetchVersionHistory,
  registerModel,
  scanModels,
  updateModel,
  type ModelRegistryItem,
} from '../api/modelRegistry'

const activeTab = ref('list')

// ========== 模型列表 ==========
const loadingList = ref(false)
const modelList = ref<ModelRegistryItem[]>([])
const filterFamily = ref('')
const filterType = ref('')
const filterStatus = ref('')
const filterKeyword = ref('')
const page = ref(1)
const pageSize = 20
const total = ref(0)
const selectedRows = ref<ModelRegistryItem[]>([])
const scanning = ref(false)

// ========== 版本对比 ==========
const loadingVersions = ref(false)
const compareFamily = ref('chatts')
const versionList = ref<ModelRegistryItem[]>([])
const selectedForCompare = ref<ModelRegistryItem[]>([])
const comparisonResult = ref<ModelRegistryItem[]>([])

// ========== 参数配置 ==========
const configModelId = ref('')
const configDetail = ref<ModelRegistryItem | null>(null)

// ========== 详情 Drawer ==========
const detailVisible = ref(false)
const loadingDetail = ref(false)
const detailData = ref<ModelRegistryItem & { evaluations?: unknown[] } | null>(null)

// ========== 注册/编辑对话框 ==========
const dialogVisible = ref(false)
const editingId = ref('')
const submitting = ref(false)
const dialogForm = reactive({
  name: '',
  model_family: 'chatts',
  model_type: 'lora',
  version: '',
  model_path: '',
  base_model: '',
  status: 'active',
  tags_str: '',
  description: '',
})


function getEvalMetric(row: ModelRegistryItem & { latest_eval?: { metrics?: { summary?: Record<string, number> } } }, key: string): string {
  const val = row.latest_eval?.metrics?.summary?.[key]
  return val != null ? val.toFixed(4) : '-'
}

// ==================== 模型列表 ====================

async function loadModels(): Promise<void> {
  loadingList.value = true
  try {
    const params: Record<string, unknown> = {
      limit: pageSize,
      offset: (page.value - 1) * pageSize,
    }
    if (filterFamily.value) params.model_family = filterFamily.value
    if (filterType.value) params.model_type = filterType.value
    if (filterStatus.value) params.status = filterStatus.value
    if (filterKeyword.value.trim()) params.keyword = filterKeyword.value.trim()

    const resp = await fetchModels(params)
    const data = getApiData<{ total?: number; models?: ModelRegistryItem[] }>(resp)
    total.value = Number(data.total || 0)
    modelList.value = Array.isArray(data.models) ? data.models : []
  } finally {
    loadingList.value = false
  }
}

function onPageChange(nextPage: number): void {
  page.value = nextPage
  loadModels()
}

function onSelectionChange(rows: ModelRegistryItem[]): void {
  selectedRows.value = rows
}

async function scanDiskModels(): Promise<void> {
  scanning.value = true
  try {
    const resp = await scanModels(filterFamily.value || 'chatts')
    const data = getApiData<{ registered?: number; skipped?: number }>(resp)
    ElMessage.success(`扫描完成: 新注册 ${data.registered || 0}, 已存在 ${data.skipped || 0}`)
    await loadModels()
  } finally {
    scanning.value = false
  }
}

async function archiveModel(modelId: string): Promise<void> {
  await deleteModel(modelId)
  ElMessage.success('模型已归档')
  await loadModels()
}

// ==================== 详情 ====================

async function openDetailDrawer(modelId: string): Promise<void> {
  loadingDetail.value = true
  detailVisible.value = true
  try {
    const resp = await fetchModelDetail(modelId)
    detailData.value = getApiData<ModelRegistryItem & { evaluations?: unknown[] }>(resp)
  } finally {
    loadingDetail.value = false
  }
}

// ==================== 版本对比 ====================

async function loadVersionHistory(): Promise<void> {
  loadingVersions.value = true
  try {
    const resp = await fetchVersionHistory(compareFamily.value)
    const data = getApiData<{ versions?: Record<string, ModelRegistryItem[]> }>(resp)
    const familyVersions = data.versions?.[compareFamily.value]
    versionList.value = Array.isArray(familyVersions) ? familyVersions : []
  } finally {
    loadingVersions.value = false
  }
}

function onCompareSelectionChange(rows: ModelRegistryItem[]): void {
  selectedForCompare.value = rows
}

async function runComparison(): Promise<void> {
  if (selectedForCompare.value.length < 2) {
    ElMessage.warning('请至少选择两个模型')
    return
  }
  const ids = selectedForCompare.value.map((r) => r.id)
  const resp = await compareModels(ids)
  const data = getApiData<{ models?: ModelRegistryItem[] }>(resp)
  comparisonResult.value = Array.isArray(data.models) ? data.models : []
}

// ==================== 参数配置 ====================

async function loadModelConfig(): Promise<void> {
  if (!configModelId.value) {
    configDetail.value = null
    return
  }
  const resp = await fetchModelDetail(configModelId.value)
  configDetail.value = getApiData<ModelRegistryItem>(resp)
}

// ==================== 注册 / 编辑 ====================

function openRegisterDialog(): void {
  editingId.value = ''
  dialogForm.name = ''
  dialogForm.model_family = 'chatts'
  dialogForm.model_type = 'lora'
  dialogForm.version = ''
  dialogForm.model_path = ''
  dialogForm.base_model = ''
  dialogForm.status = 'active'
  dialogForm.tags_str = ''
  dialogForm.description = ''
  dialogVisible.value = true
}

function openEditDialog(row: ModelRegistryItem): void {
  editingId.value = row.id
  dialogForm.name = row.name
  dialogForm.version = row.version || ''
  dialogForm.status = row.status
  dialogForm.tags_str = (row.tags || []).join(',')
  dialogForm.description = row.description || ''
  dialogVisible.value = true
}

async function submitDialog(): Promise<void> {
  if (!dialogForm.name.trim()) {
    ElMessage.warning('名称不能为空')
    return
  }
  submitting.value = true
  try {
    const tagsArr = dialogForm.tags_str
      .split(',')
      .map((t) => t.trim())
      .filter((t) => t.length > 0)

    if (editingId.value) {
      await updateModel(editingId.value, {
        name: dialogForm.name.trim(),
        version: dialogForm.version.trim() || null,
        status: dialogForm.status,
        tags: tagsArr,
        description: dialogForm.description.trim() || null,
      })
      ElMessage.success('模型信息已更新')
    } else {
      if (!dialogForm.model_path.trim()) {
        ElMessage.warning('模型路径不能为空')
        return
      }
      await registerModel({
        name: dialogForm.name.trim(),
        model_family: dialogForm.model_family,
        model_type: dialogForm.model_type,
        version: dialogForm.version.trim() || null,
        model_path: dialogForm.model_path.trim(),
        base_model: dialogForm.base_model.trim() || null,
        status: dialogForm.status,
        tags: tagsArr,
        description: dialogForm.description.trim() || null,
      })
      ElMessage.success('模型已注册')
    }

    dialogVisible.value = false
    await loadModels()
  } finally {
    submitting.value = false
  }
}

// ==================== Init ====================

onMounted(() => {
  loadModels().catch(() => ElMessage.error('加载模型列表失败'))
})
</script>

<style scoped>
.model-registry-page {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.pager-wrap {
  margin-top: 12px;
  display: flex;
  justify-content: flex-end;
}
</style>
