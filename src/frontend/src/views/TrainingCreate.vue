<template>
  <div class="training-page">
    <PageHeader title="训练任务" subtitle="配置并管理时序模型微调训练任务">
      <el-button @click="refreshOptions">刷新选项</el-button>
    </PageHeader>

    <!-- 统一面板: 看板 + Tab(运行记录/定义列表) + 日志Drawer -->
    <TaskDefinitionPanel
      ref="panelRef"
      task-type="training"
      :get-log-fn="getTrainingLog"
      @create="openFormDialog()"
      @edit="openFormDialog($event)"
      @duplicate="openFormDialog($event, true)"
      @executed="onExecuted"
    />

    <!-- 新建/编辑定义 Dialog -->
    <el-dialog v-model="dialogVisible" :title="editingDefId ? '编辑训练定义' : '新建训练定义'" width="650px" destroy-on-close>
      <el-form label-width="130px">
        <!-- ===== 基础设置 ===== -->
        <el-divider content-position="left">基础设置</el-divider>

        <el-form-item label="定义名称">
          <el-input v-model="defName" placeholder="训练-chatts-..." style="width: 100%" />
        </el-form-item>

        <el-form-item label="模型族">
          <el-select v-model="form.modelFamily" style="width: 200px" @change="onFamilyChange">
            <el-option label="chatts" value="chatts" />
            <el-option label="qwen" value="qwen" />
          </el-select>
        </el-form-item>

        <el-form-item label="训练配置">
          <el-select v-model="form.configName" filterable style="width: 100%" placeholder="选择训练配置">
            <el-option v-for="c in configOptions" :key="c.name" :label="`${c.name} (${c.method || 'lora'})`" :value="c.name" />
          </el-select>
        </el-form-item>

        <el-form-item label="训练数据集">
          <el-select v-model="form.dataset" filterable style="width: 100%" placeholder="选择数据集">
            <el-option v-for="d in datasetOptions" :key="d" :label="d" :value="d" />
          </el-select>
        </el-form-item>

        <el-form-item label="版本标签">
          <el-input v-model="form.versionTag" placeholder="可选, 如: v1.0 或 exp_lora_r8" style="width: 300px" />
        </el-form-item>

        <!-- ===== QuickStart 参数 ===== -->
        <el-divider content-position="left">训练参数 (QuickStart)</el-divider>

        <el-form-item label="学习率">
          <el-input-number v-model="form.learningRate" :min="0.000001" :max="0.01" :step="0.00001" :precision="6" controls-position="right" style="width: 200px" />
        </el-form-item>
        <el-form-item label="训练轮数">
          <el-input-number v-model="form.numEpochs" :min="1" :max="100" :step="1" style="width: 200px" />
        </el-form-item>
        <el-form-item label="Batch Size">
          <el-input-number v-model="form.batchSize" :min="1" :max="64" :step="1" style="width: 200px" />
        </el-form-item>
        <el-form-item label="LoRA Rank">
          <el-input-number v-model="form.loraRank" :min="1" :max="256" :step="4" style="width: 200px" />
        </el-form-item>
        <el-form-item label="LoRA Alpha">
          <el-input-number v-model="form.loraAlpha" :min="1" :max="512" :step="8" style="width: 200px" />
        </el-form-item>
        <el-form-item label="设备">
          <el-select v-model="form.device" style="width: 200px">
            <el-option label="auto" value="auto" />
            <el-option label="cuda:0" value="cuda:0" />
            <el-option label="cuda:1" value="cuda:1" />
            <el-option label="cpu" value="cpu" />
          </el-select>
        </el-form-item>

        <!-- ===== 自动评估 ===== -->
        <el-divider content-position="left">训练后评估</el-divider>
        <el-form-item label="自动评估">
          <el-switch v-model="form.autoEval" />
        </el-form-item>
        <el-form-item v-if="form.autoEval" label="评估数据集">
          <el-input v-model="form.evalDatasetName" placeholder="golden" style="width: 300px" />
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="saveDefinition">
          {{ editingDefId ? '更新定义' : '保存定义' }}
        </el-button>
        <el-button v-if="!editingDefId" type="success" :loading="submitting" @click="saveAndExecute">保存并执行</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { getApiData } from '../utils/format'
import PageHeader from '../components/PageHeader.vue'
import TaskDefinitionPanel from '../components/TaskDefinitionPanel.vue'

import {
  getTrainingLog,
  listTrainingConfigs,
  listTrainingDatasets,
  type TrainingConfigOption,
} from '../api/training'
import {
  createTaskCenterDefinition,
  createTaskCenterRun,
  type TaskCenterDefinitionRow,
} from '../api/taskCenter'

const panelRef = ref<InstanceType<typeof TaskDefinitionPanel> | null>(null)
const submitting = ref(false)
const dialogVisible = ref(false)
const editingDefId = ref('')
const defName = ref('')

const configOptions = ref<TrainingConfigOption[]>([])
const datasetOptions = ref<string[]>([])

const form = reactive({
  modelFamily: 'chatts' as 'chatts' | 'qwen',
  configName: '',
  dataset: '',
  versionTag: '',
  learningRate: 0.0001,
  numEpochs: 3,
  batchSize: 2,
  loraRank: 8,
  loraAlpha: 16,
  device: 'auto',
  autoEval: false,
  evalDatasetName: 'golden',
})

// ==================== Helpers ====================

async function refreshOptions(): Promise<void> {
  const family = form.modelFamily
  const [cfgResp, dsResp] = await Promise.all([
    listTrainingConfigs(family), listTrainingDatasets(family),
  ])
  const cfgData = getApiData<{ configs?: TrainingConfigOption[] }>(cfgResp)
  configOptions.value = Array.isArray(cfgData.configs) ? cfgData.configs : []
  if (configOptions.value.length > 0 && !form.configName) {
    form.configName = configOptions.value[0]?.name ?? ''
  }
  const dsData = getApiData<{ datasets?: string[] }>(dsResp)
  datasetOptions.value = Array.isArray(dsData.datasets) ? dsData.datasets : []
  if (datasetOptions.value.length > 0 && !form.dataset) {
    form.dataset = datasetOptions.value[0] ?? ''
  }
}

function onFamilyChange(): void {
  form.configName = ''
  form.dataset = ''
  refreshOptions()
}

function resetForm(): void {
  form.configName = ''
  form.dataset = ''
  form.versionTag = ''
  form.learningRate = 0.0001
  form.numEpochs = 3
  form.batchSize = 2
  form.loraRank = 8
  form.loraAlpha = 16
  form.device = 'auto'
  form.autoEval = false
  form.evalDatasetName = 'golden'
}

// ==================== Dialog ====================

function openFormDialog(row?: TaskCenterDefinitionRow, duplicate = false): void {
  if (row) {
    editingDefId.value = duplicate ? '' : row.id
    defName.value = duplicate ? `${row.name} (副本)` : row.name
    const cfg = row.config || {}
    form.modelFamily = (String(cfg.model_family || 'chatts')) as 'chatts' | 'qwen'
    form.configName = String(cfg.config_name || '')
    form.dataset = String(cfg.dataset || '')
    form.versionTag = String(cfg.version_tag || '')
    form.learningRate = Number(cfg.learning_rate || 0.0001)
    form.numEpochs = Number(cfg.num_train_epochs || 3)
    form.batchSize = Number(cfg.per_device_train_batch_size || 2)
    form.loraRank = Number(cfg.lora_rank || 8)
    form.loraAlpha = Number(cfg.lora_alpha || 16)
    form.device = String(cfg.device || 'auto')
    form.autoEval = Boolean(cfg.auto_eval)
    form.evalDatasetName = String(cfg.eval_dataset_name || 'golden')
  } else {
    editingDefId.value = ''
    defName.value = `训练-${form.modelFamily}-${new Date().toISOString().substring(0, 16)}`
    resetForm()
  }
  dialogVisible.value = true
}

function buildConfig(): Record<string, unknown> {
  return {
    config_name: form.configName,
    model_family: form.modelFamily,
    version_tag: form.versionTag || undefined,
    auto_eval: form.autoEval,
    eval_dataset_name: form.autoEval ? form.evalDatasetName : undefined,
    learning_rate: form.learningRate,
    num_train_epochs: form.numEpochs,
    per_device_train_batch_size: form.batchSize,
    lora_rank: form.loraRank,
    lora_alpha: form.loraAlpha,
    device: form.device,
    dataset: form.dataset || undefined,
  }
}

async function saveDefinition(): Promise<void> {
  if (!form.configName) { ElMessage.warning('请选择训练配置'); return }
  submitting.value = true
  try {
    await createTaskCenterDefinition({
      name: defName.value || `训练-${form.modelFamily}-${form.configName}`,
      task_type: 'training',
      trigger_mode: 'manual',
      config: buildConfig(),
    })
    ElMessage.success('定义已保存')
    dialogVisible.value = false
    panelRef.value?.refreshAll()
  } finally {
    submitting.value = false
  }
}

async function saveAndExecute(): Promise<void> {
  if (!form.configName) { ElMessage.warning('请选择训练配置'); return }
  submitting.value = true
  try {
    const config = buildConfig()
    const defResp = await createTaskCenterDefinition({
      name: defName.value || `训练-${form.modelFamily}-${form.configName}`,
      task_type: 'training',
      trigger_mode: 'manual',
      config,
    })
    const defData = getApiData<{ definition?: { id?: string } }>(defResp)

    const runResp = await createTaskCenterRun({
      definition_id: defData.definition?.id,
      task_type: 'training',
      trigger_mode: 'manual',
      input_payload: config,
      auto_execute: true,
    })
    const runData = (runResp && typeof runResp === 'object' ? (runResp as { data?: Record<string, unknown> }).data : {}) || {}
    const runId = String(runData.run_id || '')
    if (runId) ElMessage.success(`定义已保存并执行: ${runId.substring(0, 8)}...`)
    dialogVisible.value = false
    panelRef.value?.refreshAll()
  } finally {
    submitting.value = false
  }
}

function onExecuted(_runId: string): void {
  // Panel handles refresh internally
}

onMounted(() => { refreshOptions() })
</script>

<style scoped>
.training-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
</style>
