<template>
  <div class="inference-page">
    <PageHeader title="推理任务" subtitle="配置并管理时序异常检测推理任务">
      <el-button @click="refreshOptions">刷新选项</el-button>
    </PageHeader>

    <!-- 统一面板: 看板 + Tab(运行记录/定义列表) + 日志Drawer -->
    <TaskDefinitionPanel
      ref="panelRef"
      task-type="inference"
      :get-log-fn="getInferenceLog"
      @create="openFormDialog()"
      @edit="openFormDialog($event)"
      @duplicate="openFormDialog($event, true)"
      @executed="onExecuted"
    />

    <!-- 新建/编辑定义 Dialog -->
    <el-dialog v-model="dialogVisible" :title="editingDefId ? '编辑推理定义' : '新建推理定义'" width="700px" destroy-on-close>
      <el-form label-width="130px">
        <!-- ===== 基础设置 ===== -->
        <el-divider content-position="left">基础设置</el-divider>

        <el-form-item label="定义名称">
          <el-input v-model="defName" placeholder="推理-chatts-2026..." style="width: 100%" />
        </el-form-item>

        <el-form-item label="算法">
          <el-select v-model="form.algorithm" style="width: 100%" @change="onAlgorithmChange">
            <el-option v-for="a in algorithms" :key="a" :label="a" :value="a" />
          </el-select>
        </el-form-item>

        <el-form-item label="基础模型路径">
          <el-input v-model="form.baseModelPath" placeholder="自动根据算法填充" clearable />
        </el-form-item>

        <!-- LoRA 选择器 (仅 chatts/qwen) -->
        <template v-if="showLlmParams">
          <el-form-item label="LoRA 微调模型">
            <el-select v-model="form.loraRunPath" style="width: 100%" filterable clearable placeholder="无 (使用原始模型)" @change="onLoraRunChange">
              <el-option label="无 (使用原始模型)" value="" />
              <el-option v-for="m in loraModels" :key="m.path" :label="`${m.name} (${m.type})`" :value="m.path" />
            </el-select>
          </el-form-item>
          <el-form-item label="Checkpoint" v-if="form.loraRunPath">
            <el-select v-model="form.loraCheckpoint" style="width: 200px" clearable>
              <el-option label="最终模型" value="" />
              <el-option v-for="cp in checkpoints" :key="cp" :label="cp" :value="cp" />
            </el-select>
          </el-form-item>
        </template>

        <!-- 数据来源 -->
        <el-form-item label="数据来源">
          <el-radio-group v-model="dataSourceMode">
            <el-radio value="dataset">已有数据集</el-radio>
            <el-radio value="iotdb">从 IoTDB 采集后推理</el-radio>
          </el-radio-group>
        </el-form-item>

        <el-form-item v-if="dataSourceMode === 'dataset'" label="输入数据">
          <el-select v-model="selectedAssetPaths" style="width: 100%" multiple filterable clearable placeholder="选择采集数据文件">
            <el-option v-for="item in assetOptions" :key="item.path" :label="`${item.filename} | ${item.path}`" :value="item.path" />
          </el-select>
        </el-form-item>

        <template v-if="dataSourceMode === 'iotdb'">
          <el-form-item label="数据源">
            <el-select v-model="selectedSourceId" style="width: 100%" filterable placeholder="选择已配置的 IoTDB 数据源">
              <el-option v-for="s in iotdbSources" :key="s.id" :label="`${s.name} (${s.source_path})`" :value="s.id" />
            </el-select>
          </el-form-item>
          <el-form-item label="开始时间">
            <el-input v-model="iotdbStartTime" placeholder="2023-01-01 00:00:00（可选）" />
          </el-form-item>
          <el-form-item label="结束时间">
            <el-input v-model="iotdbEndTime" placeholder="2023-01-02 00:00:00（可选）" />
          </el-form-item>
        </template>

        <!-- ===== 公共推理参数 ===== -->
        <el-divider content-position="left">采样与阈值</el-divider>

        <el-form-item label="降采样点数">
          <el-input-number v-model="form.nDownsample" :min="100" :max="200000" :step="500" style="width: 200px" />
        </el-form-item>
        <el-form-item label="异常阈值">
          <el-input-number v-model="form.threshold" :min="0" :max="1" :step="0.05" :precision="2" style="width: 200px" />
        </el-form-item>
        <el-form-item label="降采样模式">
          <el-select v-model="form.downsampleMode" style="width: 200px">
            <el-option label="auto" value="auto" />
            <el-option label="fixed" value="fixed" />
            <el-option label="ratio" value="ratio" />
            <el-option label="off" value="off" />
          </el-select>
        </el-form-item>

        <!-- ===== LLM 参数 (仅 chatts/qwen) ===== -->
        <template v-if="showLlmParams">
          <el-divider content-position="left">LLM 参数 ({{ form.algorithm }})</el-divider>

          <el-form-item label="4-bit 量化">
            <el-select v-model="form.loadIn4bit" style="width: 200px">
              <el-option label="auto" value="auto" />
              <el-option label="true" value="true" />
              <el-option label="false" value="false" />
            </el-select>
          </el-form-item>
          <el-form-item label="max_new_tokens">
            <el-input-number v-model="form.maxNewTokens" :min="100" :max="8192" :step="100" style="width: 200px" />
          </el-form-item>
          <el-form-item label="设备">
            <el-select v-model="form.llmDevice" style="width: 200px">
              <el-option label="auto" value="auto" />
              <el-option label="cuda:0" value="cuda:0" />
              <el-option label="cuda:1" value="cuda:1" />
              <el-option label="cpu" value="cpu" />
            </el-select>
          </el-form-item>
          <el-form-item label="使用缓存">
            <el-select v-model="form.useCache" style="width: 200px">
              <el-option label="auto" value="auto" />
              <el-option label="true" value="true" />
              <el-option label="false" value="false" />
            </el-select>
          </el-form-item>
        </template>

        <!-- ===== Timer 参数 ===== -->
        <template v-if="form.algorithm === 'timer'">
          <el-divider content-position="left">Timer 参数</el-divider>
          <el-form-item label="设备">
            <el-input v-model="form.timerDevice" style="width: 200px" />
          </el-form-item>
          <el-form-item label="回溯长度">
            <el-input-number v-model="form.timerLookback" :min="8" :max="2048" :step="8" style="width: 200px" />
          </el-form-item>
          <el-form-item label="阈值系数 K">
            <el-input-number v-model="form.timerThresholdK" :min="0.1" :max="10" :step="0.5" :precision="1" style="width: 200px" />
          </el-form-item>
          <el-form-item label="检测方法">
            <el-select v-model="form.timerMethod" style="width: 200px">
              <el-option label="reconstruction" value="reconstruction" />
              <el-option label="anomaly_score" value="anomaly_score" />
            </el-select>
          </el-form-item>
          <el-form-item label="流式模式">
            <el-switch v-model="form.timerStreaming" />
          </el-form-item>
        </template>

        <!-- ===== ADTK 参数 ===== -->
        <template v-if="form.algorithm === 'adtk_hbos'">
          <el-divider content-position="left">ADTK HBOS 参数</el-divider>
          <el-form-item label="直方图区间数">
            <el-input-number v-model="form.adtkBinNums" :min="5" :max="500" :step="5" style="width: 200px" />
          </el-form-item>
          <el-form-item label="异常比例">
            <el-input-number v-model="form.adtkHbosRatio" :min="0.01" :max="0.5" :step="0.01" :precision="2" style="width: 200px" />
          </el-form-item>
        </template>
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
import { onMounted, reactive, ref, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { getApiData } from '../utils/format'
import PageHeader from '../components/PageHeader.vue'
import TaskDefinitionPanel from '../components/TaskDefinitionPanel.vue'
import { listIotdbSources, type IotdbSourceItem } from '../api/iotdbSource'

import { listRawDatasets, type RawDatasetFile } from '../api/data'
import {
  getInferenceLog,
  listInferenceAlgorithms,
  listTrainedModels,
  type InferenceAlgorithm,
} from '../api/inference'
import {
  createTaskCenterDefinition,
  createTaskCenterRun,
  type TaskCenterDefinitionRow,
} from '../api/taskCenter'

interface TrainedModel {
  name: string
  path: string
  type: string
  checkpoints?: string[]
}

const ALGORITHM_DEFAULT_MODELS: Record<string, string> = {
  chatts: '/home/share/llm_models/bytedance-research/ChatTS-8B',
  qwen: '/home/share/models/Qwen3-VL-8B-train-8192_base',
  timer: '/home/share/llm_models/thuml/timer-base-84m',
  adtk_hbos: '',
  ensemble: '',
}

const panelRef = ref<InstanceType<typeof TaskDefinitionPanel> | null>(null)
const submitting = ref(false)
const dialogVisible = ref(false)
const editingDefId = ref('')
const defName = ref('')

const algorithms = ref<string[]>(['chatts', 'qwen', 'adtk_hbos', 'timer', 'ensemble'])
const assetOptions = ref<RawDatasetFile[]>([])
const selectedAssetPaths = ref<string[]>([])
const loraModels = ref<TrainedModel[]>([])
const checkpoints = ref<string[]>([])

// IoTDB source
const dataSourceMode = ref<'dataset' | 'iotdb'>('dataset')
const iotdbSources = ref<IotdbSourceItem[]>([])
const selectedSourceId = ref('')
const iotdbStartTime = ref('')
const iotdbEndTime = ref('')

const form = reactive({
  algorithm: 'chatts',
  baseModelPath: ALGORITHM_DEFAULT_MODELS['chatts'],
  loraRunPath: '',
  loraCheckpoint: '',
  nDownsample: 5000,
  threshold: 0.1,
  downsampleMode: 'auto',
  loadIn4bit: 'auto',
  maxNewTokens: 1200,
  llmDevice: 'auto',
  useCache: 'auto',
  timerDevice: 'cuda:0',
  timerLookback: 96,
  timerThresholdK: 3.0,
  timerMethod: 'reconstruction',
  timerStreaming: false,
  adtkBinNums: 50,
  adtkHbosRatio: 0.1,
})

const showLlmParams = computed(() => form.algorithm === 'chatts' || form.algorithm === 'qwen')

// ==================== Helpers ====================

function onAlgorithmChange(): void {
  form.baseModelPath = ALGORITHM_DEFAULT_MODELS[form.algorithm] || ''
  form.loraRunPath = ''
  form.loraCheckpoint = ''
  checkpoints.value = []
  if (showLlmParams.value) loadLoraModels()
}

async function loadLoraModels(): Promise<void> {
  const family = form.algorithm === 'qwen' ? 'qwen' : 'chatts'
  const resp = await listTrainedModels(family)
  const data = getApiData<{ models?: TrainedModel[] }>(resp)
  loraModels.value = Array.isArray(data.models) ? data.models : []
}

function onLoraRunChange(): void {
  form.loraCheckpoint = ''
  if (!form.loraRunPath) { checkpoints.value = []; return }
  const found = loraModels.value.find((m: TrainedModel) => m.path === form.loraRunPath)
  checkpoints.value = found?.checkpoints || []
}

function buildInferenceParams(): Record<string, unknown> {
  const params: Record<string, unknown> = {
    n_downsample: form.nDownsample,
    threshold: form.threshold,
    downsample_mode: form.downsampleMode,
    base_model_path: form.baseModelPath,
  }
  if (form.loraRunPath) {
    let adapter = form.loraRunPath
    if (form.loraCheckpoint) adapter = `${form.loraRunPath}/${form.loraCheckpoint}`
    params.lora_adapter_path = adapter
  }
  if (showLlmParams.value) {
    params.chatts_load_in_4bit = form.loadIn4bit
    params.chatts_max_new_tokens = form.maxNewTokens
    params.qwen_max_new_tokens = form.maxNewTokens
    params.chatts_device = form.llmDevice
    params.chatts_use_cache = form.useCache
  }
  if (form.algorithm === 'timer') {
    params.timer_device = form.timerDevice
    params.timer_lookback_length = form.timerLookback
    params.timer_threshold_k = form.timerThresholdK
    params.timer_method = form.timerMethod
    params.timer_streaming = form.timerStreaming
  }
  if (form.algorithm === 'adtk_hbos') {
    params.bin_nums = form.adtkBinNums
    params.hbos_ratio = form.adtkHbosRatio
  }
  return params
}

async function refreshOptions(): Promise<void> {
  const [algoResp, datasetResp, sourcesResp] = await Promise.all([
    listInferenceAlgorithms(), listRawDatasets(), listIotdbSources(),
  ])
  const algoData = getApiData<{ algorithms?: InferenceAlgorithm[] }>(algoResp)
  const algoList = Array.isArray(algoData.algorithms) ? algoData.algorithms : []
  if (algoList.length > 0) algorithms.value = algoList.map((a: InferenceAlgorithm) => a.id || a.name)
  const datasetData = getApiData<{ datasets?: RawDatasetFile[] }>(datasetResp)
  assetOptions.value = Array.isArray(datasetData.datasets) ? datasetData.datasets : []
  const sourcesData = getApiData<{ sources?: IotdbSourceItem[] }>(sourcesResp)
  iotdbSources.value = Array.isArray(sourcesData.sources) ? sourcesData.sources : []
  if (showLlmParams.value) await loadLoraModels()
}

function resetForm(): void {
  form.algorithm = 'chatts'
  form.baseModelPath = ALGORITHM_DEFAULT_MODELS['chatts']
  form.loraRunPath = ''
  form.loraCheckpoint = ''
  selectedAssetPaths.value = []
  form.nDownsample = 5000
  form.threshold = 0.1
  form.downsampleMode = 'auto'
  form.loadIn4bit = 'auto'
  form.maxNewTokens = 1200
  form.llmDevice = 'auto'
  form.useCache = 'auto'
  dataSourceMode.value = 'dataset'
  selectedSourceId.value = ''
  iotdbStartTime.value = ''
  iotdbEndTime.value = ''
}

// ==================== Dialog ====================

function openFormDialog(row?: TaskCenterDefinitionRow, duplicate = false): void {
  if (row) {
    editingDefId.value = duplicate ? '' : row.id
    defName.value = duplicate ? `${row.name} (副本)` : row.name
    const cfg = row.config || {}
    form.algorithm = String(cfg.algorithm || 'chatts')
    form.baseModelPath = String(cfg.base_model_path || ALGORITHM_DEFAULT_MODELS[form.algorithm] || '')
    form.loraRunPath = String(cfg.lora_adapter_path || '')
    form.nDownsample = Number(cfg.n_downsample || 5000)
    form.threshold = Number(cfg.threshold || 0.1)
    form.downsampleMode = String(cfg.downsample_mode || 'auto')
    if (cfg.input_files && Array.isArray(cfg.input_files)) {
      dataSourceMode.value = 'dataset'
      selectedAssetPaths.value = cfg.input_files as string[]
    }
    if (cfg.iotdb_source_id) {
      dataSourceMode.value = 'iotdb'
      selectedSourceId.value = String(cfg.iotdb_source_id)
    }
  } else {
    editingDefId.value = ''
    defName.value = `推理-${form.algorithm}-${new Date().toISOString().substring(0, 16)}`
    resetForm()
  }
  dialogVisible.value = true
}

function buildDefinitionConfig(): Record<string, unknown> {
  const config: Record<string, unknown> = { algorithm: form.algorithm, ...buildInferenceParams() }
  if (dataSourceMode.value === 'dataset') {
    config.input_files = selectedAssetPaths.value
  } else {
    config.iotdb_source_id = selectedSourceId.value
    const source = iotdbSources.value.find(s => s.id === selectedSourceId.value)
    if (source) config.iotdb_source_name = source.name
    if (iotdbStartTime.value.trim()) config.start_time = iotdbStartTime.value.trim()
    if (iotdbEndTime.value.trim()) config.end_time = iotdbEndTime.value.trim()
  }
  return config
}

async function saveDefinition(): Promise<void> {
  submitting.value = true
  try {
    await createTaskCenterDefinition({
      name: defName.value || `推理-${form.algorithm}`,
      task_type: 'inference',
      trigger_mode: 'manual',
      config: buildDefinitionConfig(),
    })
    ElMessage.success('定义已保存')
    dialogVisible.value = false
    panelRef.value?.refreshAll()
  } finally {
    submitting.value = false
  }
}

async function saveAndExecute(): Promise<void> {
  submitting.value = true
  try {
    const config = buildDefinitionConfig()
    const defResp = await createTaskCenterDefinition({
      name: defName.value || `推理-${form.algorithm}`,
      task_type: 'inference',
      trigger_mode: 'manual',
      config,
    })
    const defData = getApiData<{ definition?: { id?: string } }>(defResp)

    const runResp = await createTaskCenterRun({
      definition_id: defData.definition?.id,
      task_type: 'inference',
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
  // Panel already refreshed internally
}

onMounted(() => { refreshOptions() })
</script>

<style scoped>
.inference-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
</style>
