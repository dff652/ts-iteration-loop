<template>
  <div class="dataset-detail-page">
    <PageHeader :title="`数据集: ${filename}`" subtitle="查看数据集的统计信息和样本数据">
      <el-button @click="router.back()">← 返回列表</el-button>
      <el-button type="primary" @click="loadAll">刷新</el-button>
    </PageHeader>

    <!-- 统计卡片 -->
    <el-row :gutter="12" class="stats-row">
      <el-col :xs="12" :sm="6">
        <StatCard label="总行数" :value="totalRows" icon="📊" color="primary" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <StatCard label="列数" :value="columnCount" icon="📋" color="success" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <StatCard label="文件大小" :value="fileSizeDisplay" icon="💾" />
      </el-col>
      <el-col :xs="12" :sm="6">
        <StatCard label="时间范围" :value="timeRangeDisplay" icon="🕐" />
      </el-col>
    </el-row>

    <!-- 列信息 -->
    <el-card shadow="never">
      <template #header>
        <div class="section-header">
          <span>列信息分析</span>
          <el-tag type="info">{{ columnCount }} 列</el-tag>
        </div>
      </template>
      <el-table :data="columnStats" border stripe v-loading="loading">
        <el-table-column prop="name" label="列名" min-width="160" show-overflow-tooltip />
        <el-table-column prop="type" label="推断类型" width="120">
          <template #default="{ row }">
            <el-tag :type="row.type === 'number' ? 'primary' : row.type === 'datetime' ? 'warning' : 'info'" size="small">
              {{ row.type }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="nonNull" label="非空数" width="100" />
        <el-table-column prop="nullCount" label="空值数" width="100" />
        <el-table-column prop="unique" label="唯一值" width="100" />
        <el-table-column prop="min" label="最小值" width="140" show-overflow-tooltip />
        <el-table-column prop="max" label="最大值" width="140" show-overflow-tooltip />
        <el-table-column prop="sample" label="样本值" min-width="180" show-overflow-tooltip />
      </el-table>
    </el-card>

    <!-- 数据预览 -->
    <el-card shadow="never">
      <template #header>
        <div class="section-header">
          <span>数据预览</span>
          <div class="preview-controls">
            <span>行数：</span>
            <el-input-number v-model="previewLimit" :min="10" :max="5000" :step="50" style="width: 140px" />
            <el-button type="primary" @click="loadPreview">加载</el-button>
          </div>
        </div>
      </template>
      <el-table :data="previewRows" border stripe height="420" v-loading="loadingPreview">
        <el-table-column type="index" label="#" width="60" />
        <el-table-column
          v-for="col in previewColumns"
          :key="col"
          :prop="col"
          :label="col"
          min-width="130"
          show-overflow-tooltip
        />
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { formatSize, getApiData } from '../utils/format'
import PageHeader from '../components/PageHeader.vue'
import StatCard from '../components/StatCard.vue'
import { previewDatasetFile, listRawDatasets, type RawDatasetFile } from '../api/data'

interface ColumnStat {
  name: string
  type: string
  nonNull: number
  nullCount: number
  unique: number
  min: string
  max: string
  sample: string
}

const route = useRoute()
const router = useRouter()

const filename = computed(() => String(route.query.file || ''))
const loading = ref(false)
const loadingPreview = ref(false)
const previewLimit = ref(500)
const previewRows = ref<Array<Record<string, unknown>>>([])
const previewColumns = ref<string[]>([])
const columnStats = ref<ColumnStat[]>([])
const fileInfo = ref<RawDatasetFile | null>(null)

const totalRows = computed(() => previewRows.value.length >= previewLimit.value ? `${previewLimit.value}+` : String(previewRows.value.length))
const columnCount = computed(() => previewColumns.value.length)
const fileSizeDisplay = computed(() => fileInfo.value ? formatSize(fileInfo.value.size_bytes) : '-')

const timeRangeDisplay = computed(() => {
  // Try to detect time column and show range
  const timeCol = previewColumns.value.find(c =>
    /time|date|timestamp|日期/i.test(c)
  )
  if (!timeCol || previewRows.value.length === 0) return '-'
  const values = previewRows.value
    .map(r => String(r[timeCol] || ''))
    .filter(v => v && v !== 'null' && v !== 'None')
  if (values.length < 2) return '-'
  const first = values[0]!
  const last = values[values.length - 1]!
  // Shorten display
  const shorten = (s: string) => s.length > 16 ? s.substring(0, 16) : s
  return `${shorten(first)} ~ ${shorten(last)}`
})

function inferType(values: unknown[]): string {
  const nonNull = values.filter(v => v !== null && v !== undefined && String(v) !== '' && String(v) !== 'null')
  if (nonNull.length === 0) return 'empty'

  // Check if datetime
  const sample = String(nonNull[0])
  if (/^\d{4}[-/]\d{2}[-/]\d{2}/.test(sample)) return 'datetime'
  if (/^\d{13,}$/.test(sample)) return 'timestamp'

  // Check if number
  const numericCount = nonNull.filter(v => !isNaN(Number(v))).length
  if (numericCount > nonNull.length * 0.8) return 'number'

  return 'string'
}

function analyzeColumns(rows: Array<Record<string, unknown>>, columns: string[]): ColumnStat[] {
  return columns.map(col => {
    const values = rows.map(r => r[col])
    const nonNull = values.filter(v => v !== null && v !== undefined && String(v) !== '' && String(v) !== 'null')
    const uniqueSet = new Set(nonNull.map(v => String(v)))
    const type = inferType(values)

    let min = '-'
    let max = '-'
    if (type === 'number' && nonNull.length > 0) {
      const nums = nonNull.map(v => Number(v)).filter(n => !isNaN(n))
      if (nums.length > 0) {
        min = Math.min(...nums).toFixed(4)
        max = Math.max(...nums).toFixed(4)
      }
    } else if (nonNull.length > 0) {
      const sorted = [...nonNull].map(String).sort()
      min = sorted[0]!.substring(0, 20)
      max = sorted[sorted.length - 1]!.substring(0, 20)
    }

    return {
      name: col,
      type,
      nonNull: nonNull.length,
      nullCount: values.length - nonNull.length,
      unique: uniqueSet.size,
      min,
      max,
      sample: nonNull.length > 0 ? String(nonNull[0]).substring(0, 40) : '-',
    }
  })
}

async function loadPreview(): Promise<void> {
  if (!filename.value) return
  loadingPreview.value = true
  try {
    const response = await previewDatasetFile(filename.value, previewLimit.value)
    const data = getApiData<{ preview?: Array<Record<string, unknown>> }>(response)
    const rows = Array.isArray(data.preview) ? data.preview : []
    previewRows.value = rows
    const firstRow = rows[0] ?? {}
    previewColumns.value = rows.length > 0 ? Object.keys(firstRow) : []
    columnStats.value = analyzeColumns(rows, previewColumns.value)
  } finally {
    loadingPreview.value = false
  }
}

async function loadFileInfo(): Promise<void> {
  try {
    const response = await listRawDatasets()
    const data = getApiData<{ datasets?: RawDatasetFile[] }>(response)
    const list = Array.isArray(data.datasets) ? data.datasets : []
    fileInfo.value = list.find(d => d.filename === filename.value) || null
  } catch {
    // ignore
  }
}

async function loadAll(): Promise<void> {
  loading.value = true
  try {
    await Promise.all([loadPreview(), loadFileInfo()])
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  if (!filename.value) {
    ElMessage.warning('未指定数据集文件')
    router.replace('/data/datasets')
    return
  }
  loadAll()
})
</script>

<style scoped>
.dataset-detail-page {
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
  font-family: var(--font-display);
  font-weight: 600;
}

.preview-controls {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 400;
  font-size: 13px;
}
</style>
