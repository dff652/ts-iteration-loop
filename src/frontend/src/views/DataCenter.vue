<template>
  <div class="datasets-page">
    <PageHeader title="数据集" subtitle="管理已采集的时序数据文件">
      <el-button @click="loadDatasets">刷新列表</el-button>
    </PageHeader>

    <el-card shadow="never">
      <div class="toolbar">
        <el-input
          v-model="searchKeyword"
          placeholder="搜索文件名"
          clearable
          style="width: 260px"
          @keyup.enter="loadDatasets"
        />
        <el-button type="primary" @click="loadDatasets">查询</el-button>
      </div>

      <el-table
        v-loading="loadingDatasets"
        :data="filteredDatasets"
        border
        stripe
        style="width: 100%"
        @row-click="onDatasetRowClick"
      >
        <el-table-column prop="filename" label="文件名" min-width="220" show-overflow-tooltip />
        <el-table-column prop="path" label="路径" min-width="280" show-overflow-tooltip />
        <el-table-column label="修改时间" width="180">
          <template #default="{ row }">{{ formatTime(row.modified_time ? (row.modified_time > 1e12 ? new Date(row.modified_time).toISOString() : new Date(row.modified_time * 1000).toISOString()) : null) }}</template>
        </el-table-column>
        <el-table-column label="大小" width="120">
          <template #default="{ row }">{{ formatSize(row.size_bytes) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="120" align="center">
          <template #default="{ row }">
            <el-button type="primary" link @click.stop="goToDetail(row)">详情</el-button>
            <el-button type="info" link @click.stop="openPreviewDrawer(row)">预览</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 数据预览抽屉 -->
    <el-drawer v-model="previewDrawerVisible" :title="`数据预览 - ${selectedFilename}`" size="60%">
      <div class="preview-toolbar">
        <span>预览行数：</span>
        <el-input-number v-model="previewLimit" :min="10" :max="5000" :step="10" style="width: 140px" />
        <el-button type="primary" @click="loadPreview">加载</el-button>
      </div>
      <el-table v-loading="loadingPreview" :data="previewRows" border stripe height="500" style="margin-top: 12px">
        <el-table-column
          v-for="col in previewColumns"
          :key="col"
          :prop="col"
          :label="col"
          min-width="130"
          show-overflow-tooltip
        />
      </el-table>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { formatTime, formatSize, getApiData } from '../utils/format'
import PageHeader from '../components/PageHeader.vue'

import {
  listRawDatasets,
  previewDatasetFile,
  type RawDatasetFile,
} from '../api/data'

const loadingDatasets = ref(false)
const loadingPreview = ref(false)
const router = useRouter()
const datasets = ref<RawDatasetFile[]>([])
const searchKeyword = ref('')
const previewRows = ref<Array<Record<string, unknown>>>([])
const previewColumns = ref<string[]>([])
const previewLimit = ref(100)
const selectedFilename = ref('')
const previewDrawerVisible = ref(false)

const filteredDatasets = computed(() => {
  if (!searchKeyword.value.trim()) return datasets.value
  const kw = searchKeyword.value.trim().toLowerCase()
  return datasets.value.filter((d) => (d.filename || '').toLowerCase().includes(kw))
})

async function loadDatasets(): Promise<void> {
  loadingDatasets.value = true
  try {
    const response = await listRawDatasets()
    const data = getApiData<{ datasets?: RawDatasetFile[] }>(response)
    datasets.value = Array.isArray(data.datasets) ? data.datasets : []
  } finally {
    loadingDatasets.value = false
  }
}

function onDatasetRowClick(row: RawDatasetFile): void {
  goToDetail(row)
}

function goToDetail(row: RawDatasetFile): void {
  router.push({ path: '/data/datasets/detail', query: { file: row.filename } })
}

function openPreviewDrawer(row: RawDatasetFile): void {
  selectedFilename.value = String(row.filename || '')
  previewDrawerVisible.value = true
  loadPreview()
}

async function loadPreview(): Promise<void> {
  if (!selectedFilename.value.trim()) return
  loadingPreview.value = true
  try {
    const response = await previewDatasetFile(selectedFilename.value.trim(), Number(previewLimit.value) || 100)
    const data = getApiData<{ preview?: Array<Record<string, unknown>> }>(response)
    const rows = Array.isArray(data.preview) ? data.preview : []
    previewRows.value = rows
    const firstRow = rows[0] ?? {}
    previewColumns.value = rows.length > 0 ? Object.keys(firstRow) : []
  } finally {
    loadingPreview.value = false
  }
}

onMounted(() => {
  loadDatasets().catch(() => {
    ElMessage.error('加载数据列表失败')
  })
})
</script>

<style scoped>
.datasets-page {
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

.preview-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
