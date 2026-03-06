<template>
  <div class="annotation-center-page">
    <el-row :gutter="12">
      <el-col :xs="24" :xl="14">
        <el-card shadow="never">
          <template #header>
            <div class="card-header">
              <span>标注点位总览</span>
              <div class="header-actions">
                <el-select v-model="annotationKind" style="width: 120px">
                  <el-option label="全部来源" value="all" />
                  <el-option label="AUTO" value="auto" />
                  <el-option label="HUMAN" value="human" />
                </el-select>
                <el-select v-model="statusFilter" style="width: 140px">
                  <el-option label="全部状态" value="" />
                  <el-option label="未审核" value="unreviewed" />
                  <el-option label="待审" value="pending" />
                  <el-option label="已通过" value="approved" />
                  <el-option label="需修订" value="needs_fix" />
                </el-select>
                <el-input v-model="keyword" placeholder="按 point_id 检索" clearable style="width: 220px" />
                <el-button type="primary" @click="loadPoints">查询</el-button>
              </div>
            </div>
          </template>

          <el-table
            v-loading="loadingPoints"
            :data="pointRows"
            border
            stripe
            height="320"
            @selection-change="onSelectionChange"
            @row-click="onPointRowClick"
          >
            <el-table-column type="selection" width="45" />
            <el-table-column prop="point_id" label="Point ID" min-width="240" show-overflow-tooltip />
            <el-table-column prop="source_kind" label="来源" width="90" />
            <el-table-column prop="annotation_count" label="标注数" width="90" />
            <el-table-column prop="segment_count" label="段数" width="90" />
            <el-table-column label="分数" width="120">
              <template #default="{ row }">
                {{ row.score == null ? '-' : Number(row.score).toFixed(3) }}
              </template>
            </el-table-column>
            <el-table-column prop="status" label="审核状态" width="110" />
            <el-table-column prop="reviewer" label="审核人" width="110" show-overflow-tooltip />
            <el-table-column label="更新时间" width="180">
              <template #default="{ row }">{{ formatTime(row.updated_at) }}</template>
            </el-table-column>
          </el-table>

          <div class="review-actions">
            <el-input v-model="reviewerInput" placeholder="审核人（可选）" style="width: 180px" />
            <el-button type="success" :disabled="selectedPointIds.length === 0" @click="batchUpdateStatus('approved')">通过</el-button>
            <el-button type="warning" :disabled="selectedPointIds.length === 0" @click="batchUpdateStatus('needs_fix')">需修订</el-button>
            <el-button :disabled="selectedPointIds.length === 0" @click="batchUpdateStatus('pending')">重新审核</el-button>
            <el-tag type="info">已选 {{ selectedPointIds.length }} 项</el-tag>
            <el-tag type="success">通过 {{ stats.approved }}</el-tag>
            <el-tag type="warning">需修订 {{ stats.needs_fix }}</el-tag>
            <el-tag>待审 {{ stats.pending }}</el-tag>
            <el-tag type="info">未审核 {{ stats.unreviewed }}</el-tag>
          </div>

          <el-divider content-position="left">点位详情</el-divider>
          <el-descriptions :column="2" border v-if="pointDetail">
            <el-descriptions-item label="Point ID">{{ pointDetail.point_id }}</el-descriptions-item>
            <el-descriptions-item label="标注记录">{{ pointDetail.annotation_summary.count }}</el-descriptions-item>
            <el-descriptions-item label="最近标注来源">{{ pointDetail.annotation_summary.latest_source_kind || '-' }}</el-descriptions-item>
            <el-descriptions-item label="最近审核状态">{{ pointDetail.review_summary.latest_status || '-' }}</el-descriptions-item>
            <el-descriptions-item label="最近推理算法">{{ pointDetail.inference_summary.latest_method || '-' }}</el-descriptions-item>
            <el-descriptions-item label="最近推理分数">
              {{ pointDetail.inference_summary.latest_score_avg == null ? '-' : Number(pointDetail.inference_summary.latest_score_avg).toFixed(3) }}
            </el-descriptions-item>
          </el-descriptions>
          <el-empty v-else description="点击上方点位查看详情" :image-size="70" />

          <el-divider content-position="left">点位时间线</el-divider>
          <el-table :data="timelineRows" border stripe height="260" v-loading="loadingTimeline">
            <el-table-column prop="type" label="类型" width="110" />
            <el-table-column label="时间" width="180">
              <template #default="{ row }">{{ formatTime(row.time) }}</template>
            </el-table-column>
            <el-table-column label="数据" min-width="320" show-overflow-tooltip>
              <template #default="{ row }">{{ jsonCompact(row.data) }}</template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>

      <el-col :xs="24" :xl="10">
        <el-card shadow="never">
          <template #header>
            <div class="card-header">
              <span>标注工具工作台</span>
              <div class="header-actions">
                <el-button @click="reloadFrame">刷新</el-button>
                <el-button type="primary" @click="openInNewTab">新窗口</el-button>
              </div>
            </div>
          </template>
          <iframe :key="frameKey" :src="annotatorUrl" class="annotator-frame" title="Annotator" />
        </el-card>

        <el-card shadow="never" class="files-card">
          <template #header>
            <div class="card-header">
              <span>可标注文件</span>
              <el-button @click="loadAnnotatableFiles">刷新</el-button>
            </div>
          </template>
          <el-table :data="annotatableFiles" border stripe height="220" v-loading="loadingFiles">
            <el-table-column prop="name" label="文件名" min-width="280" show-overflow-tooltip />
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { formatTime, jsonCompact, getApiData } from '../utils/format'

import { listAnnotatableFiles } from '../api/annotation'
import { batchUpdateReviewQueue, listReviewQueue, type ReviewQueueItem, type ReviewQueueStats } from '../api/review'
import {
  getPointDetail,
  getPointTimeline,
  type PointSummaryResponse,
  type PointTimelineEvent,
} from '../api/points'

const keyword = ref('')
const statusFilter = ref('')
const annotationKind = ref('all')
const reviewerInput = ref('')
const pointRows = ref<ReviewQueueItem[]>([])
const loadingPoints = ref(false)
const loadingTimeline = ref(false)
const loadingFiles = ref(false)
const pointDetail = ref<PointSummaryResponse | null>(null)
const timelineRows = ref<PointTimelineEvent[]>([])
const annotatableFiles = ref<Array<{ name: string }>>([])
const selectedPointIds = ref<string[]>([])
const stats = ref<ReviewQueueStats>({
  total: 0,
  pending: 0,
  approved: 0,
  needs_fix: 0,
  unreviewed: 0,
})

const frameKey = ref(0)
const annotatorUrl = computed(() => {
  const custom = String(import.meta.env.VITE_ANNOTATOR_URL || '').trim()
  if (custom) return custom
  const host = window.location.hostname || '127.0.0.1'
  return `http://${host}:5000`
})


function toFileName(item: unknown): string {
  if (typeof item === 'string') {
    return item.trim()
  }
  if (!item || typeof item !== 'object') {
    return ''
  }
  const row = item as Record<string, unknown>
  const name = String(row.name || row.filename || row.path || '').trim()
  return name
}

async function loadPoints(): Promise<void> {
  loadingPoints.value = true
  try {
    const response = await listReviewQueue({
      status: statusFilter.value || undefined,
      annotation_kind: annotationKind.value || 'all',
      keyword: keyword.value.trim() || undefined,
      limit: 300,
      offset: 0,
    })
    const data = getApiData<{ items?: ReviewQueueItem[]; stats?: ReviewQueueStats }>(response)
    pointRows.value = Array.isArray(data.items) ? data.items : []
    stats.value = {
      total: Number(data.stats?.total || 0),
      pending: Number(data.stats?.pending || 0),
      approved: Number(data.stats?.approved || 0),
      needs_fix: Number(data.stats?.needs_fix || 0),
      unreviewed: Number(data.stats?.unreviewed || 0),
    }
    selectedPointIds.value = []
  } finally {
    loadingPoints.value = false
  }
}

function onSelectionChange(rows: ReviewQueueItem[]): void {
  selectedPointIds.value = rows
    .map((row) => String(row.point_id || '').trim())
    .filter((pointId) => pointId.length > 0)
}

async function batchUpdateStatus(status: 'pending' | 'approved' | 'needs_fix'): Promise<void> {
  if (selectedPointIds.value.length === 0) {
    ElMessage.warning('请先选择至少一项')
    return
  }
  await batchUpdateReviewQueue({
    point_ids: selectedPointIds.value,
    status,
    reviewer: reviewerInput.value.trim() || undefined,
  })
  ElMessage.success(`已更新 ${selectedPointIds.value.length} 项为 ${status}`)
  await loadPoints()
}

async function onPointRowClick(row: ReviewQueueItem): Promise<void> {
  const pointId = String(row.point_id || '').trim()
  if (!pointId) return

  loadingTimeline.value = true
  try {
    const [detailResp, timelineResp] = await Promise.all([
      getPointDetail(pointId),
      getPointTimeline(pointId, 100),
    ])
    const detailData = getApiData<PointSummaryResponse>(detailResp)
    const timelineData = getApiData<{ events?: PointTimelineEvent[] }>(timelineResp)
    pointDetail.value = detailData
    timelineRows.value = Array.isArray(timelineData.events) ? timelineData.events : []
  } finally {
    loadingTimeline.value = false
  }
}

async function loadAnnotatableFiles(): Promise<void> {
  loadingFiles.value = true
  try {
    const response = await listAnnotatableFiles()
    const data = getApiData<Record<string, unknown>>(response)
    let files: string[] = []
    const asAny = data as { files?: unknown; data?: unknown }

    if (Array.isArray(asAny.files)) {
      files = asAny.files.map((item) => toFileName(item)).filter((item) => item.length > 0)
    } else if (Array.isArray(asAny.data)) {
      files = asAny.data.map((item) => toFileName(item)).filter((item) => item.length > 0)
    } else {
      files = Object.keys(data || {})
    }
    annotatableFiles.value = files.slice(0, 200).map((name) => ({ name }))
  } catch {
    annotatableFiles.value = []
    ElMessage.warning('获取可标注文件列表失败')
  } finally {
    loadingFiles.value = false
  }
}

function reloadFrame(): void {
  frameKey.value += 1
}

function openInNewTab(): void {
  window.open(annotatorUrl.value, '_blank', 'noopener,noreferrer')
}


onMounted(() => {
  Promise.all([loadPoints(), loadAnnotatableFiles()]).catch(() => {
    ElMessage.error('初始化标注中心失败')
  })
})
</script>

<style scoped>
.annotation-center-page {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.hint-alert {
  margin-top: 12px;
}

.review-actions {
  margin-top: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.annotator-frame {
  width: 100%;
  height: 420px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  background: var(--bg-dark);
}

.files-card {
  margin-top: 12px;
}
</style>
