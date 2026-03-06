<template>
  <div class="annotation-review-page">
    <PageHeader title="审核管理" subtitle="管理标注点位的审核状态和进度">
      <el-button type="primary" @click="loadPoints">查询</el-button>
    </PageHeader>

    <!-- 进度看板 -->
    <el-card shadow="never" class="progress-card">
      <template #header>
        <div class="section-header">
          <span>标注进度看板</span>
          <el-tag type="info">共 {{ stats.total }} 个点位</el-tag>
        </div>
      </template>
      <el-row :gutter="16">
        <el-col :xs="24" :md="12">
          <el-row :gutter="12">
            <el-col :span="12">
              <StatCard label="已通过" :value="stats.approved" icon="✅" color="success" />
            </el-col>
            <el-col :span="12">
              <StatCard label="需修订" :value="stats.needs_fix" icon="⚠️" color="warning" />
            </el-col>
            <el-col :span="12" style="margin-top: 12px">
              <StatCard label="待审核" :value="stats.pending" icon="⏳" color="primary" />
            </el-col>
            <el-col :span="12" style="margin-top: 12px">
              <StatCard label="未审核" :value="stats.unreviewed" icon="📋" />
            </el-col>
          </el-row>
        </el-col>
        <el-col :xs="24" :md="12">
          <div class="progress-bars">
            <div class="progress-item">
              <div class="progress-label">
                <span>审核完成率</span>
                <span class="progress-value">{{ completionRate }}%</span>
              </div>
              <el-progress :percentage="completionRate" :color="completionRate >= 80 ? '#22c55e' : completionRate >= 50 ? '#f59e0b' : '#ef4444'" :show-text="false" :stroke-width="12" />
            </div>
            <div class="progress-item">
              <div class="progress-label">
                <span>通过率 (已审核中)</span>
                <span class="progress-value">{{ approvalRate }}%</span>
              </div>
              <el-progress :percentage="approvalRate" :color="'#22c55e'" :show-text="false" :stroke-width="12" />
            </div>
            <div class="progress-item">
              <div class="progress-label">
                <span>需修订率 (已审核中)</span>
                <span class="progress-value">{{ fixRate }}%</span>
              </div>
              <el-progress :percentage="fixRate" :color="'#f59e0b'" :show-text="false" :stroke-width="12" />
            </div>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <!-- 筛选与数据表 -->
    <el-card shadow="never">
      <div class="toolbar">
        <el-select v-model="annotationKind" style="width: 120px" @change="resetAndLoad">
          <el-option label="全部来源" value="all" />
          <el-option label="AUTO" value="auto" />
          <el-option label="HUMAN" value="human" />
        </el-select>
        <el-select v-model="statusFilter" style="width: 140px" @change="resetAndLoad">
          <el-option label="全部状态" value="" />
          <el-option label="未审核" value="unreviewed" />
          <el-option label="待审" value="pending" />
          <el-option label="已通过" value="approved" />
          <el-option label="需修订" value="needs_fix" />
        </el-select>
        <el-input v-model="keyword" placeholder="按 point_id 检索" clearable style="width: 220px" @keyup.enter="resetAndLoad" />
        <el-button type="primary" @click="resetAndLoad">查询</el-button>
      </div>

      <el-table
        v-loading="loadingPoints"
        :data="pointRows"
        border
        stripe
        height="380"
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
        <el-table-column prop="status" label="审核状态" width="110">
          <template #default="{ row }">
            <el-tag :type="statusTagType(row.status)" size="small">{{ row.status || '未审核' }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="reviewer" label="审核人" width="110" show-overflow-tooltip />
        <el-table-column label="更新时间" width="180">
          <template #default="{ row }">{{ formatTime(row.updated_at) }}</template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pager-wrap">
        <el-pagination
          layout="total, sizes, prev, pager, next, jumper"
          :current-page="currentPage"
          :page-size="pageSize"
          :page-sizes="[20, 50, 100, 200]"
          :total="stats.total"
          @current-change="onPageChange"
          @size-change="onPageSizeChange"
        />
      </div>

      <!-- 批量操作 -->
      <div class="review-actions">
        <el-input v-model="reviewerInput" placeholder="审核人（可选）" style="width: 180px" />
        <el-button type="success" :disabled="selectedPointIds.length === 0" @click="batchUpdateStatus('approved')">通过</el-button>
        <el-button type="warning" :disabled="selectedPointIds.length === 0" @click="batchUpdateStatus('needs_fix')">需修订</el-button>
        <el-button :disabled="selectedPointIds.length === 0" @click="batchUpdateStatus('pending')">重新审核</el-button>
        <el-tag type="info">已选 {{ selectedPointIds.length }} 项</el-tag>
      </div>
    </el-card>

    <!-- 点位详情 + 时间线 -->
    <el-drawer v-model="detailDrawerVisible" :title="`点位详情: ${selectedPointId}`" size="50%">
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

      <el-divider content-position="left">时间线</el-divider>
      <el-table :data="timelineRows" border stripe height="300" v-loading="loadingTimeline">
        <el-table-column prop="type" label="类型" width="110">
          <template #default="{ row }">
            <el-tag :type="row.type === 'annotation' ? 'primary' : row.type === 'inference' ? 'warning' : 'success'" size="small">{{ row.type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="时间" width="180">
          <template #default="{ row }">{{ formatTime(row.time) }}</template>
        </el-table-column>
        <el-table-column label="数据" min-width="320" show-overflow-tooltip>
          <template #default="{ row }">{{ jsonCompact(row.data) }}</template>
        </el-table-column>
      </el-table>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { formatTime, jsonCompact, statusTagType, getApiData } from '../utils/format'
import PageHeader from '../components/PageHeader.vue'
import StatCard from '../components/StatCard.vue'

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
const pointDetail = ref<PointSummaryResponse | null>(null)
const timelineRows = ref<PointTimelineEvent[]>([])
const selectedPointIds = ref<string[]>([])
const selectedPointId = ref('')
const detailDrawerVisible = ref(false)
const currentPage = ref(1)
const pageSize = ref(50)

const stats = ref<ReviewQueueStats>({
  total: 0,
  pending: 0,
  approved: 0,
  needs_fix: 0,
  unreviewed: 0,
})

// 进度计算
const reviewed = computed(() => stats.value.approved + stats.value.needs_fix + stats.value.pending)
const completionRate = computed(() => {
  if (stats.value.total === 0) return 0
  return Math.round((reviewed.value / stats.value.total) * 100)
})
const approvalRate = computed(() => {
  if (reviewed.value === 0) return 0
  return Math.round((stats.value.approved / reviewed.value) * 100)
})
const fixRate = computed(() => {
  if (reviewed.value === 0) return 0
  return Math.round((stats.value.needs_fix / reviewed.value) * 100)
})

function resetAndLoad(): void {
  currentPage.value = 1
  loadPoints()
}

function onPageChange(page: number): void {
  currentPage.value = page
  loadPoints()
}

function onPageSizeChange(size: number): void {
  pageSize.value = size
  currentPage.value = 1
  loadPoints()
}

async function loadPoints(): Promise<void> {
  loadingPoints.value = true
  try {
    const offset = (currentPage.value - 1) * pageSize.value
    const response = await listReviewQueue({
      status: statusFilter.value || undefined,
      annotation_kind: annotationKind.value || 'all',
      keyword: keyword.value.trim() || undefined,
      limit: pageSize.value,
      offset,
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

  selectedPointId.value = pointId
  detailDrawerVisible.value = true
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

onMounted(() => {
  loadPoints().catch(() => {
    ElMessage.error('初始化审核管理失败')
  })
})
</script>

<style scoped>
.annotation-review-page {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-family: var(--font-display);
  font-weight: 600;
}

.progress-bars {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 8px 0;
}

.progress-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.progress-label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 13px;
  color: var(--text-secondary);
}

.progress-value {
  font-family: var(--font-mono);
  font-weight: 600;
  font-size: 14px;
  color: var(--text-primary);
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

.review-actions {
  margin-top: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
</style>
