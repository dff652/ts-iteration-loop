<template>
  <el-card shadow="never">
    <template #header>
      <div class="history-header">
        <span class="history-header__title">{{ title }}</span>
        <el-button @click="$emit('refresh')">刷新</el-button>
      </div>
    </template>

    <el-table :data="rows" v-loading="loading" border stripe>
      <el-table-column prop="run_id" label="Run ID" min-width="230" show-overflow-tooltip />
      <el-table-column label="状态" width="120">
        <template #default="{ row }">
          <el-tag :type="statusTagType(row.status)">{{ row.status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="trigger_mode" label="触发" width="100" />
      <el-table-column label="创建时间" width="180">
        <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
      </el-table-column>
      <el-table-column label="结束时间" width="180">
        <template #default="{ row }">{{ formatTime(row.completed_at) }}</template>
      </el-table-column>
      <el-table-column label="操作" width="220" fixed="right">
        <template #default="{ row }">
          <el-button type="primary" link @click="$emit('detail', row.run_id)">详情</el-button>
          <el-button type="warning" link @click="$emit('retry', row.run_id)">重试</el-button>
          <el-button type="danger" link @click="$emit('cancel', row.run_id)">取消</el-button>
        </template>
      </el-table-column>
    </el-table>

    <div class="pager-wrap">
      <el-pagination
        layout="total, prev, pager, next"
        :current-page="currentPage"
        :page-size="pageSize"
        :total="total"
        @current-change="$emit('page-change', $event)"
      />
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { formatTime, statusTagType } from '../utils/format'
import type { TaskCenterRunRow } from '../api/taskCenter'

defineProps<{
  title?: string
  rows: TaskCenterRunRow[]
  loading: boolean
  currentPage: number
  pageSize: number
  total: number
}>()

defineEmits<{
  refresh: []
  detail: [runId: string]
  retry: [runId: string]
  cancel: [runId: string]
  'page-change': [page: number]
}>()
</script>

<style scoped>
.history-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.history-header__title {
  font-family: var(--font-display);
  font-weight: 600;
  color: var(--text-primary);
}

.pager-wrap {
  margin-top: 12px;
  display: flex;
  justify-content: flex-end;
}
</style>
