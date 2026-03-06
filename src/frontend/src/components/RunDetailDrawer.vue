<template>
  <el-drawer v-model="visible" :title="title" size="50%">
    <el-skeleton :rows="4" animated v-if="loading" />
    <template v-else-if="detail">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="Run ID">{{ detail.run_id || '-' }}</el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="statusTagType(detail.status || '')">{{ detail.status || '-' }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item v-if="detail.task_type" label="任务类型">{{ detail.task_type }}</el-descriptions-item>
        <el-descriptions-item v-if="detail.trigger_mode" label="触发方式">{{ detail.trigger_mode }}</el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ formatTime(detail.created_at || null) }}</el-descriptions-item>
        <el-descriptions-item label="结束时间">{{ formatTime(detail.completed_at || null) }}</el-descriptions-item>
        <el-descriptions-item label="错误信息" :span="2">{{ detail.error || '-' }}</el-descriptions-item>
      </el-descriptions>

      <el-divider content-position="left">步骤状态</el-divider>
      <el-table :data="detail.steps || []" border stripe>
        <el-table-column prop="step_name" label="Step" width="150" />
        <el-table-column label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="statusTagType(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="依赖" min-width="140">
          <template #default="{ row }">
            {{ Array.isArray(row.depends_on) && row.depends_on.length ? row.depends_on.join(', ') : '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="message" label="消息" min-width="200" show-overflow-tooltip />
        <el-table-column label="开始" width="170">
          <template #default="{ row }">{{ formatTime(row.started_at) }}</template>
        </el-table-column>
        <el-table-column label="结束" width="170">
          <template #default="{ row }">{{ formatTime(row.completed_at) }}</template>
        </el-table-column>
      </el-table>
    </template>
  </el-drawer>
</template>

<script setup lang="ts">
import { formatTime, statusTagType } from '../utils/format'
import type { TaskCenterRunStatusData } from '../api/taskCenter'

defineProps<{
  title?: string
  loading: boolean
  detail: TaskCenterRunStatusData | null
}>()

const visible = defineModel<boolean>({ default: false })
</script>
