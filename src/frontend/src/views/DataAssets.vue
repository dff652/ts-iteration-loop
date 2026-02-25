<template>
  <div class="data-assets-container">
    <el-card class="box-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span>点位资产列表</span>
          <el-button type="primary" @click="handleRefresh" :icon="Refresh">刷新</el-button>
        </div>
      </template>

      <el-table
        v-loading="loading"
        :data="datasets"
        style="width: 100%"
        border
        stripe
      >
        <el-table-column prop="tags" label="类型/标签" width="120">
          <template #default="scope">
            <el-tag :type="scope.row.dataset_type === 'inference_result' ? 'success' : 'info'">
              {{ scope.row.dataset_type }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="uri" label="资源路径 (URI)" min-width="300" show-overflow-tooltip />
        <el-table-column prop="created_at" label="创建时间" width="180">
          <template #default="scope">
            {{ formatTime(scope.row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150" align="center">
          <template #default="scope">
            <el-button size="small" type="primary" link @click="viewDetails(scope.row)">
              详情
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { fetchDatasets, type DatasetAsset } from '../api/assets'
import { Refresh } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const datasets = ref<DatasetAsset[]>([])
const loading = ref(false)

const loadData = async () => {
  loading.value = true
  try {
    const res = await fetchDatasets()
    if (res.data && res.data.assets) {
      datasets.value = res.data.assets
    } else {
      // Direct array mode fallback depending on FastAPI wrapper
      datasets.value = Array.isArray(res) ? res : []
    }
  } catch (error) {
    ElMessage.error('加载资产失败')
  } finally {
    loading.value = false
  }
}

const handleRefresh = () => {
  loadData()
}

const formatTime = (timeStr: string) => {
  if (!timeStr) return '-'
  const date = new Date(timeStr)
  return date.toLocaleString()
}

const viewDetails = (row: DatasetAsset) => {
  ElMessage.info(`查看详情功能开发中... ${row.id}`)
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.data-assets-container {
  padding: 0;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
