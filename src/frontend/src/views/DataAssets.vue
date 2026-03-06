<template>
  <div class="data-assets-page">
    <PageHeader title="标注数据管理" subtitle="管理已标注的点位数据文件">
      <el-button type="primary" @click="handleRefresh" :icon="Refresh">刷新</el-button>
    </PageHeader>

    <el-row :gutter="12" class="stats-row">
      <el-col :xs="12" :sm="12">
        <StatCard label="已标注文件数" :value="annotatedFiles.length" icon="📝" color="primary" />
      </el-col>
      <el-col :xs="12" :sm="12">
        <StatCard label="总标注数" :value="totalAnnotations" icon="📍" color="success" />
      </el-col>
    </el-row>

    <el-card shadow="never">
      <div class="toolbar">
        <el-input v-model="searchKeyword" placeholder="搜索文件名称" clearable style="width: 240px" @keyup.enter="handleRefresh" />
        <el-button type="primary" @click="handleRefresh">查询</el-button>
      </div>

      <el-table
        v-loading="loading"
        :data="filteredFiles"
        style="width: 100%"
        border
        stripe
      >
        <el-table-column prop="filename" label="文件名称" min-width="220" show-overflow-tooltip>
          <template #default="scope">
            {{ scope.row.filename || scope.row.name }}
          </template>
        </el-table-column>
        <el-table-column prop="annotation_count" label="标注数量" width="120">
          <template #default="scope">
            <el-tag type="success" effect="plain" v-if="scope.row.annotation_count">{{ scope.row.annotation_count }} 条</el-tag>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="size_bytes" label="文件大小" width="120">
          <template #default="scope">
            {{ formatSize(scope.row.size_bytes) }}
          </template>
        </el-table-column>
        <el-table-column label="最后修改" width="180">
          <template #default="scope">
            {{ formatTime(scope.row.modified_time ? scope.row.modified_time * 1000 : null) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="120" align="center">
          <template #default="scope">
            <el-button size="small" type="primary" link @click="continueAnnotation(scope.row)">
              继续标注
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { listAnnotatorFiles } from '../api/annotation'
import type { RawDatasetFile } from '../api/data'
import { Refresh } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { formatTime, formatSize } from '../utils/format'
import PageHeader from '../components/PageHeader.vue'
import StatCard from '../components/StatCard.vue'

const router = useRouter()
const annotatedFiles = ref<RawDatasetFile[]>([])
const loading = ref(false)
const searchKeyword = ref('')

const totalAnnotations = computed(() => {
  return annotatedFiles.value.reduce((sum, f) => sum + (f.annotation_count || 0), 0)
})

const filteredFiles = computed(() => {
  let list = annotatedFiles.value
  if (searchKeyword.value.trim()) {
    const kw = searchKeyword.value.trim().toLowerCase()
    list = list.filter(f => (f.filename || f.name || '').toLowerCase().includes(kw))
  }
  return list
})

const loadData = async () => {
  loading.value = true
  try {
    // We pass an empty string to use the default path defined by backend user config
    const res: any = await listAnnotatorFiles('')
    const inner = res?.data || res
    const files = inner.files || inner
    if (Array.isArray(files)) {
      annotatedFiles.value = files.filter((f: any) => f.has_annotations)
    } else {
      annotatedFiles.value = []
    }
  } catch (e: any) {
    ElMessage.error(`加载标注数据失败: ${e.message}`)
    annotatedFiles.value = []
  } finally {
    loading.value = false
  }
}

const handleRefresh = () => {
  loadData()
}

const continueAnnotation = (row: RawDatasetFile) => {
  router.push(`/annotation/workbench?file=${encodeURIComponent(row.filename || row.name)}`)
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.data-assets-page {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.stats-row {
  margin-bottom: 4px;
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
}
</style>
