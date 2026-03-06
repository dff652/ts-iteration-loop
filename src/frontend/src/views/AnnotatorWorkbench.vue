<template>
  <div class="annotator-page">
    <el-card shadow="never" class="toolbar-card">
      <div class="toolbar">
        <div class="title-wrap">
          <h3>标注工具工作台</h3>
          <p class="subtitle">在新界面内嵌运行 Annotator，保留新开窗口兜底入口。</p>
        </div>
        <div class="actions">
          <el-button @click="reloadFrame">刷新嵌入页</el-button>
          <el-button type="primary" @click="openInNewTab">新窗口打开</el-button>
        </div>
      </div>
      <el-alert
        v-if="showHint"
        type="warning"
        :closable="false"
        show-icon
        title="若嵌入页为空白，请确认 Annotator(5000) 已启动，或点击“新窗口打开”。"
      />
    </el-card>

    <el-card shadow="never" class="frame-card">
      <iframe :key="frameKey" :src="annotatorUrl" class="annotator-frame" title="Annotator Workbench" />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

const frameKey = ref(0)
const showHint = ref(true)

const annotatorUrl = computed(() => {
  const custom = String(import.meta.env.VITE_ANNOTATOR_URL || '').trim()
  if (custom) return custom
  const host = window.location.hostname || '127.0.0.1'
  return `http://${host}:5000`
})

function reloadFrame(): void {
  frameKey.value += 1
}

function openInNewTab(): void {
  window.open(annotatorUrl.value, '_blank', 'noopener,noreferrer')
}
</script>

<style scoped>
.annotator-page {
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: calc(100vh - 140px);
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.title-wrap h3 {
  margin: 0;
  font-size: 18px;
}

.subtitle {
  margin: 4px 0 0;
  color: #606266;
  font-size: 13px;
}

.actions {
  display: flex;
  gap: 8px;
}

.toolbar-card :deep(.el-card__body) {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.frame-card {
  flex: 1;
}

.frame-card :deep(.el-card__body) {
  height: 100%;
  padding: 10px;
}

.annotator-frame {
  width: 100%;
  height: 100%;
  min-height: 520px;
  border: 1px solid #dcdfe6;
  border-radius: 6px;
  background: #fff;
}

@media (max-width: 768px) {
  .annotator-page {
    height: auto;
    min-height: calc(100vh - 140px);
  }

  .annotator-frame {
    min-height: 420px;
  }
}
</style>
