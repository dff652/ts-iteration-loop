<template>
  <el-card shadow="never">
    <template #header>
      <div class="log-header">
        <span class="log-header__title">
          {{ title }}
          <el-tag v-if="status" :type="statusTagType(status)" style="margin-left: 8px">{{ status }}</el-tag>
        </span>
        <div class="log-header__actions">
          <slot name="actions" />
          <el-button @click="$emit('refresh')">刷新</el-button>
        </div>
      </div>
    </template>
    <div class="task-log-terminal" ref="logContainer">
      <pre>{{ logText || placeholder }}</pre>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { statusTagType } from '../utils/format'

const props = defineProps<{
  title?: string
  logText: string
  status?: string
  placeholder?: string
}>()

defineEmits<{
  refresh: []
}>()

const logContainer = ref<HTMLElement | null>(null)

watch(() => props.logText, () => {
  nextTick(() => {
    if (logContainer.value) {
      logContainer.value.scrollTop = logContainer.value.scrollHeight
    }
  })
})
</script>

<style scoped>
.log-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.log-header__title {
  font-family: var(--font-display);
  font-weight: 600;
  display: flex;
  align-items: center;
}

.log-header__actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.task-log-terminal {
  background-color: #fafafa;
  border: 1px solid var(--border-color);
  border-left: 3px solid var(--accent-primary, #7E4C64);
  border-radius: 6px;
  height: 280px;
  overflow-y: auto;
  padding: 12px;
}

.task-log-terminal pre {
  margin: 0;
  font-family: var(--font-mono);
  font-size: 13px;
  color: #333;
  line-height: 1.6;
  white-space: pre-wrap;
  word-wrap: break-word;
}
</style>
