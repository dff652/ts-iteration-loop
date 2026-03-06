<template>
  <el-dialog
    v-model="visible"
    title="🏷️ 标签管理"
    width="600px"
    @closed="onClosed"
  >
    <div class="label-settings-body">
      <!-- Tabs for toggling 'overall' and 'local' categories -->
      <el-tabs v-model="activeTab" class="settings-tabs">
        <el-tab-pane label="整体属性" name="overall"></el-tab-pane>
        <el-tab-pane label="局部变化" name="local"></el-tab-pane>
      </el-tabs>

      <!-- Category List -->
      <div class="category-list">
        <div
          v-for="(cat, catId) in currentCategories"
          :key="catId"
          class="category-card"
        >
          <div class="category-header">
            <el-input
              v-model="cat.name"
              size="small"
              placeholder="分类名称"
              class="cat-name-input"
            />
            <div class="category-actions">
              <!-- Color picker for local category -->
              <el-color-picker
                v-if="activeTab === 'local'"
                v-model="cat.color"
                size="small"
                show-alpha
                class="cat-color-picker"
                title="分类默认底色"
              />
              <el-button
                type="danger"
                icon="Delete"
                circle
                size="small"
                plain
                @click="deleteCategory(catId)"
                title="删除分类"
              />
            </div>
          </div>

          <!-- Labels inside current category -->
          <div class="labels-container">
            <div
              v-for="(label, idx) in cat.labels"
              :key="label.id"
              class="label-item"
            >
              <el-input
                v-model="label.text"
                size="small"
                placeholder="标签名称"
                class="label-name-input"
              />
              <el-color-picker
                v-if="activeTab === 'local'"
                v-model="label.color"
                size="small"
                class="label-color-picker"
                title="标签颜色"
              />
              <el-button
                type="info"
                icon="Close"
                circle
                size="small"
                link
                @click="deleteLabelFromCategory(catId, idx)"
                title="删除标签"
              />
            </div>
            <el-button
              size="small"
              class="add-label-btn"
              @click="addLabelToCategory(catId)"
            >
              + 添加标签
            </el-button>
          </div>
        </div>

        <el-button type="primary" plain class="add-cat-btn" @click="addCategory">
          + 添加分类
        </el-button>
      </div>
    </div>
    <template #footer>
      <div class="dialog-footer">
        <el-button @click="visible = false">取消</el-button>
        <el-button type="primary" @click="handleSave">保存变更</el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'

import { ElMessage, ElMessageBox } from 'element-plus'
import { saveLabels } from '../api/annotation'

export interface LabelItem {
  id: string
  text: string
  color?: string
}

export interface Category {
  name: string
  labels: LabelItem[]
  color?: string
}

export interface LabelsConfig {
  overall_attribute: Record<string, Category>
  local_change: Record<string, Category>
}

// Props and Emits
const props = defineProps<{
  modelValue: boolean
  initialLabels: LabelsConfig
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', val: boolean): void
  (e: 'saved', newLabels: LabelsConfig): void
}>()

// State
const visible = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val)
})

const activeTab = ref<'overall' | 'local'>('overall')

// Local editable copy of labels
const editableLabels = ref<LabelsConfig>({
  overall_attribute: {},
  local_change: {}
})

// Current shown categories depending on Tab
const currentCategories = computed(() => {
  return activeTab.value === 'overall'
    ? editableLabels.value.overall_attribute
    : editableLabels.value.local_change
})

// Initialize/Reset
watch(
  () => props.modelValue,
  (newVal) => {
    if (newVal) {
      activeTab.value = 'overall'
      editableLabels.value = JSON.parse(JSON.stringify(props.initialLabels))
      if (!editableLabels.value.overall_attribute) {
        editableLabels.value.overall_attribute = {}
      }
      if (!editableLabels.value.local_change) {
        editableLabels.value.local_change = {}
      }
    }
  }
)

const palette = [
  '#ef4444', '#f97316', '#f59e0b', '#eab308', '#84cc16',
  '#22c55e', '#10b981', '#14b8a6', '#06b6d4', '#0ea5e9',
  '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef',
  '#ec4899', '#f43f5e', '#78716c', '#64748b', '#0f172a'
]

function generateRandomColor(): string {
  const c = palette[Math.floor(Math.random() * palette.length)]
  return c || '#7E4C64'
}

// Action: Category
function addCategory() {
  const newId = 'cat_' + Date.now()
  const target = currentCategories.value
  target[newId] = {
    name: '新分类',
    labels: [],
    color: activeTab.value === 'local' ? '#6b7280' : undefined
  }
}

async function deleteCategory(catId: string) {
  try {
    await ElMessageBox.confirm(`确定删除分类 "${currentCategories.value[catId]?.name}" 吗？`, '警告', {
      type: 'warning'
    })
    delete currentCategories.value[catId]
  } catch {
    // cancelled
  }
}

// Action: Label
function addLabelToCategory(catId: string) {
  const cat = currentCategories.value[catId]
  if (!cat) return
  if (!cat.labels) cat.labels = []
  
  cat.labels.push({
    id: 'label_' + Date.now(),
    text: '新标签',
    color: activeTab.value === 'local' ? generateRandomColor() : undefined
  })
}

function deleteLabelFromCategory(catId: string, idx: number) {
  const cat = currentCategories.value[catId]
  if (cat && cat.labels) {
    cat.labels.splice(idx, 1)
  }
}

// Action: Save & Network
async function handleSave() {
  try {
    const payload = JSON.parse(JSON.stringify(editableLabels.value))
    await saveLabels(payload)
    ElMessage.success('标签配置保存成功！')
    emit('saved', payload)
    visible.value = false
  } catch (e: any) {
    ElMessage.error(`保存失败: ${e.message}`)
  }
}

function onClosed() {
  // cleanup if necessary
}
</script>

<style scoped>
.label-settings-body {
  min-height: 400px;
  max-height: 60vh;
  overflow-y: auto;
  padding-right: 8px;
}

.category-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-top: 12px;
}

.category-card {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  padding: 12px;
  background: var(--el-fill-color-light);
}

.category-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--el-border-color-lighter);
}

.cat-name-input {
  width: 200px;
  font-weight: 500;
}

.category-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.labels-container {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.label-item {
  display: flex;
  align-items: center;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color);
  border-radius: 6px;
  padding: 4px;
  gap: 4px;
}

.label-name-input {
  width: 120px;
}

.label-color-picker :deep(.el-color-picker__trigger) {
  padding: 2px;
  width: 24px;
  height: 24px;
}

.cat-color-picker :deep(.el-color-picker__trigger) {
  padding: 2px;
  width: 28px;
  height: 28px;
}

.add-label-btn {
  margin-left: 4px;
}

.add-cat-btn {
  align-self: flex-start;
  margin-top: 8px;
}
</style>
