<template>
  <el-card shadow="never" class="stat-card" :class="[`stat-card--${color}`]">
    <div class="stat-card__icon">
      <slot name="icon">
        <span class="stat-card__emoji">{{ icon }}</span>
      </slot>
    </div>
    <div class="stat-card__content">
      <div class="stat-card__value">{{ value }}</div>
      <div class="stat-card__label">{{ label }}</div>
    </div>
  </el-card>
</template>

<script setup lang="ts">
defineProps<{
  label: string
  value: string | number
  icon?: string
  color?: 'default' | 'primary' | 'success' | 'warning' | 'danger'
}>()
</script>

<style scoped>
.stat-card {
  border-left: 4px solid var(--border-color) !important;
  border-radius: 8px;
  background-color: var(--bg-panel);
}

.stat-card :deep(.el-card__body) {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px 20px;
}

.stat-card__emoji {
  font-size: 24px;
}

.stat-card__value {
  font-family: var(--font-mono);
  font-size: 28px;
  font-weight: 700;
  line-height: 1;
  color: var(--text-primary);
}

.stat-card__label {
  margin-top: 4px;
  font-family: var(--font-display);
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
}

.stat-card--primary { border-left-color: var(--accent-primary) !important; }
.stat-card--primary .stat-card__value { color: var(--accent-primary); }

.stat-card--success { border-left-color: var(--el-color-success) !important; }
.stat-card--success .stat-card__value { color: var(--el-color-success); }

.stat-card--warning { border-left-color: var(--el-color-warning) !important; }
.stat-card--warning .stat-card__value { color: var(--el-color-warning); }

.stat-card--danger { border-left-color: var(--el-color-danger) !important; }
.stat-card--danger .stat-card__value { color: var(--el-color-danger); }
</style>
