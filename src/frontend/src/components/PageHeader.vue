<template>
  <div class="page-header">
    <div class="page-header__left">
      <h2 class="page-header__title">
        <slot name="icon" />
        {{ title }}
      </h2>
      <p v-if="subtitle" class="page-header__subtitle">{{ subtitle }}</p>
    </div>
    <div class="page-header__actions">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
  subtitle?: string
}>()
</script>

<style scoped>
.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 0 16px 0;
  gap: 12px;
  flex-wrap: wrap;
}

.page-header__title {
  margin: 0;
  font-family: var(--font-display);
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
  display: flex;
  align-items: center;
  gap: 8px;
}

.page-header__subtitle {
  margin: 4px 0 0 0;
  font-size: 13px;
  color: var(--text-secondary);
}

.page-header__actions {
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
