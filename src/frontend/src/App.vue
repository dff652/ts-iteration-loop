<template>
  <!-- 登录页：裸 router-view，无侧边栏 -->
  <router-view v-if="isLoginPage" />

  <!-- 已登录：完整布局 -->
  <el-container v-else class="app-container">
    <el-aside v-if="!isFullscreenPage" width="220px" class="app-sidebar">
      <div class="sidebar-logo">
        <span class="logo-icon">📊</span>
        <h2>TS-LOOP</h2>
      </div>
      <el-menu
        :default-active="route.path"
        class="sys-menu"
        background-color="transparent"
        text-color="var(--text-secondary)"
        active-text-color="var(--accent-primary)"
        router
        :default-openeds="['data', 'annotation', 'model']"
      >
        <!-- 工作台 (主页) -->
        <el-menu-item index="/tasks">
          <el-icon><Operation /></el-icon>
          <span>工作台</span>
        </el-menu-item>

        <!-- 数据管理 -->
        <el-sub-menu index="data">
          <template #title>
            <el-icon><FolderOpened /></el-icon>
            <span>数据管理</span>
          </template>
          <el-menu-item index="/data/datasets">
            <el-icon><Document /></el-icon>
            <span>数据集</span>
          </el-menu-item>
          <el-menu-item index="/data/import">
            <el-icon><Upload /></el-icon>
            <span>数据导入</span>
          </el-menu-item>
          <el-menu-item index="/assets/data">
            <el-icon><DataLine /></el-icon>
            <span>数据集</span>
          </el-menu-item>
        </el-sub-menu>

        <!-- 标注中心 -->
        <el-sub-menu index="annotation">
          <template #title>
            <el-icon><DataBoard /></el-icon>
            <span>标注中心</span>
          </template>
          <el-menu-item index="/annotation/workbench">
            <el-icon><Edit /></el-icon>
            <span>标注工作台</span>
          </el-menu-item>
          <el-menu-item index="/annotation/review">
            <el-icon><Finished /></el-icon>
            <span>审核管理</span>
          </el-menu-item>
        </el-sub-menu>

        <!-- 模型中心 -->
        <el-sub-menu index="model">
          <template #title>
            <el-icon><Cpu /></el-icon>
            <span>模型中心</span>
          </template>
          <el-menu-item index="/model/inference">
            <el-icon><Monitor /></el-icon>
            <span>推理任务</span>
          </el-menu-item>
          <el-menu-item index="/model/training">
            <el-icon><SetUp /></el-icon>
            <span>训练任务</span>
          </el-menu-item>
          <el-menu-item index="/model/registry">
            <el-icon><Histogram /></el-icon>
            <span>模型仓库</span>
          </el-menu-item>
        </el-sub-menu>

      </el-menu>
    </el-aside>

    <el-container class="app-main-wrapper">
      <el-header v-if="!isFullscreenPage" class="app-header">
        <div class="header-breadcrumb">
          <span class="path-current">{{ route.meta.title || route.name }}</span>
        </div>
        <div class="header-status">
          <template v-if="currentUser">
            <span class="user-name">{{ currentUser.display_name }}</span>
            <el-button type="danger" link size="small" @click="handleLogout">退出</el-button>
          </template>
          <span class="status-dot"></span>
          <span class="status-text">系统在线</span>
        </div>
      </el-header>

      <el-main class="app-main">
        <router-view v-slot="{ Component }">
          <transition name="fade-transform" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { getStoredUser, logout, type UserInfo } from './api/auth'

const route = useRoute()
const isLoginPage = computed(() => route.path === '/login')
const isFullscreenPage = computed(() => route.meta.fullscreen === true)
const currentUser = computed<UserInfo | null>(() => getStoredUser())

function handleLogout(): void {
  logout()
}
</script>

<style scoped>
.app-container {
  height: 100vh;
  background-color: var(--bg-dark);
}

.app-sidebar {
  background-color: var(--bg-panel);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  box-shadow: 1px 0 3px rgba(0, 0, 0, 0.05);
  overflow-y: auto;
}

.sidebar-logo {
  height: 56px;
  display: flex;
  align-items: center;
  padding: 0 20px;
  border-bottom: 1px solid var(--border-color);
  gap: 10px;
  flex-shrink: 0;
}

.logo-icon {
  font-size: 20px;
}

.sidebar-logo h2 {
  margin: 0;
  font-family: var(--font-display);
  font-size: 18px;
  font-weight: 700;
  color: var(--accent-primary);
}

.sys-menu {
  border-right: none;
  font-family: var(--font-display);
  font-weight: 500;
  margin-top: 4px;
}

/* Sub-menu group styling */
.sys-menu :deep(.el-sub-menu__title) {
  height: 40px;
  line-height: 40px;
  margin: 2px 8px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
}

.sys-menu :deep(.el-sub-menu__title:hover) {
  background-color: var(--bg-hover) !important;
}

.sys-menu :deep(.el-sub-menu .el-menu-item) {
  height: 38px;
  line-height: 38px;
  padding-left: 52px !important;
  margin: 1px 8px;
  border-radius: 6px;
  font-size: 13px;
  min-width: unset;
}

.el-menu-item {
  height: 44px;
  line-height: 44px;
  margin: 2px 8px;
  border-radius: 6px;
  font-size: 14px;
}

.el-menu-item:hover,
.sys-menu :deep(.el-menu-item:hover) {
  background-color: var(--bg-hover) !important;
}

.el-menu-item.is-active,
.sys-menu :deep(.el-menu-item.is-active) {
  background-color: var(--accent-primary-light, #f8f4f6) !important;
  border-right: none;
  color: var(--accent-primary);
  font-weight: 600;
}

.app-main-wrapper {
  background-color: var(--bg-dark);
}

.app-header {
  height: 50px;
  background-color: var(--bg-panel);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

.header-breadcrumb {
  font-family: var(--font-display);
  font-size: 15px;
  color: var(--text-primary);
  font-weight: 600;
}

.header-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: var(--font-display);
  font-size: 12px;
  color: var(--el-color-success);
}

.status-dot {
  width: 6px;
  height: 6px;
  background-color: var(--el-color-success);
  border-radius: 50%;
}

.app-main {
  padding: 20px;
  height: calc(100vh - 50px);
  overflow-y: auto;
  background-color: var(--bg-dark);
}

/* fade-transform */
.fade-transform-leave-active,
.fade-transform-enter-active {
  transition: all .2s cubic-bezier(0, 0.55, 0.45, 1);
}

.fade-transform-enter-from {
  opacity: 0;
  transform: translateY(8px);
}

.fade-transform-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}
</style>
