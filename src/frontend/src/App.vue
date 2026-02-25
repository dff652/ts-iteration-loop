<template>
  <el-container class="app-container">
    <el-aside width="240px" class="app-sidebar">
      <div class="sidebar-logo">
        <h2>TS-Loop</h2>
      </div>
      <el-menu
        :default-active="route.path"
        class="el-menu-vertical"
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409EFF"
        router
      >
        <el-menu-item index="/assets">
          <el-icon><DataBoard /></el-icon>
          <span>数据资产管理</span>
        </el-menu-item>
        <el-menu-item index="/inference">
          <el-icon><Monitor /></el-icon>
          <span>推理与监控</span>
        </el-menu-item>
        <el-menu-item index="/training">
          <el-icon><Cpu /></el-icon>
          <span>模型微调</span>
        </el-menu-item>
        <el-menu-item index="/models">
          <el-icon><Box /></el-icon>
          <span>模型资产对比</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    
    <el-container>
      <el-header class="app-header">
        <div class="header-breadcrumb">
          <el-breadcrumb separator="/">
            <el-breadcrumb-item>Iteration Loop</el-breadcrumb-item>
            <el-breadcrumb-item>{{ route.meta.title || route.name }}</el-breadcrumb-item>
          </el-breadcrumb>
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
import { useRoute } from 'vue-router'
const route = useRoute()
</script>

<style>
body, html {
  margin: 0;
  padding: 0;
  height: 100%;
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB',
  'Microsoft YaHei', '微软雅黑', Arial, sans-serif;
}

#app {
  height: 100vh;
}

.app-container {
  height: 100%;
}

.app-sidebar {
  background-color: #304156;
  transition: width 0.28s;
}

.sidebar-logo {
  height: 60px;
  line-height: 60px;
  text-align: center;
  color: #fff;
  border-bottom: 1px solid #1f2d3d;
  background-color: #2b3643;
}

.sidebar-logo h2 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
}

.el-menu-vertical {
  border-right: none;
}

.app-header {
  height: 60px;
  background-color: #fff;
  box-shadow: 0 1px 4px rgba(0,21,41,.08);
  display: flex;
  align-items: center;
  padding: 0 20px;
}

.app-main {
  background-color: #f0f2f5;
  padding: 20px;
}

/* fade-transform transition */
.fade-transform-leave-active,
.fade-transform-enter-active {
  transition: all .3s;
}

.fade-transform-enter-from {
  opacity: 0;
  transform: translateX(-30px);
}

.fade-transform-leave-to {
  opacity: 0;
  transform: translateX(30px);
}
</style>
