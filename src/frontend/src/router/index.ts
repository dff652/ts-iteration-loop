import { createRouter, createWebHistory } from 'vue-router'
import { isLoggedIn } from '../api/auth'

const router = createRouter({
    history: createWebHistory(),
    routes: [
        {
            path: '/',
            redirect: '/tasks'
        },
        // ==================== 登录 ====================
        {
            path: '/login',
            name: 'Login',
            component: () => import('../views/Login.vue'),
            meta: { title: '登录', public: true }
        },
        // ==================== 数据管理 ====================
        {
            path: '/data/datasets',
            name: 'Datasets',
            component: () => import('../views/DataCenter.vue'),
            meta: { title: '数据集', group: 'data' }
        },
        {
            path: '/data/datasets/detail',
            name: 'DatasetDetail',
            component: () => import('../views/DatasetDetail.vue'),
            meta: { title: '数据集详情', group: 'data' }
        },
        {
            path: '/data/import',
            name: 'DataImport',
            component: () => import('../views/DataImport.vue'),
            meta: { title: '数据导入', group: 'data' }
        },
        {
            path: '/data/assets',
            name: 'DataAssets',
            component: () => import('../views/DataAssets.vue'),
            meta: { title: '数据集', group: 'data' }
        },
        // ==================== 标注中心 ====================
        {
            path: '/annotation/workbench',
            name: 'AnnotationWorkbench',
            component: () => import('../views/AnnotationWorkbench.vue'),
            meta: { title: '标注工作台', group: 'annotation', fullscreen: true }
        },
        {
            path: '/annotation/review',
            name: 'AnnotationReview',
            component: () => import('../views/AnnotationReview.vue'),
            meta: { title: '审核管理', group: 'annotation' }
        },
        // ==================== 模型中心 ====================
        {
            path: '/model/inference',
            name: 'Inference',
            component: () => import('../views/InferenceCreate.vue'),
            meta: { title: '推理任务', group: 'model' }
        },
        {
            path: '/model/training',
            name: 'Training',
            component: () => import('../views/TrainingCreate.vue'),
            meta: { title: '训练任务', group: 'model' }
        },
        {
            path: '/model/registry',
            name: 'ModelRegistry',
            component: () => import('../views/ModelRegistry.vue'),
            meta: { title: '模型仓库', group: 'model' }
        },
        // ==================== 工作台 ====================
        {
            path: '/tasks',
            name: 'TaskCenter',
            component: () => import('../views/TaskCenterDashboard.vue'),
            meta: { title: '工作台', group: 'tasks' }
        },
        // ==================== 兼容旧路由 (redirect) ====================
        { path: '/data-center', redirect: '/data/datasets' },
        { path: '/annotation-center', redirect: '/annotation/review' },
        { path: '/annotator', redirect: '/annotation/workbench' },
        { path: '/inference', redirect: '/model/inference' },
        { path: '/training', redirect: '/model/training' },
        { path: '/models', redirect: '/model/registry' },
        { path: '/assets', redirect: '/data/assets' },
        { path: '/task-center', redirect: '/tasks' },
    ]
})

// ==================== 路由守卫 ====================
router.beforeEach((to, _from, next) => {
    // 公开页面（登录）不需要认证
    if (to.meta.public) {
        next()
        return
    }
    // 未登录 → 跳转登录页
    if (!isLoggedIn()) {
        next({ path: '/login', query: { redirect: to.fullPath } })
        return
    }
    next()
})

export default router
