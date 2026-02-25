import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
    history: createWebHistory(),
    routes: [
        {
            path: '/',
            redirect: '/assets'
        },
        {
            path: '/assets',
            name: 'DataAssets',
            component: () => import('../views/DataAssets.vue'),
            meta: { title: '数据资产管理' }
        },
        {
            path: '/inference',
            name: 'Inference',
            component: () => import('../views/DataAssets.vue'), // Placeholder
            meta: { title: '推理与监控' }
        },
        {
            path: '/training',
            name: 'Training',
            component: () => import('../views/DataAssets.vue'), // Placeholder
            meta: { title: '模型微调' }
        },
        {
            path: '/models',
            name: 'Models',
            component: () => import('../views/DataAssets.vue'), // Placeholder
            meta: { title: '模型资产对比' }
        },
    ]
})

export default router
