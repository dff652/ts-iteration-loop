import request from './request'

export interface ModelRegistryItem {
    id: string
    name: string
    model_family: string
    model_type: string
    version: string | null
    model_path: string
    base_model: string | null
    config: Record<string, unknown> | null
    metrics: Record<string, unknown> | null
    train_loss: number | null
    status: string
    tags: string[]
    description: string | null
    source_task_id: string | null
    created_by: string | null
    created_at: string | null
    updated_at: string | null
}

export function fetchModels(params: Record<string, unknown>) {
    return request({ url: '/models/', method: 'get', params })
}

export function registerModel(data: Record<string, unknown>) {
    return request({ url: '/models/', method: 'post', data })
}

export function fetchModelDetail(modelId: string) {
    return request({ url: `/models/${modelId}`, method: 'get' })
}

export function updateModel(modelId: string, data: Record<string, unknown>) {
    return request({ url: `/models/${modelId}`, method: 'put', data })
}

export function deleteModel(modelId: string) {
    return request({ url: `/models/${modelId}`, method: 'delete' })
}

export function fetchModelLoss(modelId: string) {
    return request({ url: `/models/${modelId}/loss`, method: 'get' })
}

export function compareModels(modelIds: string[]) {
    return request({ url: '/models/compare', method: 'post', data: { model_ids: modelIds } })
}

export function scanModels(modelFamily: string = 'chatts') {
    return request({ url: '/models/scan', method: 'post', params: { model_family: modelFamily } })
}

export function fetchVersionHistory(modelFamily?: string) {
    return request({ url: '/models/versions/history', method: 'get', params: modelFamily ? { model_family: modelFamily } : {} })
}
