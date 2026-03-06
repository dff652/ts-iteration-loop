import request from './request'

export interface IotdbSourceItem {
    id: string
    name: string
    host: string
    port: string
    username: string
    source_path: string
    point_name: string
    target_points: number
    description: string | null
    created_at: string | null
    updated_at: string | null
}

export interface IotdbSourceCreatePayload {
    name: string
    host: string
    port: string
    username: string
    password: string
    source_path: string
    point_name: string
    target_points: number
    description?: string
}

export interface IotdbSourceUpdatePayload {
    name?: string
    host?: string
    port?: string
    username?: string
    password?: string
    source_path?: string
    point_name?: string
    target_points?: number
    description?: string
}

export interface IotdbSourceAcquirePayload {
    start_time?: string
    end_time?: string
    target_points?: number
}

export function listIotdbSources() {
    return request({ url: '/data/sources', method: 'get' })
}

export function createIotdbSource(data: IotdbSourceCreatePayload) {
    return request({ url: '/data/sources', method: 'post', data })
}

export function updateIotdbSource(id: string, data: IotdbSourceUpdatePayload) {
    return request({ url: `/data/sources/${id}`, method: 'put', data })
}

export function deleteIotdbSource(id: string) {
    return request({ url: `/data/sources/${id}`, method: 'delete' })
}

export function acquireFromSource(id: string, data: IotdbSourceAcquirePayload) {
    return request({ url: `/data/sources/${id}/acquire`, method: 'post', data })
}
