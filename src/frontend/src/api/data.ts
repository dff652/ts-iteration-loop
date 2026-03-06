import request from './request'

export interface RawDatasetFile {
  name: string
  filename: string
  path: string
  size_bytes?: number
  modified_time?: number
  has_annotations?: boolean
  annotation_count?: number
}

export function listRawDatasets() {
  return request({
    url: '/data/datasets',
    method: 'get',
  })
}

export interface AcquireTaskPayload {
  source: string
  host: string
  port: string
  user: string
  password: string
  point_name: string
  target_points: number
  start_time?: string
  end_time?: string
}

export interface TaskCreateResponse {
  task_id: string
  status: string
  message: string
}

export function startAcquireTask(data: AcquireTaskPayload) {
  return request({
    url: '/data/acquire',
    method: 'post',
    data,
  })
}

export function createDatasetFromIotdb(data: AcquireTaskPayload) {
  return request({
    url: '/data/datasets/acquire',
    method: 'post',
    data,
  })
}

export function getAcquireTaskStatus(taskId: string) {
  return request({
    url: `/data/status/${taskId}`,
    method: 'get',
  })
}

export function getAcquireTaskLog(taskId: string, offset = 0, maxBytes = 200000) {
  return request({
    url: `/data/log/${taskId}`,
    method: 'get',
    params: {
      offset,
      max_bytes: maxBytes,
    },
  })
}

export function previewDatasetFile(filename: string, limit = 100) {
  return request({
    url: `/data/preview/${encodeURIComponent(filename)}`,
    method: 'get',
    params: { limit },
  })
}

export function uploadDatasetFile(file: File, datasetName?: string, overwrite = false) {
  const formData = new FormData()
  formData.append('file', file)
  if (datasetName && datasetName.trim()) {
    formData.append('dataset_name', datasetName.trim())
  }
  formData.append('overwrite', String(overwrite))
  return request({
    url: '/data/datasets/upload',
    method: 'post',
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
}
