import request from './request'

export interface InferenceAlgorithm {
  id: string
  name: string
  description?: string
}

export interface InferenceCreatePayload {
  model: string
  algorithm: string
  input_files: string[]
  params?: Record<string, unknown>
}

export interface TaskCreateResponse {
  task_id: string
  status: string
  message: string
}

export interface TaskStatusResponse {
  task_id: string
  status: string
  message: string
  error?: string
}

export function listInferenceAlgorithms() {
  return request({
    url: '/inference/algorithms',
    method: 'get',
  })
}

export function startBatchInference(data: InferenceCreatePayload) {
  return request({
    url: '/inference/batch',
    method: 'post',
    data,
  })
}

export function getInferenceTaskStatus(taskId: string) {
  return request({
    url: `/inference/status/${taskId}`,
    method: 'get',
  })
}

export function getInferenceLog(taskId: string, offset: number = 0) {
  return request({
    url: `/inference/log/${taskId}`,
    method: 'get',
    params: { offset, max_bytes: 200000 },
  })
}

export function listTrainedModels(modelFamily: string) {
  return request({
    url: '/training/models',
    method: 'get',
    params: { model_family: modelFamily },
  })
}

