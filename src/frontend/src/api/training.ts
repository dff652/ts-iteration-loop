import request from './request'

export interface TrainingConfigOption {
  name: string
  path?: string
  method?: string
  description?: string
}

export interface TrainingCreatePayload {
  config_name: string
  version_tag?: string
  model_family: 'chatts' | 'qwen'
  auto_eval: boolean
  eval_truth_dir?: string
  eval_data_dir?: string
  eval_dataset_name?: string
  eval_output_dir?: string
  eval_device?: string
  eval_method?: string
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

export function listTrainingConfigs(modelFamily: 'chatts' | 'qwen') {
  return request({
    url: '/training/configs',
    method: 'get',
    params: { model_family: modelFamily },
  })
}

export function listTrainingDatasets(modelFamily: 'chatts' | 'qwen') {
  return request({
    url: '/training/datasets',
    method: 'get',
    params: { model_family: modelFamily },
  })
}

export function startTrainingTask(data: TrainingCreatePayload) {
  return request({
    url: '/training/start',
    method: 'post',
    data,
  })
}

export function getTrainingTaskStatus(taskId: string) {
  return request({
    url: `/training/status/${taskId}`,
    method: 'get',
  })
}

export function getTrainingLog(taskId: string) {
  return request({
    url: `/training/log/${taskId}`,
    method: 'get',
  })
}

export function stopTrainingTask(taskId: string) {
  return request({
    url: `/training/stop/${taskId}`,
    method: 'post',
  })
}

export function listBaseModels(modelFamily: 'chatts' | 'qwen') {
  return request({
    url: '/training/models',
    method: 'get',
    params: { model_family: modelFamily },
  })
}

