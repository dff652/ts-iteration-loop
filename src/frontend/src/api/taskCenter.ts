import request from './request'

export interface TaskCenterRunRow {
  run_id: string
  definition_id: string | null
  task_type: string
  trigger_mode: string
  status: string
  error: string
  step_count: number
  step_status_counts: Record<string, number>
  created_at: string | null
  started_at: string | null
  completed_at: string | null
}

export interface TaskCenterStep {
  id: number
  step_name: string
  status: string
  depends_on: string[]
  message: string
  started_at: string | null
  completed_at: string | null
}

export interface TaskCenterRunStatusData {
  run_id: string
  definition_id: string | null
  task_type: string
  trigger_mode: string
  status: string
  error: string
  steps: TaskCenterStep[]
  created_at: string | null
  started_at: string | null
  completed_at: string | null
}

export interface TaskCenterDeadLetter {
  id: string
  source: string
  event_key: string
  definition_id: string
  dedupe_key: string
  execute_mode: string
  error: string
  created_at: string
}

export interface TaskCenterDefinitionRow {
  id: string
  name: string
  task_type: string
  trigger_mode: string
  schedule_cron: string | null
  enabled: boolean
  config: Record<string, unknown>
  created_at: string | null
}

export function fetchTaskCenterRuns(params: Record<string, unknown>) {
  return request({
    url: '/task-center/runs',
    method: 'get',
    params,
  })
}

export function fetchTaskCenterDefinitions() {
  return request({
    url: '/task-center/definitions',
    method: 'get',
  })
}

export function createTaskCenterRun(data: Record<string, unknown>) {
  return request({
    url: '/task-center/runs',
    method: 'post',
    data,
  })
}

export function createTaskCenterDefinition(data: Record<string, unknown>) {
  return request({
    url: '/task-center/definitions',
    method: 'post',
    data,
  })
}

export function fetchTaskCenterRunStatus(runId: string) {
  return request({
    url: `/task-center/runs/${runId}/status`,
    method: 'get',
  })
}

export function executeTaskCenterRun(runId: string, simulate: boolean) {
  return request({
    url: `/task-center/runs/${runId}/execute`,
    method: 'post',
    data: { simulate },
  })
}

export function cancelTaskCenterRun(runId: string) {
  return request({
    url: `/task-center/runs/${runId}/cancel`,
    method: 'post',
  })
}

export function retryTaskCenterRun(runId: string) {
  return request({
    url: `/task-center/runs/${runId}/retry`,
    method: 'post',
  })
}

export function fetchTaskCenterEventMetrics() {
  return request({
    url: '/task-center/events/metrics',
    method: 'get',
    __silent: true,
  } as Record<string, unknown>)
}

export function fetchTaskCenterDeadLetters(params: Record<string, unknown>) {
  return request({
    url: '/task-center/events/dead-letters',
    method: 'get',
    params,
    __silent: true,
  } as Record<string, unknown>)
}

export function replayTaskCenterDeadLetter(eventId: string, executeMode: 'dispatch' | 'simulate' | 'none' = 'dispatch') {
  return request({
    url: '/task-center/events/dead-letters/replay',
    method: 'post',
    data: {
      event_id: eventId,
      execute_mode: executeMode,
    },
  })
}
