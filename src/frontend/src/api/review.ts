import request from './request'

export interface ReviewQueueItem {
  point_id: string
  source_kind: string
  annotation_count: number
  segment_count: number
  method: string | null
  score: number | null
  status: string
  reviewer: string | null
  updated_at: string | null
}

export interface ReviewQueueStats {
  total: number
  pending: number
  approved: number
  needs_fix: number
  unreviewed: number
}

export function listReviewQueue(params: {
  status?: string
  method?: string
  annotation_kind?: string
  keyword?: string
  limit?: number
  offset?: number
}) {
  return request({
    url: '/review/queue',
    method: 'get',
    params,
  })
}

export function batchUpdateReviewQueue(data: {
  point_ids: string[]
  status: 'pending' | 'approved' | 'needs_fix'
  reviewer?: string
}) {
  return request({
    url: '/review/queue/batch-update',
    method: 'post',
    data,
  })
}
