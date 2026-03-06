import request from './request'

export interface PointRow {
  point_id: string
  label: string
  source_kind: string
  annotation_count: number
  segment_count: number
  updated_at: string | null
  score_avg: number | null
  score_method: string | null
  review_status: string | null
}

export interface PointSummaryResponse {
  point_id: string
  annotation_summary: {
    count: number
    latest_source_kind: string | null
    latest_annotation_count: number
    latest_segment_count: number
    latest_updated_at: string | null
  }
  inference_summary: {
    count: number
    latest_method: string | null
    latest_score_avg: number | null
    latest_created_at: string | null
  }
  review_summary: {
    count: number
    latest_status: string | null
    latest_updated_at: string | null
  }
}

export interface PointTimelineEvent {
  type: 'annotation' | 'inference' | 'review'
  time: string | null
  data: Record<string, unknown>
}

export function listPoints(params: { keyword?: string; limit?: number } = {}) {
  return request({
    url: '/points',
    method: 'get',
    params,
  })
}

export function getPointDetail(pointId: string) {
  return request({
    url: `/points/${encodeURIComponent(pointId)}`,
    method: 'get',
  })
}

export function getPointTimeline(pointId: string, limit = 200) {
  return request({
    url: `/points/${encodeURIComponent(pointId)}/timeline`,
    method: 'get',
    params: { limit },
  })
}
