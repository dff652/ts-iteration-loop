import request from './request'

// ==================== 文件浏览 ====================

export function browseDirectory(path: string) {
  return request({
    url: '/annotator/browse',
    method: 'post',
    data: { path },
  })
}

export function listAnnotatorFiles(dataPath?: string) {
  return request({
    url: '/annotator/files',
    method: 'get',
    params: { data_path: dataPath || '' },
  })
}

// ==================== 数据读取 ====================

export function getAnnotatorData(filename: string, dataPath?: string) {
  return request({
    url: `/annotator/data/${encodeURIComponent(filename)}`,
    method: 'get',
    params: { data_path: dataPath || '' },
  })
}

// ==================== 标注 CRUD ====================

export function getAnnotations(filename: string, user?: string) {
  return request({
    url: `/annotator/annotations/${encodeURIComponent(filename)}`,
    method: 'get',
    params: { user: user || 'default' },
  })
}

export function saveAnnotations(filename: string, data: {
  filename: string
  annotations: unknown[]
  overall_attribute: Record<string, unknown>
}, user?: string) {
  return request({
    url: `/annotator/annotations/${encodeURIComponent(filename)}`,
    method: 'post',
    data,
    params: { user: user || 'default' },
  })
}

export function deleteAnnotation(filename: string, annotationId: string, user?: string) {
  return request({
    url: `/annotator/annotations/${encodeURIComponent(filename)}`,
    method: 'delete',
    params: { annotation_id: annotationId, user: user || 'default' },
  })
}

// ==================== 标签配置 ====================

export function getLabels() {
  return request({
    url: '/annotator/labels',
    method: 'get',
  })
}

export function saveLabels(labels: Record<string, unknown>) {
  return request({
    url: '/annotator/labels',
    method: 'post',
    data: labels,
  })
}

// ==================== 旧接口兼容 ====================

export function listAnnotatableFiles() {
  return request({
    url: '/annotation/files',
    method: 'get',
  })
}
