import request from './request'

// Interface representing a Dataset Asset
export interface DatasetAsset {
    id: string
    uri: string
    dataset_type: string
    tags: string
    meta_data: string
    created_at: string
    updated_at: string
}

// Fetch all available datasets
export function fetchDatasets(params?: any) {
    return request({
        url: '/assets/',
        method: 'get',
        params
    })
}

// Example to fetch source files 
export function fetchSources(sourceType: string = 'annotations', pointId?: string) {
    return request({
        url: '/assets/sources',
        method: 'get',
        params: { source_type: sourceType, point_id: pointId }
    })
}
