import axios from 'axios'
import { ElMessage } from 'element-plus'

const request = axios.create({
    baseURL: '/api/v1',
    timeout: 30000,
})

// Request Interceptor
request.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('ts_loop_token')
        if (token && config.headers) {
            config.headers.Authorization = `Bearer ${token}`
        }
        return config
    },
    (error) => {
        return Promise.reject(error)
    }
)

// Response Interceptor
request.interceptors.response.use(
    (response) => {
        const res = response.data
        // You can handle unified response structures here based on FastAPI outputs
        if (res && res.success === false) {
            ElMessage.error(res.message || 'API Request Failed')
            return Promise.reject(new Error(res.message || 'Error'))
        }
        return res
    },
    (error) => {
        console.error('API Error:', error)
        const silent = error?.config?.__silent
        if (!silent) {
            ElMessage.error(error.response?.data?.message || error.message || 'Request failed with status code ' + (error.response?.status || 'unknown'))
        }
        return Promise.reject(error)
    }
)

export default request
