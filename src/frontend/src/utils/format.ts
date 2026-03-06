export function formatTime(value: string | number | null | undefined): string {
    if (!value) return '-'
    const dt = new Date(value)
    return Number.isNaN(dt.getTime()) ? String(value) : dt.toLocaleString()
}

export function formatSize(bytes?: number): string {
    const value = Number(bytes || 0)
    if (value <= 0) return '-'
    if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(2)} MB`
    if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`
    return `${value} B`
}

export function jsonCompact(value: unknown): string {
    try {
        return JSON.stringify(value || {})
    } catch {
        return '-'
    }
}

export function formatJson(obj: unknown): string {
    if (!obj) return '-'
    try { return JSON.stringify(obj, null, 2) } catch { return String(obj) }
}

export function statusTagType(status: string): 'success' | 'warning' | 'danger' | 'info' | 'primary' {
    const s = String(status || '').toLowerCase()
    if (s === 'completed' || s === 'active' || s === 'approved') return 'success'
    if (s === 'running' || s === 'runnable' || s === 'auto') return 'primary'
    if (s === 'failed' || s === 'timeout' || s === 'archived' || s === 'deleted') return 'danger'
    if (s === 'cancelled' || s === 'stopped' || s === 'deprecated' || s === 'needs_fix') return 'warning'
    return 'info'
}

export function getApiData<T>(response: unknown): T {
    const r = response as { data?: unknown } | undefined
    return ((r?.data) ?? {}) as T
}
