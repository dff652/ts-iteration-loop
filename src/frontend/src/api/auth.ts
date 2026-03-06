import request from './request'

export interface LoginPayload {
    username: string
    password: string
}

export interface LoginResponse {
    success: boolean
    token: string
    username: string
    display_name: string
    role: string
}

export interface UserInfo {
    username: string
    display_name: string
    role: string
}

export function login(data: LoginPayload) {
    return request({
        url: '/auth/login',
        method: 'post',
        data,
    })
}

export function register(data: { username: string; password: string; display_name?: string; role?: string }) {
    return request({
        url: '/auth/register',
        method: 'post',
        data,
    })
}

// ==================== Token 管理 ====================

const TOKEN_KEY = 'ts_loop_token'
const USER_KEY = 'ts_loop_user'

export function getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
    localStorage.setItem(TOKEN_KEY, token)
}

export function removeToken(): void {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
}

export function getStoredUser(): UserInfo | null {
    const raw = localStorage.getItem(USER_KEY)
    if (!raw) return null
    try { return JSON.parse(raw) } catch { return null }
}

export function setStoredUser(user: UserInfo): void {
    localStorage.setItem(USER_KEY, JSON.stringify(user))
}

export function isLoggedIn(): boolean {
    return !!getToken()
}

export function logout(): void {
    removeToken()
    window.location.href = '/login'
}
