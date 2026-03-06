<template>
  <div class="login-page">
    <div class="login-card">
      <div class="login-header">
        <span class="logo-icon">📊</span>
        <h1>TS-LOOP</h1>
        <p class="subtitle">时序异常检测迭代循环系统</p>
      </div>

      <el-form ref="formRef" :model="form" :rules="rules" label-width="0" @submit.prevent="handleLogin">
        <el-form-item prop="username">
          <el-input
            v-model="form.username"
            placeholder="用户名"
            prefix-icon="User"
            size="large"
            @keyup.enter="handleLogin"
          />
        </el-form-item>
        <el-form-item prop="password">
          <el-input
            v-model="form.password"
            placeholder="密码"
            prefix-icon="Lock"
            type="password"
            size="large"
            show-password
            @keyup.enter="handleLogin"
          />
        </el-form-item>
        <el-form-item>
          <el-button
            type="primary"
            size="large"
            :loading="loading"
            style="width: 100%"
            @click="handleLogin"
          >
            登 录
          </el-button>
        </el-form-item>
      </el-form>

      <div class="login-footer">
        <span>默认账户: admin / admin123</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, type FormInstance, type FormRules } from 'element-plus'
import { login as loginApi, setToken, setStoredUser } from '../api/auth'

const router = useRouter()
const formRef = ref<FormInstance | null>(null)
const loading = ref(false)

const form = reactive({ username: '', password: '' })
const rules: FormRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
}

async function handleLogin(): Promise<void> {
  if (!formRef.value) return
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    loading.value = true
    try {
      const resp = await loginApi({
        username: form.username,
        password: form.password,
      })
      const data = resp as unknown as { token?: string; username?: string; display_name?: string; role?: string }
      if (data.token) {
        setToken(data.token)
        setStoredUser({
          username: data.username || form.username,
          display_name: data.display_name || form.username,
          role: data.role || 'annotator',
        })
        ElMessage.success(`欢迎, ${data.display_name || data.username}`)
        router.push('/')
      }
    } catch (err: unknown) {
      // Error already shown by interceptor
      console.error('Login failed', err)
    } finally {
      loading.value = false
    }
  })
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}

.login-card {
  width: 400px;
  padding: 40px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  backdrop-filter: blur(12px);
}

.login-header {
  text-align: center;
  margin-bottom: 32px;
}

.logo-icon {
  font-size: 40px;
}

.login-header h1 {
  margin: 8px 0 4px;
  font-size: 28px;
  font-weight: 700;
  color: #e0e0e0;
  letter-spacing: 2px;
}

.subtitle {
  color: #999;
  font-size: 13px;
  margin: 0;
}

.login-footer {
  text-align: center;
  margin-top: 16px;
  color: #666;
  font-size: 12px;
}
</style>
