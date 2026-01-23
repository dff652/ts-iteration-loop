import Vue from 'vue'
import Router from 'vue-router'
import Index from '@/views/Index'
import Login from '@/views/Login'

Vue.use(Router)

const router = new Router({
	routes: [
		{
			path: '/login',
			name: 'login',
			component: Login
		},
		{
			path: '/',
			name: 'home',
			component: Index,
			props: true,
			meta: { requiresAuth: true }
		}
	]
})

// Route guard - redirect to login if not authenticated
router.beforeEach((to, from, next) => {
	const token = localStorage.getItem('token')

	if (to.meta.requiresAuth && !token) {
		next('/login')
	} else if (to.path === '/login' && token) {
		next('/')
	} else {
		next()
	}
})

export default router