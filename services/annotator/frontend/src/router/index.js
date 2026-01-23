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
			meta: { requiresAuth: false }
		}
	]
})

// Route guard - BYPASS AUTH
router.beforeEach((to, from, next) => {
	next()
})

export default router