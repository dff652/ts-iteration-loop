import Vue from 'vue'
import App from './App'
import router from './router'

// Import Bootstrap CSS
import 'bootstrap/dist/css/bootstrap.min.css'

// Global components
import BaseView from '@/components/BaseView'
import BaseNavbar from '@/components/BaseNavbar'

Vue.config.productionTip = false

// Register global components
Vue.component('BaseView', BaseView)
Vue.component('BaseNavbar', BaseNavbar)

// Custom directive for visibility toggle
Vue.directive('visible', function (el, binding) {
	el.style.visibility = binding.value ? 'visible' : 'hidden'
})

/* eslint-disable no-new */
new Vue({
	el: '#app',
	router,
	components: { App },
	template: '<App/>'
})