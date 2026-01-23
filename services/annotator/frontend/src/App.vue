<template>
  <div id="app">
    <router-view/>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: #f4f4f4;
  color: #2c3e50;
}

#app {
  height: 100%;
}

* {
  box-sizing: border-box;
}

/* Global light theme styles */
.container-fluid {
  background-color: #f4f4f4;
}

a {
  color: #7E4C64;
}

a:hover {
  color: #6a3f54;
}

/* Navbar override for light theme */
.navbar {
  background-color: #fff !important;
  border-bottom: 1px solid #dee2e6;
}

.navbar-brand, .nav-link {
  color: #2c3e50 !important;
}

.nav-link:hover {
  color: #7E4C64 !important;
}
</style>