<template>
  <header class="header">
    <h1 class="logo">EyeSist</h1>
  </header>
</template>

<style scoped>
.header {
  width: 100%;
  height: 60px;
  background: #1e293b;

  display: flex;
  align-items: center;
  justify-content: flex-start;

  padding: 0 20px;

  position: fixed;
  top: 0;
  left: 0;

  z-index: 1000;
}

.logo {
  color: #38bdf8;
  font-size: 22px;
  font-weight: bold;
}
</style>