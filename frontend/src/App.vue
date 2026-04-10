<script setup>
import AppHeader from "@/components/AppHeader.vue";
import Camera from "@/components/Camera.vue";
import TextOutput from "@/components/TextOutput.vue";
import Keyboard from "@/components/Keyboard.vue";
</script>

<template>
  <AppHeader />

  <div class="page-content">
    <div class="center-row">
      <Camera />
      <TextOutput />
    </div>
    <div class="keyboard-section">
      <Keyboard />
    </div>
  </div>
</template>

<style>
.page-content {
  padding-top: 60px;
}

.center-row {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-top: 40px;
}

.keyboard-section {
  margin: 20px auto;
  width: 50%;
}
</style>