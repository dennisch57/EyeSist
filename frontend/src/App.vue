<script setup>
import AppHeader from "@/components/AppHeader.vue";
// import Camera from "@/components/Camera.vue";
import CameraWebGazer from "@/components/CameraWebGazer.vue";
import TextOutput from "@/components/TextOutput.vue";
import Keyboard from "@/components/Keyboard.vue";
</script>

<template>
  <AppHeader />

  <div class="page-content">
    <div class="center-row">
      <div class="keyboard-wrapper">
        <Keyboard />
      </div>
      <TextOutput />
    </div>

    <div class="camera-section">
      <CameraWebGazer />
    </div>
  </div>
</template>

<style>
.page-content {
  padding-top: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.center-row {
  display: flex;
  justify-content: center;
  align-items: flex-start; /* Aligns top of keyboard with top of output */
  gap: 20px;
  margin-top: 40px;
  width: 90%; /* Allows it to expand to fit the keyboard */
}

.keyboard-wrapper {
  /* This ensures the keyboard doesn't get too small when next to the output */
  width: 50%;
}

.camera-section {
  margin: 40px auto;
  display: flex;
  justify-content: center;
}

</style>