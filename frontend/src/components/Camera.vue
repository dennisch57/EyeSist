<template>
  <div class="camera-card">
    <p class="title">Camera Preview</p>

    <div class="camera-box">
      <video ref="videoRef" autoplay playsinline></video>

      <div v-if="!isActive" class="overlay">
        <p>Start tracking to see video</p>
      </div>
    </div>

    <button class="btn" @click="startCamera">
      Start Eye Tracking
    </button>
  </div>
</template>

<script setup>
import { ref } from "vue";

const videoRef = ref(null);
const isActive = ref(false);

const startCamera = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true
    });

    videoRef.value.srcObject = stream;
    isActive.value = true;
  } catch (err) {
    console.error("Camera error:", err);
  }
};
</script>

<style scoped>
.camera-card {
  background: #1e293b;
  padding: 16px;
  border-radius: 12px;
  width: 400px;
}

.title {
  color: #cbd5f5;
  margin-bottom: 10px;
}

.camera-box {
  position: relative;
  height: 220px;
  background: #0f172a;
  border-radius: 10px;
  overflow: hidden;
}

video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.overlay {
  position: absolute;
  inset: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #94a3b8;
  background: rgba(15, 23, 42, 0.7);
}

.btn {
  margin-top: 12px;
  width: 100%;
  padding: 10px;
  background: #3b82f6;
  border: none;
  border-radius: 8px;
  color: white;
  cursor: pointer;
}
</style>