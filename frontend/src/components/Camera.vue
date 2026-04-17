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

    <button class="calibrate-btn" @click="showCalibration = true">
      Calibrate
    </button>

    <CalibrationModal v-if="showCalibration" @close="showCalibration = false" />
  </div>
</template>

<script setup>
  import { ref } from "vue";
  import { useEyeStore } from "@/store/eyeStore";

  const { direction } = useEyeStore();

  const handlePrediction = (dir) => {
    direction.value = dir;
  };

  const videoRef = ref(null);
  const isActive = ref(false);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { min: 640, ideal: 1920, max: 1920 },
          height: { min: 480, ideal: 1080, max: 1080 },
          frameRate: { ideal: 30 }
        }
      });

      videoRef.value.srcObject = stream;
      isActive.value = true;

      startStreaming();
    } catch (err) {
      console.error("Camera error:", err);
    }
  };

  let interval = null;

  const sendFrame = () => {
    const canvas = document.createElement("canvas");
    const video = videoRef.value;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    console.log("Resolution:", video.videoWidth, "x", video.videoHeight);

    const base64 = canvas.toDataURL("image/jpeg");

    sendToBackend(base64);
  };

  const startStreaming = () => {
    interval = setInterval(sendFrame, 200); // 5 FPS (good balance)
  };

  // const sendToBackend2 = (base64) => {
  //   fetch("/api/eye-tracking", {
  //     method: "POST",
  //     headers: { "Content-Type": "application/json" },
  //     body: JSON.stringify({ image: base64 })
  //   })
  //     .then((res) => res.json())
  //     .then((data) => {
  //       console.log("Backend response:", data);
  //     })
  //     .catch((err) => {
  //       console.error("Error sending frame:", err);
  //     });
  // };

  const sendToBackend = async (image) => {
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image }),
      });

      const data = await res.json();

      handlePrediction(data.direction);
    } catch (err) {
      console.error(err);
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