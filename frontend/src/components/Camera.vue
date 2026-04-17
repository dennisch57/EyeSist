<template>
  <div class="camera-card">
    <p class="title">Camera Preview</p>

    <div ref="cameraBoxRef" class="camera-box">
      <video ref="videoRef" autoplay playsinline></video>

      <!-- TODO: Remove bbox overlay after YOLO detection testing is complete. -->
      <div v-if="isActive" class="bbox-layer">
        <div
          v-for="(box, index) in boxes"
          :key="index"
          class="bbox"
          :style="getBoxStyle(box)"
        ></div>
      </div>

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
  import CalibrationModal from './CalibrationModal.vue';
  import { onMounted, onUnmounted } from "vue";

  const { direction } = useEyeStore();
  const showCalibration = ref(false);

  const handlePrediction = (dir) => {
    direction.value = dir;
  };

  const videoRef = ref(null);
  const cameraBoxRef = ref(null);
  const isActive = ref(false);
  const boxes = ref([]);
  const frameSize = ref({ width: 1, height: 1 });

  let socket = null;
  let canvas = null;
  let ctx = null;
  let awaitingResponse = false;

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
      await videoRef.value.play();
      startStreaming();
    } catch (err) {
      console.error("Camera error:", err);
    }
  };

  const connectWebSocket = () => {
    socket = new WebSocket("ws://localhost:8000/ws/predict");
    socket.binaryType = "arraybuffer";

    socket.onopen = () => {
      console.log("WebSocket connected");
      socket.send(JSON.stringify({ session_id: null }));
      if (isActive.value) {
        startStreaming();
      }
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Backend response:", data);

      if (typeof data.gaze === "string") {
        handlePrediction(data.gaze);
      }

      boxes.value = Array.isArray(data.boxes) ? data.boxes : [];
      if (data.frame_size?.width && data.frame_size?.height) {
        frameSize.value = data.frame_size;
      }

      awaitingResponse = false;
      if (isActive.value) {
        sendFrame();
      }
    };

    socket.onclose = () => {
      console.log("WebSocket disconnected");
      awaitingResponse = false;
    };
  };

  const sendFrame = async () => {
    const video = videoRef.value;

    if (!video || !socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }

    if (awaitingResponse || video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }

    if (!canvas) {
      canvas = document.createElement("canvas");
      ctx = canvas.getContext("2d");
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const blob = await new Promise((resolve) => {
      canvas.toBlob(resolve, "image/jpeg", 0.8);
    });
    if (!blob) {
      return;
    }

    const buffer = await blob.arrayBuffer();
    awaitingResponse = true;
    socket.send(buffer);
  };

  const startStreaming = () => {
    if (!awaitingResponse) {
      sendFrame();
    }
  };

  const getBoxStyle = (box) => {
    const container = cameraBoxRef.value;
    if (!container || !box) {
      return {};
    }

    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const frameWidth = frameSize.value.width;
    const frameHeight = frameSize.value.height;
    const scale = Math.max(containerWidth / frameWidth, containerHeight / frameHeight);
    const renderedWidth = frameWidth * scale;
    const renderedHeight = frameHeight * scale;
    const offsetX = (containerWidth - renderedWidth) / 2;
    const offsetY = (containerHeight - renderedHeight) / 2;

    return {
      left: `${offsetX + box.x1 * scale}px`,
      top: `${offsetY + box.y1 * scale}px`,
      width: `${(box.x2 - box.x1) * scale}px`,
      height: `${(box.y2 - box.y1) * scale}px`,
    };
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

  onMounted(() => {
    connectWebSocket();
  });

  onUnmounted(() => {
    if (socket) socket.close();
  });

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

.bbox-layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.bbox {
  position: absolute;
  border: 2px solid #22c55e;
  border-radius: 8px;
  box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.5);
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
