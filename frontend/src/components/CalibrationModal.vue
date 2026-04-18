<template>
  <div class="modal-overlay">
    <div class="modal">
      <h2>Calibration</h2>

      <video ref="videoRef" autoplay playsinline class="camera" />

      <h3 v-if="!completed && !isUploading">
        {{ currentStepConfig.instruction }}
      </h3>

      <h1 v-if="countdown > 0 && !isCapturing && !isUploading && !completed">
        {{ countdown }}
      </h1>

      <p v-if="isCapturing">
        Capturing {{ currentStepConfig.displayLabel }}: {{ captureCount }}/100
      </p>

      <p v-if="isUploading">Running calibration...</p>
      <p v-if="statusMessage">{{ statusMessage }}</p>

      <p v-if="metrics" class="metrics">
        Base: {{ formatAccuracy(metrics.base_accuracy) }} | Ridge:
        {{ formatAccuracy(metrics.ridge_accuracy) }}
      </p>

      <button
        v-if="!isCapturing && !isCountingDown && !isUploading && !completed"
        @click="startCountdown"
      >
        Start Capture
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from "vue";
import { onMounted, onUnmounted } from "vue";
import { useEyeStore } from "@/store/eyeStore";

const emit = defineEmits(["close", "complete"]);
const { ensureSessionId, setSessionId } = useEyeStore();

const steps = [
  { wireLabel: "left", displayLabel: "LEFT", instruction: "Look LEFT" },
  { wireLabel: "right", displayLabel: "RIGHT", instruction: "Look RIGHT" },
  { wireLabel: "up", displayLabel: "UP", instruction: "Look UP" },
  { wireLabel: "down", displayLabel: "DOWN", instruction: "Look DOWN" },
  { wireLabel: "straight", displayLabel: "OPEN", instruction: "Open your eyes" },
  { wireLabel: "closed", displayLabel: "CLOSED", instruction: "Close your eyes" },
];

const currentStep = ref(0);
const countdown = ref(3);
const isCountingDown = ref(false);
const isCapturing = ref(false);
const isUploading = ref(false);
const completed = ref(false);
const captureCount = ref(0);
const statusMessage = ref("");
const metrics = ref(null);
const videoRef = ref(null);

let stream = null;
const sessionId = ensureSessionId();

const currentStepConfig = computed(() => steps[currentStep.value] ?? steps[steps.length - 1]);

const startCamera = async () => {
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 3840, max: 3840 },
      height: { ideal: 2160, max: 2160 },
      frameRate: { ideal: 30, max: 30 },
    },
  });

  videoRef.value.srcObject = stream;
  await videoRef.value.play();
};

const stopCamera = () => {
  if (!stream) {
    return;
  }
  stream.getTracks().forEach((track) => track.stop());
  stream = null;
};

onMounted(() => {
  void startCamera();
});

onUnmounted(() => {
  stopCamera();
});

const startCountdown = () => {
  isCountingDown.value = true;
  countdown.value = 3;
  statusMessage.value = "";

  const interval = setInterval(() => {
    countdown.value -= 1;
    if (countdown.value > 0) {
      return;
    }

    clearInterval(interval);
    isCountingDown.value = false;
    void captureCurrentLabel();
  }, 1000);
};

const captureVideoFrame = async () => {
  const video = videoRef.value;
  if (!video || !video.videoWidth || !video.videoHeight) {
    throw new Error("Camera is not ready");
  }

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");

  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  const blob = await new Promise((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", 1.0);
  });

  if (!blob) {
    throw new Error("Failed to capture calibration frame");
  }

  return blob.arrayBuffer();
};

const captureCurrentLabel = async () => {
  const { wireLabel, displayLabel } = currentStepConfig.value;
  const reset = currentStep.value === 0;

  isCapturing.value = true;
  captureCount.value = 0;
  statusMessage.value = `Connecting for ${displayLabel}...`;

  const socket = new WebSocket("ws://localhost:8000/ws/calibrate");
  socket.binaryType = "arraybuffer";

  try {
    await new Promise((resolve, reject) => {
      socket.onopen = resolve;
      socket.onerror = () => reject(new Error("Calibration socket failed to open"));
    });

    socket.send(JSON.stringify({ session_id: sessionId, label: wireLabel, reset }));

    await new Promise((resolve, reject) => {
      socket.onmessage = async (event) => {
        const data = JSON.parse(event.data);

        if (data.status === "error") {
          reject(new Error(data.detail || "Calibration failed"));
          return;
        }

        if (typeof data.captured_count === "number") {
          captureCount.value = data.captured_count;
        }

        if (data.status === "ready" || data.status === "capturing" || data.status === "retry") {
          statusMessage.value =
            data.status === "retry"
              ? data.reason || "No eye detected, retrying..."
              : `Capturing ${displayLabel}...`;

          try {
            socket.send(await captureVideoFrame());
          } catch (error) {
            reject(error);
          }
          return;
        }

        if (data.status === "complete") {
          resolve();
        }
      };

      socket.onclose = () => {
        if (captureCount.value < 100) {
          reject(new Error("Calibration socket closed before completion"));
        }
      };

      socket.onerror = () => reject(new Error("Calibration socket error"));
    });

    socket.close();
    isCapturing.value = false;
    statusMessage.value = `${displayLabel} complete`;

    currentStep.value += 1;
    if (currentStep.value < steps.length) {
      return;
    }

    await finishCalibration();
  } catch (error) {
    socket.close();
    isCapturing.value = false;
    statusMessage.value = error instanceof Error ? error.message : "Calibration failed";
  }
};

const finishCalibration = async () => {
  isUploading.value = true;
  statusMessage.value = "Training personalized model...";

  try {
    const response = await fetch("http://localhost:8000/model/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Calibration failed");
    }

    metrics.value = data;
    completed.value = true;
    setSessionId(sessionId);
    statusMessage.value = "Calibration successful";
    emit("complete", sessionId);

    setTimeout(() => {
      stopCamera();
      emit("close");
    }, 3000);
  } catch (error) {
    statusMessage.value = error instanceof Error ? error.message : "Calibration failed";
  } finally {
    isUploading.value = false;
  }
};

const formatAccuracy = (value) => `${((value ?? 0) * 100).toFixed(1)}%`;
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
}

.modal {
  background: #1e293b;
  padding: 20px;
  border-radius: 12px;
  width: 500px;
  text-align: center;
}

.camera {
  width: 100%;
  border-radius: 10px;
  margin: 10px 0;
  transform: scaleX(-1);
}

.metrics {
  color: white;
}
</style>
