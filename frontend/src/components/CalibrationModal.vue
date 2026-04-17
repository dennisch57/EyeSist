<template>
  <div class="modal-overlay">
    <div class="modal">

      <h2>Calibration</h2>

      <video ref="videoRef" autoplay class="camera" />

      <h3 v-if="!completed">
        Look {{ steps[currentStep] }}
      </h3>

      <h1 v-if="countdown > 0 && !isCapturing && !completed">
        {{ countdown }}
      </h1>

      <p v-if="isCapturing">Capturing...</p>

      <button v-if="!isCapturing && !completed" @click="startCountdown">
        Start Capture
      </button>

      <h2 v-if="completed">Calibration Completed ✅</h2>

    </div>
  </div>
</template>

<script setup>
    import { ref } from "vue";

    const emit = defineEmits(["close"]);

    const steps = ["LEFT", "RIGHT", "UP", "DOWN", "OPEN", "CLOSED"];

    const currentStep = ref(0);
    const isCapturing = ref(false);
    const countdown = ref(3);
    const completed = ref(false);

    const videoRef = ref(null);

    const collectedData = ref({
    LEFT: [],
    RIGHT: [],
    UP: [],
    DOWN: [],
    OPEN: [],
    CLOSED: []
    });

    const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
    });

    videoRef.value.srcObject = stream;
    };

    onMounted(() => {
    startCamera();
    });

    const startCountdown = () => {
    countdown.value = 3;

    const interval = setInterval(() => {
        countdown.value--;

        if (countdown.value === 0) {
        clearInterval(interval);
        startCapture();
        }
    }, 1000);
    };

    const startCapture = () => {
    isCapturing.value = true;

    const direction = steps[currentStep.value];
    const video = videoRef.value;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    let frames = [];
    let count = 0;

    const interval = setInterval(() => {
        ctx.drawImage(video, 0, 0);

        frames.push(canvas.toDataURL("image/jpeg"));
        count++;

        if (count >= 200) {
        clearInterval(interval);

        collectedData.value[direction] = frames;
        isCapturing.value = false;

        nextStep();
        }
    }, 100); // ~10 fps
    };

    const nextStep = () => {
    currentStep.value++;

    if (currentStep.value >= steps.length) {
        finishCalibration();
    }
    };

    const finishCalibration = async () => {
    completed.value = true;

    // send to backend
    await fetch("http://localhost:8000/calibrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectedData.value)
    });

    setTimeout(() => {
        emit("close");
    }, 5000);
    };
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0,0,0,0.7);

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
}

button {
  margin-top: 10px;
}
</style>