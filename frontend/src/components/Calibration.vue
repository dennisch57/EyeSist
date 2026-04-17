<template>
  <div class="calibration">

    <h2 v-if="currentStep < directions.length">
      Look {{ directions[currentStep] }}
    </h2>

    <h1 v-if="countdown > 0 && !isCapturing">
      {{ countdown }}
    </h1>

    <h3 v-if="isCapturing">
      Capturing...
    </h3>

    <video ref="videoRef" autoplay class="camera" />

    <button @click="startCountdown" v-if="currentStep === 0">
      Start Calibration
    </button>

    <h2 v-if="currentStep >= directions.length">
      Calibration Complete ✅
    </h2>

  </div>
</template>

<script setup>
    import { ref } from "vue";

    const directions = ["left", "right", "up", "down", "open", "closed"];

    const currentStep = ref(0);
    const isCapturing = ref(false);
    const countdown = ref(3);

    const collectedData = ref({
        LEFT: [],
        RIGHT: [],
        UP: [],
        DOWN: [],
        OPEN: [],
        CLOSED: []
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

    const captureFrames = async (direction) => {
        isCapturing.value = true;

        const video = videoRef.value;
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        let frames = [];
        let count = 0;

        const interval = setInterval(() => {
            ctx.drawImage(video, 0, 0);

            const base64 = canvas.toDataURL("image/jpeg");

            frames.push(base64);
            count++;

            if (count >= 50) { // ~50 frames per class
            clearInterval(interval);

            collectedData.value[direction] = frames;
            isCapturing.value = false;

            nextStep();
            }
        }, 100); // every 100ms
    };

    const startCapture = () => {
        const direction = directions[currentStep.value];
        captureFrames(direction);
    };

    const nextStep = () => {
        currentStep.value++;

        if (currentStep.value < directions.length) {
            startCountdown();
        } else {
            finishCalibration();
        }
    };

    const finishCalibration = () => {
        console.log("Calibration done:", collectedData.value);

    // later: send to backend
    };
</script>

<style scoped>
    .calibration {
    text-align: center;
    color: white;
    }

    .camera {
    width: 400px;
    border-radius: 12px;
    margin-top: 20px;
    }

    h2 {
    font-size: 28px;
    }

    h1 {
    font-size: 60px;
    color: #3b82f6;
    }
</style>