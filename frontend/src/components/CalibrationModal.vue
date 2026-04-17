<template>
  <div class="modal-overlay">
    <div class="modal">

      <h2>Calibration</h2>

      <video ref="videoRef" autoplay class="camera" />

    <!-- <div class="camera-wrapper">
        <video ref="videoRef" autoplay class="camera" />
        <div class="face-outline"></div>
    </div> -->

      <h3 v-if="!completed && !(steps[currentStep] === 'OPEN' || steps[currentStep] === 'CLOSED')">
        Look {{ steps[currentStep] }}
      </h3>

      <h3 v-if="!completed && (steps[currentStep] === 'OPEN' || steps[currentStep] === 'CLOSED')">
        {{ steps[currentStep] }} your eyes
      </h3>

      <h1 v-if="countdown > 0 && !isCapturing && !completed">
        {{ countdown }}
      </h1>

      <p v-if="isCapturing">Capturing...</p>

      <button :disabled="isUploading" v-if="!isCapturing && !isCountingDown && !completed" @click="startCountdown">
        Start Capture
    </button>

      <p v-if="isUploading">Uploading calibration... (this may take a few seconds)</p>

        <p v-if="uploadStatus === 'success'" style="color: green;">
        ✅ Calibration successful
        </p>

        <p v-if="uploadStatus === 'error'" style="color: red;">
        ❌ Upload failed. Please try again.
        </p>

    </div>
  </div>
</template>

<script setup>
    import { ref } from "vue";
    import { onMounted, onUnmounted } from "vue";

    const emit = defineEmits(["close"]);

    const steps = ["LEFT", "RIGHT", "UP", "DOWN", "OPEN", "CLOSED"];

    const currentStep = ref(0);
    const isCapturing = ref(false);
    const countdown = ref(3);
    const isCountingDown = ref(false);
    const completed = ref(false);

    const videoRef = ref(null);

    const isUploading = ref(false);
    const uploadStatus = ref(""); // "", "success", "error"

    const collectedData = ref({
    LEFT: [],
    RIGHT: [],
    UP: [],
    DOWN: [],
    OPEN: [],
    CLOSED: []
    });

    let stream = null;

    const startCamera = async () => {
    stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { min: 640, ideal: 1920, max: 1920 },
          height: { min: 480, ideal: 1080, max: 1080 },
          frameRate: { ideal: 30 }
        }
      });

    videoRef.value.srcObject = stream;
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    };

    onMounted(() => {
        startCamera();
    });

    onUnmounted(() => {
        stopCamera();
    });

    const startCountdown = () => {
    isCountingDown.value = true;
    countdown.value = 3;

    const interval = setInterval(() => {
        countdown.value--;

        if (countdown.value === 0) {
            clearInterval(interval);
            isCountingDown.value = false;
            startCapture();
            }
        }, 1000);
    };

    // TODO: make the sound works
    const playDing = () => {
        const audio = new Audio("/sounds/ding.mp3");
        audio.play();
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
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0);
        ctx.restore();

        frames.push(canvas.toDataURL("image/jpeg"));
        count++;

        if (count >= 100) { // ~100 frames per class
        clearInterval(interval);

        playDing(); // 👈 ADD HERE

        collectedData.value[direction] = frames;
        isCapturing.value = false;

        nextStep();
        }
    }, 50); // ~10 fps
    };

    const nextStep = () => {
    currentStep.value++;

    if (currentStep.value >= steps.length) {
        finishCalibration();
    }
    };

    const base64ToBlob = (base64) => {
        const arr = base64.split(",");
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);

        let n = bstr.length;
        const u8arr = new Uint8Array(n);

        while (n--) {
            u8arr[n] = bstr.charCodeAt(n);
        }

        return new Blob([u8arr], { type: mime });
    };

    const buildFormData = (data) => {
        const formData = new FormData();

        Object.keys(data).forEach((label) => {
            data[label].forEach((base64, index) => {
            const blob = base64ToBlob(base64);

            formData.append(
                "files",
                blob,
                `${label}_${index}.jpg`
            );

            formData.append("labels", label);
            });
        });

        return formData;
    };

    const finishCalibration = async () => {
        isUploading.value = true;
        uploadStatus.value = "";

        try {
            const formData = buildFormData(collectedData.value);

            const res = await fetch("http://localhost:8000/calibrate", {
            method: "POST",
            body: formData
            });

            if (!res.ok) {
            throw new Error("Upload failed");
            }

            uploadStatus.value = "success";

            // small delay so user sees it
            setTimeout(() => {
            stopCamera();
            emit("close");
            }, 3000);

        } catch (err) {
            console.error(err);
            uploadStatus.value = "error";
        } finally {
            isUploading.value = false;
        }
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
  transform: scaleX(-1); /* mirror */
}

.camera-wrapper {
  position: relative;
  display: inline-block;
}

/* .face-outline {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 200px;
  height: 260px;

  transform: translate(-50%, -50%);
  border: 2px dashed #3b82f6;
  border-radius: 50% / 60%;
  pointer-events: none;
} */

button {
  margin-top: 10px;
}
</style>