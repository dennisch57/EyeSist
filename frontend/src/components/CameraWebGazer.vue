<template>
  <div class="camera-card">
    <p class="title">WebGazer Eye Tracking</p>

    <div class="camera-box">
      <!-- WebGazer injects its own <video> into <body>; we just show status -->
      <div class="overlay" :class="{ active: isActive }">
        <p v-if="!isActive">Start tracking to begin</p>
        <p v-else-if="isCalibrating">
          🎯 Calibrating... click each dot while looking at it
        </p>
        <p v-else>
          👁 Tracking active — gaze at a key and hold to press
        </p>
      </div>

      <!-- Dwell progress ring shown near gaze point -->
      <div
        v-if="isActive && dwellTarget"
        class="dwell-indicator"
        :style="dwellIndicatorStyle"
      >
        <svg viewBox="0 0 36 36">
          <circle cx="18" cy="18" r="15" fill="none" stroke="#334155" stroke-width="3"/>
          <circle
            cx="18" cy="18" r="15"
            fill="none"
            stroke="#3b82f6"
            stroke-width="3"
            stroke-dasharray="94.2"
            :stroke-dashoffset="94.2 - (94.2 * dwellProgress)"
            stroke-linecap="round"
            transform="rotate(-90 18 18)"
          />
        </svg>
      </div>
    </div>

    <button class="btn" :disabled="isActive" @click="startTracking">
      {{ isActive ? 'Tracking Active' : 'Start Eye Tracking' }}
    </button>

    <button class="calibrate-btn" :disabled="!isActive" @click="startCalibration">
      Calibrate (9-point)
    </button>

    <button class="stop-btn" :disabled="!isActive" @click="stopTracking">
      Stop Tracking
    </button>

    <!-- Calibration overlay: rendered at root level via teleport -->
    <Teleport to="body">
      <div v-if="isCalibrating" class="calibration-overlay">
        <p class="calibration-hint">
          Look at each dot and <strong>click it</strong> ({{ calibrationIndex + 1 }}/{{ calibrationPoints.length }})
        </p>
        <div
          v-for="(pt, i) in calibrationPoints"
          :key="i"
          class="cal-dot"
          :class="{
            current: i === calibrationIndex,
            done: i < calibrationIndex
          }"
          :style="{ left: pt.x + '%', top: pt.y + '%' }"
          @click="handleCalDotClick(i)"
        >
          <!-- Show click count on the active dot -->
          <span v-if="i === calibrationIndex" class="dot-count">
            {{ currentDotClicks }}/{{ CLICKS_PER_DOT }}
          </span>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, computed, onUnmounted } from 'vue'
import { useEyeStore } from '@/store/eyeStore'

const { direction } = useEyeStore()

// ── State ──────────────────────────────────────────────────────────────────
const isActive = ref(false)
const isCalibrating = ref(false)
const calibrationIndex = ref(0)

// Dwell-to-press state
const DWELL_MS = 800          // ms the user must gaze at a key to press it
const dwellTarget = ref(null) // the DOM element currently being dwelt on
const dwellStart = ref(null)
const dwellProgress = ref(0)  // 0–1
const dwellPos = ref({ x: 0, y: 0 }) // screen coords for the indicator

const dwellIndicatorStyle = computed(() => ({
  left: `${dwellPos.value.x}px`,
  top:  `${dwellPos.value.y}px`,
}))

// ── Calibration points (% of screen) ──────────────────────────────────────
const calibrationPoints = [
  { x: 10, y: 10 }, { x: 50, y: 10 }, { x: 90, y: 10 },
  { x: 10, y: 50 }, { x: 50, y: 50 }, { x: 90, y: 50 },
  { x: 10, y: 90 }, { x: 50, y: 90 }, { x: 90, y: 90 },
]

// ── WebGazer lifecycle ─────────────────────────────────────────────────────
const startTracking = async () => {
  const wg = window.webgazer
  if (!wg) return

  await wg.setGazeListener(onGaze).begin()

  wg.applyKalmanFilter(true)

  wg.removeMouseEventListeners()

  wg.showVideo(false)
  wg.showFaceOverlay(false)
  wg.showPredictionPoints(true)

  isActive.value = true
}

const stopTracking = () => {
  window.webgazer?.end()
  isActive.value = false
  isCalibrating.value = false
  dwellTarget.value = null
}

const startCalibration = () => {
  calibrationIndex.value = 0
  isCalibrating.value = true
}

// ── Calibration logic ─────────────────────────────────────────────────────
const CLICKS_PER_DOT = 5
const currentDotClicks = ref(0)

const handleCalDotClick = (i) => {
  if (i !== calibrationIndex.value) return
  if (calibrationIndex.value >= calibrationPoints.length) return

  const px = (calibrationPoints[i].x / 100) * window.innerWidth
  const py = (calibrationPoints[i].y / 100) * window.innerHeight

  // Record this click as training data
  window.webgazer.recordScreenPosition(px, py, 'click')
  currentDotClicks.value++

  // Only advance to next dot after enough clicks
  if (currentDotClicks.value >= CLICKS_PER_DOT) {
    currentDotClicks.value = 0
    calibrationIndex.value++

    if (calibrationIndex.value === calibrationPoints.length) {
      isCalibrating.value = false
      calibrationIndex.value = 0

      const data = window.webgazer.getRegression()
      console.log('Regression data after calibration:', data)
    }
  }
}

const BUFFER_SIZE = 12  // increase for smoother, decrease for snappier
const gazeBuffer = { x: [], y: [] }

const pushToBuffer = (x, y) => {
  gazeBuffer.x.push(x)
  gazeBuffer.y.push(y)
  if (gazeBuffer.x.length > BUFFER_SIZE) {
    gazeBuffer.x.shift()
    gazeBuffer.y.shift()
  }
}

const getBufferAverage = () => ({
  x: gazeBuffer.x.reduce((a, b) => a + b, 0) / gazeBuffer.x.length,
  y: gazeBuffer.y.reduce((a, b) => a + b, 0) / gazeBuffer.y.length,
})

// ── Gaze listener ──────────────────────────────────────────────────────────
let animFrame = null

const SMOOTHING = 0.06     // lower = smoother
const MOVE_THRESHOLD = 20  // higher = more stable while holding gaze
const smoothed = ref({ x: window.innerWidth / 2, y: window.innerHeight / 2 })

const onGaze = (data) => {
  if (!data || isCalibrating.value) return

  // Layer 1: rolling average to kill outlier spikes
  pushToBuffer(data.x, data.y)
  const avg = getBufferAverage()

  // Layer 2: exponential smoothing on top of the average
  smoothed.value.x = smoothed.value.x + SMOOTHING * (avg.x - smoothed.value.x)
  smoothed.value.y = smoothed.value.y + SMOOTHING * (avg.y - smoothed.value.y)

  // Layer 3: dead zone — only commit position if moved enough
  const dx = smoothed.value.x - dwellPos.value.x
  const dy = smoothed.value.y - dwellPos.value.y
  if (Math.sqrt(dx * dx + dy * dy) > MOVE_THRESHOLD) {
    dwellPos.value = { ...smoothed.value }
  }

  const { x, y } = dwellPos.value

  // Hit test + dwell (unchanged)
  const el = document.elementFromPoint(x, y)
  const keyEl = el?.closest('[data-webgazer-key]') ?? null

  if (keyEl && keyEl === dwellTarget.value) {
    const elapsed = Date.now() - dwellStart.value
    dwellProgress.value = Math.min(elapsed / DWELL_MS, 1)
    if (elapsed >= DWELL_MS) {
      keyEl.click()
      dwellStart.value = Date.now() + DWELL_MS * 0.5
      dwellProgress.value = 0
    }
  } else {
    dwellTarget.value = keyEl
    dwellStart.value = Date.now()
    dwellProgress.value = 0
  }
}

// ── Cleanup ────────────────────────────────────────────────────────────────
onUnmounted(() => {
  stopTracking()
  if (animFrame) cancelAnimationFrame(animFrame)
})
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
  height: 120px;
  background: #0f172a;
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.overlay {
  color: #64748b;
  font-size: 14px;
  text-align: center;
  padding: 16px;
  transition: color 0.3s;
}
.overlay.active { color: #cbd5f5; }

/* Dwell ring — fixed to viewport coords */
.dwell-indicator {
  position: fixed;
  width: 36px;
  height: 36px;
  transform: translate(-50%, -50%);
  pointer-events: none;
  z-index: 9999;
  transition: opacity 0.1s;
}

.btn, .calibrate-btn, .stop-btn {
  margin-top: 12px;
  width: 100%;
  border: none;
  border-radius: 10px;
  padding: 12px 14px;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.15s ease, opacity 0.15s ease;
}

.btn         { background: #22c55e; color: #0f172a; }
.calibrate-btn { background: #3b82f6; color: white; }
.stop-btn    { background: #ef4444; color: white; }

.btn:disabled, .calibrate-btn:disabled, .stop-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* ── Calibration overlay ── */
.calibration-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.85);
  z-index: 10000;
}

.calibration-hint {
  position: absolute;
  top: 16px;
  left: 50%;
  transform: translateX(-50%);
  color: white;
  font-size: 16px;
  text-align: center;
}

.cal-dot {
  position: absolute;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  background: #334155;
  border: 3px solid #64748b;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.cal-dot.current {
  background: #f59e0b;
  border-color: #fbbf24;
  box-shadow: 0 0 16px #f59e0b;
  animation: pulse 1s infinite;
}

.cal-dot.done {
  background: #22c55e;
  border-color: #16a34a;
}

.dot-count {
  color: white;
  font-size: 9px;
  font-weight: bold;
  pointer-events: none;
}

@keyframes pulse {
  0%, 100% { transform: translate(-50%, -50%) scale(1); }
  50%       { transform: translate(-50%, -50%) scale(1.25); }
}
</style>