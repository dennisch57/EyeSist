<template>
  <div class="output-card">
    <div class="header">
        <p class="title">Text Output</p>
        <button class="speak-btn" @click="speakText">🔊 Speak</button>
    </div>

    <div class="text-box">
      <span
        v-for="(t, index) in text"
        :key="index"
        :class="t.finalized ? 'final' : 'preview'"
      >
        {{ t.char }}
      </span>
    </div>

    <p class="char-count">{{ text.length }} characters</p>
  </div>
</template>

<script setup>
import { keyboardText } from "@/store/keyboardText";

const { text } = keyboardText();

const speakText = () => {
  if (!text.value) return;

  const speech = new SpeechSynthesisUtterance(text.value);
  window.speechSynthesis.speak(speech);
};
</script>

<style scoped>
.output-card {
  background: #1e293b;
  padding: 16px;
  border-radius: 12px;
  width: 400px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.title {
  color: #cbd5f5;
}

.text-box {
  width: 100%;
  height: 215px;
  background: #0f172a;
  border-radius: 8px;
  padding: 10px;
  color: white;
  word-wrap: break-word;
}

/* finalized text */
.final {
  color: white;
}

/* preview text */
.preview {
  color: #94a3b8; /* grey */
}

.char-count {
  text-align: right;
  margin-top: 8px;
  color: #94a3b8;
  font-size: 12px;
}

.speak-btn {
  background: #22c55e;
  border: none;
  padding: 6px 12px;
  border-radius: 8px;
  color: white;
  cursor: pointer;
}
</style>