<template>
  <div class="output-card">
    <div class="header">
      <p class="title">Text Output</p>
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

.final   { color: white; }
.preview { color: #94a3b8; }

.char-count {
  text-align: right;
  margin-top: 8px;
  color: #94a3b8;
  font-size: 12px;
}
</style>