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

    <div class="suggestions">
      <button
        v-for="word in suggestions"
        :key="word"
        class="suggestion-btn"
        @click="selectSuggestion(word)"
      >
        {{ word }}
      </button>
    </div>

    <p class="char-count">{{ text.length }} characters</p>
  </div>
</template>

<script setup>
  import { computed } from "vue";
  import { keyboardText } from "@/store/keyboardText";
  import { dictionary } from "@/store/dictionary";

  const { text, displayText, setText } = keyboardText();

  const currentWord = computed(() => {
    const words = displayText.value.split(" ");
    return words[words.length - 1]
  });

  const suggestions = computed(() => {
    if (!currentWord.value || currentWord.value.length < 2) return [];

    return dictionary
      .filter(word => word.startsWith(currentWord.value))
      .slice(0, 3);
  });

  const selectSuggestion = (word) => {
    const words = displayText.value.split(" ");

    words[words.length - 1] = word;

    setText(words.join(" ") + " ");
  };

  const speakText = () => {
    if (!text.value) return;

    const speech = new SpeechSynthesisUtterance(displayText.value);
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

.suggestions {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-bottom: 10px;
}

.suggestion-btn {
  min-width: 100px;
  padding: 12px;
  font-size: 18px;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 10px;
  cursor: pointer;
}

.suggestion-btn:hover {
  background: #2563eb;
}
</style>