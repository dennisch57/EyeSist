<template>
  <div class="keyboard-card">

    <!-- Row 0: Layout switcher + Speak button -->
    <div class="layout-buttons">
      <button
        v-for="(layout, i) in layouts"
        :key="layout"
        :class="['btn', selectedLayout === layout ? 'active' : '', isFocused('layout', i) ? 'active-key' : '']"
        @click="selectedLayout = layout"
      >
        {{ layout }}
      </button>
      <button
        :class="['btn', 'speak-btn', isFocused('layout', layouts.length) ? 'active-key' : '']"
        @click="speakText"
      >
        🔊 Speak
      </button>
    </div>

    <!-- Row 1: Autocomplete suggestions (only when available) -->
    <div v-if="suggestions.length > 0" class="suggestions-row">
      <button
        v-for="(word, i) in suggestions"
        :key="word"
        :class="['suggestion-btn', isFocused('suggestions', i) ? 'active-key' : '']"
        @click="selectSuggestion(word)"
      >
        {{ word }}
      </button>
    </div>

    <!-- Row 2: Main keys grid -->
    <div :class="['keys', selectedLayout.toLowerCase()]">
      <button
        v-for="(key, index) in currentKeys"
        :key="key.main || key"
        :class="['key', isFocused('keys', index) ? 'active-key' : '']"
        @click="pressKey(key.main || key)"
      >
        <template v-if="selectedLayout === 'NOKIA'">
          <div class="nokia-key">
            <span class="main">{{ key.main }}</span>
            <span class="sub">{{ key.sub }}</span>
          </div>
        </template>
        <template v-else>
          {{ key }}
        </template>
      </button>
    </div>

    <!-- Row 3: Space + Backspace -->
    <div class="bottom-row">
      <!-- NEW: Clear button -->
      <button
        :class="['key', 'clear', isFocused('bottom', 0) ? 'active-key' : '']"
        @click="clearText"
      >
        Clear Text
      </button>

      <button
        :class="['key', 'space', isFocused('bottom', 1) ? 'active-key' : '']"
        @click="pressKey('SPACE')"
      >
        SPACE
      </button>

      <button
        :class="['key', 'delete', isFocused('bottom', 2) ? 'active-key' : '']"
        @click="pressKey('⌫')"
      >
        Backspace
      </button>
    </div>

    <p class="selected">Selected Key: {{ selectedKey }}</p>
    <p class="eye-dir">Eye Direction: {{ direction }}</p>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from "vue";
import { keyboardText } from "@/store/keyboardText";
import { useEyeStore } from "@/store/eyeStore";
import { dictionary } from "@/store/dictionary";

const { addKey, selectedKey, text, displayText, setText, resetMultiTap } = keyboardText();
const { direction } = useEyeStore();

// Layout
const selectedLayout = ref("QWERTY");
const layouts = ["QWERTY", "NOKIA"];

// Keys
const qwertyKeys = [
  "1","2","3","4","5","6","7","8","9","0",
  "q","w","e","r","t","y","u","i","o","p",
  "a","s","d","f","g","h","j","k","l","z",
  "x","c","v","b","n","m",",",".","/"
];
const qwertyRowLengths = [10, 10, 10, 9];

const nokiaKeys = [
  { main: "1", sub: "" }, { main: "2", sub: "abc" }, { main: "3", sub: "def" },
  { main: "4", sub: "ghi" }, { main: "5", sub: "jkl" }, { main: "6", sub: "mno" },
  { main: "7", sub: "pqrs" }, { main: "8", sub: "tuv" }, { main: "9", sub: "wxyz" },
  { main: "*", sub: "" }, { main: "0", sub: "␣" }, { main: "#", sub: "" },
];

const currentKeys = computed(() =>
  selectedLayout.value === "NOKIA" ? nokiaKeys : qwertyKeys
);
const cols = computed(() => (selectedLayout.value === "NOKIA" ? 3 : 10));
const keysRowCount = computed(() =>
  selectedLayout.value === "NOKIA" ? 4 : qwertyRowLengths.length
);

// Autocomplete
const currentWord = computed(() => {
  const words = displayText.value.split(" ");
  return words[words.length - 1];
});
const suggestions = computed(() => {
  if (!currentWord.value || currentWord.value.length < 2) return [];
  return dictionary.filter(w => w.startsWith(currentWord.value)).slice(0, 3);
});
const selectSuggestion = (word) => {
  const words = displayText.value.split(" ");
  words[words.length - 1] = word;
  setText(words.join(" ") + " ");
};

// Speak
const speakText = () => {
  if (!text.value.length) return;
  const speech = new SpeechSynthesisUtterance(displayText.value);
  window.speechSynthesis.speak(speech);
};

// Focus state
// Sections top-to-bottom: 'layout' -> 'suggestions' (optional) -> 'keys' -> 'bottom'
const focusedSection = ref("keys");
const cursor = ref({
  layout: 0,      // 0..layouts.length (last = Speak)
  suggestions: 0,
  keys: 0,
  bottom: 0, // now 0=clear, 1=space, 2=delete
});

const isFocused = (section, index) =>
  focusedSection.value === section && cursor.value[section] === index;

// Grid helpers
const getRowCol = (flatIdx) => {
  if (selectedLayout.value === "NOKIA") {
    return { row: Math.floor(flatIdx / 3), col: flatIdx % 3 };
  }
  let rem = flatIdx;
  for (let r = 0; r < qwertyRowLengths.length; r++) {
    if (rem < qwertyRowLengths[r]) return { row: r, col: rem };
    rem -= qwertyRowLengths[r];
  }
  return { row: qwertyRowLengths.length - 1, col: rem };
};

const getIndex = (row, col) => {
  if (selectedLayout.value === "NOKIA") {
    return Math.max(0, Math.min(row * 3 + col, nokiaKeys.length - 1));
  }
  let base = 0;
  for (let r = 0; r < row && r < qwertyRowLengths.length; r++) base += qwertyRowLengths[r];
  const rowLen = qwertyRowLengths[Math.min(row, qwertyRowLengths.length - 1)];
  return base + Math.max(0, Math.min(col, rowLen - 1));
};

// Clear text
const clearText = () => {
  setText("");        // wipe everything
  resetMultiTap();    // prevent stuck preview letter
};

// Section order
const prevSection = (sec) => {
  if (sec === "bottom")      return "keys";
  if (sec === "keys")        return suggestions.value.length > 0 ? "suggestions" : "layout";
  if (sec === "suggestions") return "layout";
  return null;
};
const nextSection = (sec) => {
  if (sec === "layout")      return suggestions.value.length > 0 ? "suggestions" : "keys";
  if (sec === "suggestions") return "keys";
  if (sec === "keys")        return "bottom";
  return null;
};

// Navigation
const moveSelection = (dir) => {
  const sec = focusedSection.value;

  if (sec === "layout") {
    const max = layouts.length; // index of Speak button
    if (dir === "left")  cursor.value.layout = Math.max(0, cursor.value.layout - 1);
    if (dir === "right") cursor.value.layout = Math.min(max, cursor.value.layout + 1);
    if (dir === "down") {
      const next = nextSection("layout");
      if (next === "suggestions") {
        focusedSection.value = "suggestions";
        cursor.value.suggestions = Math.min(cursor.value.suggestions, suggestions.value.length - 1);
      } else {
        const targetCol = Math.round((cursor.value.layout / max) * (cols.value - 1));
        cursor.value.keys = getIndex(0, targetCol);
        focusedSection.value = "keys";
      }
    }
  }

  else if (sec === "suggestions") {
    const max = suggestions.value.length - 1;
    if (dir === "left")  cursor.value.suggestions = Math.max(0, cursor.value.suggestions - 1);
    if (dir === "right") cursor.value.suggestions = Math.min(max, cursor.value.suggestions + 1);
    if (dir === "up")    focusedSection.value = "layout";
    if (dir === "down") {
      const targetCol = max > 0
        ? Math.round((cursor.value.suggestions / max) * (cols.value - 1))
        : 0;
      cursor.value.keys = getIndex(0, targetCol);
      focusedSection.value = "keys";
    }
  }

  else if (sec === "keys") {
    const { row, col } = getRowCol(cursor.value.keys);
    if (dir === "right") {
      const rowLen = selectedLayout.value === "NOKIA" ? 3 : qwertyRowLengths[row];
      if (col < rowLen - 1) cursor.value.keys = getIndex(row, col + 1);
    }
    else if (dir === "left") {
      if (col > 0) cursor.value.keys = getIndex(row, col - 1);
    }
    else if (dir === "up") {
      if (row === 0) {
        const prev = prevSection("keys");
        if (prev === "suggestions") {
          const max = suggestions.value.length - 1;
          cursor.value.suggestions = max > 0
            ? Math.min(Math.round((col / (cols.value - 1)) * max), max)
            : 0;
          focusedSection.value = "suggestions";
        } else {
          const max = layouts.length;
          cursor.value.layout = Math.round((col / (cols.value - 1)) * max);
          focusedSection.value = "layout";
        }
      } else {
        const newRowLen = selectedLayout.value === "NOKIA" ? 3 : qwertyRowLengths[row - 1];
        cursor.value.keys = getIndex(row - 1, Math.min(col, newRowLen - 1));
      }
    }
    else if (dir === "down") {
      if (row >= keysRowCount.value - 1) {
        cursor.value.bottom = col >= cols.value - 2 ? 1 : 0;
        focusedSection.value = "bottom";
      } else {
        const newRowLen = selectedLayout.value === "NOKIA" ? 3 : qwertyRowLengths[row + 1];
        cursor.value.keys = getIndex(row + 1, Math.min(col, newRowLen - 1));
      }
    }
  }

  else if (sec === "bottom") {
    if (dir === "left") {
      cursor.value.bottom = Math.max(0, cursor.value.bottom - 1);
    }
    if (dir === "right") {
      cursor.value.bottom = Math.min(2, cursor.value.bottom + 1);
    }
    if (dir === "up") {
      const lastRow = keysRowCount.value - 1;
      const lastRowLen = selectedLayout.value === "NOKIA" ? 3 : qwertyRowLengths[lastRow];

      let targetCol;
      if (cursor.value.bottom === 0) targetCol = 0; // clear
      else if (cursor.value.bottom === 1) targetCol = Math.floor(lastRowLen / 2); // space
      else targetCol = lastRowLen - 1; // delete

      cursor.value.keys = getIndex(lastRow, targetCol);
      focusedSection.value = "keys";
    }
  }
};

// Activate focused element
const pressActive = () => {
  const sec = focusedSection.value;
  if (sec === "layout") {
    if (cursor.value.layout < layouts.length) {
      selectedLayout.value = layouts[cursor.value.layout];
    } else {
      speakText();
    }
  } else if (sec === "suggestions") {
    const word = suggestions.value[cursor.value.suggestions];
    if (word) selectSuggestion(word);
  } else if (sec === "keys") {
    const key = currentKeys.value[cursor.value.keys];
    pressKey(key.main ?? key);
  } else if (sec === "bottom") {
    if (cursor.value.bottom === 0) clearText();
    else if (cursor.value.bottom === 1) pressKey("SPACE");
    else pressKey("⌫");
  }
};

const pressKey = (key) => addKey(key, selectedLayout.value);

// Keyboard events
const handleKeydown = (e) => {
  const map = { ArrowRight: "right", ArrowLeft: "left", ArrowDown: "down", ArrowUp: "up" };
  if (map[e.key]) { e.preventDefault(); moveSelection(map[e.key]); return; }
  if (e.key === "Enter") { e.preventDefault(); pressActive(); }
};

onMounted(() => window.addEventListener("keydown", handleKeydown));
onUnmounted(() => window.removeEventListener("keydown", handleKeydown));

watch(selectedLayout, () => {
  resetMultiTap();
  cursor.value.keys = 0;
  focusedSection.value = "keys";
});

watch(suggestions, (newVal) => {
  if (newVal.length === 0 && focusedSection.value === "suggestions") {
    focusedSection.value = "keys";
  }
});

watch(direction, (dir) => {
  if (dir === "closed") {
    pressActive();
  } else if (["left", "right", "up", "down"].includes(dir)) {
    moveSelection(dir);
  }
});
</script>

<style scoped>
.keyboard-card {
  background: #1e293b;
  padding: 16px;
  border-radius: 12px;
}

.layout-buttons {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.btn {
  flex: 1;
  padding: 8px 12px;
  background: #334155;
  border: none;
  border-radius: 8px;
  color: white;
  cursor: pointer;
  transition: background 0.15s;
  font-size: 14px;
}
.btn.active { background: #3b82f6; }
.speak-btn  { background: #22c55e; }
.speak-btn:hover { background: #16a34a; }

.suggestions-row {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-bottom: 12px;
}

.suggestion-btn {
  min-width: 100px;
  padding: 10px 14px;
  font-size: 16px;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: background 0.15s;
}
.suggestion-btn:hover { background: #2563eb; }

.keys {
  display: grid;
  gap: 6px;
  justify-content: center;
}

.keys.qwerty { grid-template-columns: repeat(10, 40px); }
.keys.qwerty .key { width: 40px; height: 40px; font-size: 14px; }

.keys.nokia  { grid-template-columns: repeat(3, 40px); }
.keys.nokia  .key { width: 40px; height: 40px; }

.key {
  width: 40px;
  height: 40px;
  background: #334155;
  border: none;
  border-radius: 6px;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s;
}

.nokia-key { display: flex; flex-direction: column; align-items: center; justify-content: center; }
.nokia-key .main { font-size: 14px; font-weight: bold; }
.nokia-key .sub  { font-size: 9px; color: #94a3b8; }

.bottom-row {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}
.bottom-row .key { aspect-ratio: auto; height: 40px; }
.space  { flex: 2; }
.delete { flex: 1; }

.clear {
  flex: 1;
  background: #ef4444;
}

.clear:hover {
  background: #dc2626;
}

.active-key {
  background: #3b82f6 !important;
  box-shadow: 0 0 10px #3b82f6;
}

.selected, .eye-dir {
  margin-top: 8px;
  color: #94a3b8;
  font-size: 12px;
}
</style>