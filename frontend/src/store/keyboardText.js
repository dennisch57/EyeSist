import { ref, computed } from "vue";

const text = ref([]);
const selectedKey = ref("");

export function keyboardText() {

  const addKey = (key) => {
    selectedKey.value = key;

    if (key === "⌫") {
      text.value.pop();
      return;
    }

    if (key === "SPACE") {
      text.value.push({ char: " ", finalized: true });
      return;
    }

    // ✅ everything else = direct insert
    text.value.push({ char: key, finalized: true });
  };

  // helper for text output
  const displayText = computed(() =>
    text.value.map(t => t.char).join("")
  );

  const resetMultiTap = () => {
    
  };

    const setText = (newText) => {
    text.value = newText.split("").map(c => ({
      char: c,
      finalized: true
    }));
  };

  return {
    text,
    displayText,
    selectedKey,
    addKey,
    resetMultiTap,
    setText
  };
}