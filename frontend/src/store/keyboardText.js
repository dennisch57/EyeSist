import { ref, computed } from "vue";

const text = ref([]);
const selectedKey = ref("");

let lastKey = null;
let tapIndex = 0;
let timer = null;

const multiTapMap = {
  "2": ["a", "b", "c"],
  "3": ["d", "e", "f"],
  "4": ["g", "h", "i"],
  "5": ["j", "k", "l"],
  "6": ["m", "n", "o"],
  "7": ["p", "q", "r", "s"],
  "8": ["t", "u", "v"],
  "9": ["w", "x", "y", "z"],
  "0": [" "]
};

export function keyboardText() {

  const addKey = (key, layout) => {
    selectedKey.value = key;

    if (key === "⌫") {
      text.value.pop();
      return;
    }

    if (key === "SPACE") {
      text.value.push({ char: " ", finalized: true });
      return;
    }

    if (layout === "NOKIA" && multiTapMap[key]) {
    handleMultiTap(key);
    return;
  }

    // QWERTY
    text.value.push({ char: key, finalized: true });
  };

  const handleMultiTap = (key) => {
  const letters = multiTapMap[key];

  // 👉 DIFFERENT key pressed → finalize previous char
  if (key !== lastKey && text.value.length > 0) {
    text.value[text.value.length - 1].finalized = true;
  }

  if (key === lastKey && text.value.length > 0) {
    tapIndex = (tapIndex + 1) % letters.length;

    text.value[text.value.length - 1] = {
      char: letters[tapIndex],
      finalized: false
    };
  } else {
    tapIndex = 0;

    text.value.push({
      char: letters[0],
      finalized: false
    });
  }

  lastKey = key;

  // reset timer
  if (timer) clearTimeout(timer);

  timer = setTimeout(() => {
    if (text.value.length > 0) {
      text.value[text.value.length - 1].finalized = true;
    }

    lastKey = null;
    tapIndex = 0;
  }, 1000);
};

  // helper for text output
  const displayText = computed(() =>
    text.value.map(t => t.char).join("")
  );

  const resetMultiTap = () => {
    lastKey = null;
    tapIndex = 0;

    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
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