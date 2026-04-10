import { ref } from "vue";

const text = ref("");
const selectedKey = ref("");

export function keyboardText() {
  const addKey = (key) => {
    selectedKey.value = key;

    if (key === "⌫") {
      text.value = text.value.slice(0, -1);
    } else if (key === "SPACE") {
      text.value += " ";
    } else {
      text.value += key;
    }
  };

  return {
    text,
    selectedKey,
    addKey
  };
}