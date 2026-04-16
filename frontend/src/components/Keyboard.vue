<template>
  <div class="keyboard-card">
    <!-- Layout Selection -->
    <div class="layout-buttons">
      <button
        v-for="layout in layouts"
        :key="layout"
        :class="['btn', selectedLayout === layout ? 'active' : '']"
        @click="selectedLayout = layout"
      >
        {{ layout }}
      </button>
    </div>

    <!-- Keyboard -->
    <div :class="['keys', selectedLayout.toLowerCase()]">
      <button
        v-for="(key, index) in currentKeys"
        :key="key.main || key"
        :class="['key', activeIndex === index ? 'active-key' : '']"
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

    <!-- Bottom row -->
    <div class="bottom-row">
      <button class="key space" @click="pressKey('SPACE')">
        SPACE
      </button>
      <button class="key delete" @click="pressKey('⌫')">
        ⌫
      </button>
    </div>

    <!-- Selected Key Display -->
    <p class="selected">Selected Key: {{ selectedKey }}</p>

    <p>Eye Direction: {{ direction }}</p>
  </div>
</template>

<script setup>
  import { ref, computed } from "vue";
  import { watch } from "vue";
  import { keyboardText } from "@/store/keyboardText";
  import { useEyeStore } from "@/store/eyeStore";
  import { onMounted, onUnmounted } from "vue";

  const { addKey, selectedKey, resetMultiTap } = keyboardText();
  const { direction } = useEyeStore();

  const selectedLayout = ref("QWERTY");
  const activeIndex = ref(0);

  const layouts = ["QWERTY", "NOKIA"];

  const qwertyKeys = [
    "1","2","3","4","5","6","7","8","9","0",
    "Q","W","E","R","T","Y","U","I","O","P",
    "A","S","D","F","G","H","J","K","L",
    "Z","X","C","V","B","N","M",",",".","/"
  ];

  const nokiaKeys = [
    { main: "1", sub: "" },
    { main: "2", sub: "ABC" },
    { main: "3", sub: "DEF" },
    { main: "4", sub: "GHI" },
    { main: "5", sub: "JKL" },
    { main: "6", sub: "MNO" },
    { main: "7", sub: "PQRS" },
    { main: "8", sub: "TUV" },
    { main: "9", sub: "WXYZ" },
    { main: "*", sub: "" },
    { main: "0", sub: "␣" },
    { main: "#", sub: "" }
  ];

  const currentKeys = computed(() => {
    if (selectedLayout.value === "NOKIA") return nokiaKeys;
    return qwertyKeys;
  });

  const pressKey = (key) => {
    addKey(key, selectedLayout.value);
  };

  const moveSelection = (direction) => {
    const cols = selectedLayout.value === "NOKIA" ? 3 : 10;

    // update activeIndex based on direction
    if (direction === "right") activeIndex.value++;
    if (direction === "left") activeIndex.value--;
    if (direction === "down") activeIndex.value += cols;
    if (direction === "up") activeIndex.value -= cols;

    // clamp activeIndex within bounds
    if (activeIndex.value < 0) activeIndex.value = 0;
    if (activeIndex.value >= currentKeys.value.length)
      activeIndex.value = currentKeys.value.length - 1;
  };

  const handleKeydown = (e) => {
    if (e.key === "ArrowRight") moveSelection("right");
    if (e.key === "ArrowLeft") moveSelection("left");
    if (e.key === "ArrowDown") moveSelection("down");
    if (e.key === "ArrowUp") moveSelection("up");

    // simulate blink/select
    if (e.key === "Enter") {
      const key = currentKeys.value[activeIndex.value];
      pressKey(key.main || key);
    }
  };

  onMounted(() => window.addEventListener("keydown", handleKeydown));
  onUnmounted(() => window.removeEventListener("keydown", handleKeydown));

  watch(selectedLayout, () => {
    resetMultiTap();
  });

  watch(direction, (dir) => {
    if (dir === "closed") {
    const key = currentKeys.value[activeIndex.value];
    pressKey(key.main || key);
  } else if (dir !== "open") {
      moveSelection(dir);
    }
  });
  </script>

  <style scoped>
  .keyboard-card {
    background: #1e293b;
    padding: 16px;
    border-radius: 12px;
    margin-top: 20px;
  }

  .layout-buttons {
    display: flex;
    justify-content: space-between;
    margin-bottom: 15px;
  }

  .layout-buttons .btn {
    width: 50%;
  }

  .btn {
    padding: 6px 12px;
    background: #334155;
    border: none;
    border-radius: 8px;
    color: white;
    cursor: pointer;
  }

  .active {
    background: #3b82f6;
  }

  .active-key {
    background: #3b82f6;
    box-shadow: 0 0 10px #3b82f6;
  }

  .keys {
    display: grid;
    gap: 8px;
    justify-content: center; /* 👈 centers entire grid */
  }

  /* QWERTY = fill row */
  .keys.qwerty {
    grid-template-columns: repeat(10, 50px);
  }

  .keys.qwerty .key {
    width: 100%;
    height: 50px;
  }

  /* NOKIA = fixed grid */
  .keys.nokia {
    grid-template-columns: repeat(3, 50px);
  }

  .keys.nokia .key {
    width: 100%;
    height: 50px;
  }

  .key {
    width: 50px;
    height: 50px;
    background: #334155;
    border: none;
    border-radius: 8px;
    color: white;
    cursor: pointer;

    display: flex;
    align-items: center;
    justify-content: center;
  }

  .nokia-key {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .nokia-key .main {
    font-size: 16px;
    font-weight: bold;
  }

  .nokia-key .sub {
    font-size: 10px;
    color: #94a3b8;
  }

  /* Bottom row */
  .bottom-row {
    display: flex;
    gap: 10px;
    margin-top: 10px;
  }

  .bottom-row .key {
    aspect-ratio: auto;
    height: 50px;
  }

  .space {
    flex: 2;
  }

  .delete {
    flex: 1;
  }

  /* Selected text */
  .selected {
    margin-top: 10px;
    color: #94a3b8;
  }

  /* TODO: Make responsive */
  /* @media (max-width: 768px) {
    .key {
      width: 40px;
      height: 40px;
    }

    .keys.nokia {
      grid-template-columns: repeat(3, 40px);
    }
  } */
</style>