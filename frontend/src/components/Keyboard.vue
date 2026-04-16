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
        v-for="key in currentKeys"
        :key="key.main || key"
        class="key"
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
  </div>
</template>

<script setup>
  import { ref, computed } from "vue";
  import { watch } from "vue";
  import { keyboardText } from "@/store/keyboardText";

  const { addKey, selectedKey, resetMultiTap } = keyboardText();

  const selectedLayout = ref("QWERTY");

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

  watch(selectedLayout, () => {
    resetMultiTap();
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