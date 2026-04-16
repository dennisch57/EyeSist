import { ref } from "vue";

const direction = ref("CENTER");

export function useEyeStore() {
  return { direction };
}