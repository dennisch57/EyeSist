import { ref } from "vue";

const direction = ref("...");

export function useEyeStore() {
  return { direction };
}