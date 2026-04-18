import { ref } from "vue";

const direction = ref("...");
const SESSION_ID_STORAGE_KEY = "eyesist_session_id";
const sessionId = ref(null);

const loadSessionId = () => {
  if (typeof window === "undefined") {
    return sessionId.value;
  }

  sessionId.value = window.localStorage.getItem(SESSION_ID_STORAGE_KEY);
  return sessionId.value;
};

const ensureSessionId = () => {
  if (sessionId.value) {
    return sessionId.value;
  }

  const stored = loadSessionId();
  if (stored) {
    return stored;
  }

  const generated = crypto.randomUUID();
  window.localStorage.setItem(SESSION_ID_STORAGE_KEY, generated);
  sessionId.value = generated;
  return generated;
};

const setSessionId = (value) => {
  sessionId.value = value;

  if (typeof window === "undefined") {
    return;
  }

  if (value) {
    window.localStorage.setItem(SESSION_ID_STORAGE_KEY, value);
  } else {
    window.localStorage.removeItem(SESSION_ID_STORAGE_KEY);
  }
};

export function useEyeStore() {
  return { direction, sessionId, loadSessionId, ensureSessionId, setSessionId };
}
