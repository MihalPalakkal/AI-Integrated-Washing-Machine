import { initializeApp } from 'firebase/app';
import { getDatabase, ref, set, push, get, onValue, remove, query, orderByChild, limitToLast, update } from 'firebase/database';

// Firebase Web App Config
const firebaseConfig = {
  apiKey: "AIzaSyCTOJauLTZGF825ZeuiyjQqdXYn6wZpZIE",
  authDomain: "ai-washing-machine.firebaseapp.com",
  databaseURL: "https://ai-washing-machine-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "ai-washing-machine",
  storageBucket: "ai-washing-machine.firebasestorage.app",
  messagingSenderId: "204106397438",
  appId: "1:204106397438:web:2548a49744679365a797af",
  measurementId: "G-1EKJ8K9CC7"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getDatabase(app);

// ─── Types ───────────────────────────────────────────────────────────────────

export interface FabricAnalysisRecord {
  id?: string;
  timestamp: string;
  fabrics: any[];
  recommendedCycle: string;
  washCycle: any;
}

// ─── Machine State (Real-time) ───────────────────────────────────────────────

export function onMachineStateChange(callback: (data: any) => void): () => void {
  const stateRef = ref(db, 'machineState');
  const unsub = onValue(stateRef, (snapshot) => {
    const data = snapshot.val();
    if (data) callback(data);
  });
  return unsub;
}

export async function saveMachineState(state: any): Promise<void> {
  try {
    await set(ref(db, 'machineState'), state);
  } catch (e) {
    console.warn('Failed to save machine state to Firebase:', e);
  }
}

// ─── Notifications ───────────────────────────────────────────────────────────

export async function getNotificationsFromFirebase(): Promise<any[]> {
  try {
    const snapshot = await get(ref(db, 'notifications'));
    if (!snapshot.exists()) return [];
    const data = snapshot.val();
    return Object.values(data).reverse();
  } catch {
    return [];
  }
}

// ─── Wash History ────────────────────────────────────────────────────────────

export async function saveWashHistory(entry: any): Promise<void> {
  try {
    await push(ref(db, 'washHistory'), entry);
  } catch (e) {
    console.warn('Failed to save wash history:', e);
  }
}

export async function getWashHistoryFromFirebase(): Promise<any[]> {
  try {
    const snapshot = await get(query(ref(db, 'washHistory'), orderByChild('date'), limitToLast(20)));
    if (!snapshot.exists()) return [];
    const data = snapshot.val();
    return Object.entries(data)
      .map(([key, val]: [string, any]) => ({ id: key, ...val }))
      .reverse();
  } catch {
    return [];
  }
}

export function onWashHistoryChange(callback: (data: any[]) => void): () => void {
  const histRef = ref(db, 'washHistory');
  const unsub = onValue(histRef, (snapshot) => {
    if (!snapshot.exists()) { callback([]); return; }
    const data = snapshot.val();
    const list = Object.entries(data)
      .map(([key, val]: [string, any]) => ({ id: key, ...val }))
      .reverse();
    callback(list);
  });
  return unsub;
}

// ─── Fabric Analysis History ─────────────────────────────────────────────────

export async function saveFabricAnalysis(record: FabricAnalysisRecord): Promise<void> {
  try {
    await push(ref(db, 'fabricAnalysis'), record);
  } catch (e) {
    console.warn('Failed to save fabric analysis:', e);
  }
}

export async function getFabricAnalysisHistory(): Promise<FabricAnalysisRecord[]> {
  try {
    const snapshot = await get(query(ref(db, 'fabricAnalysis'), orderByChild('timestamp'), limitToLast(20)));
    if (!snapshot.exists()) return [];
    const data = snapshot.val();
    return Object.entries(data)
      .map(([key, val]: [string, any]) => ({ id: key, ...val }))
      .reverse();
  } catch {
    return [];
  }
}

export function onFabricAnalysisChange(callback: (data: FabricAnalysisRecord[]) => void): () => void {
  const analysisRef = ref(db, 'fabricAnalysis');
  const unsub = onValue(analysisRef, (snapshot) => {
    if (!snapshot.exists()) { callback([]); return; }
    const data = snapshot.val();
    const list = Object.entries(data)
      .map(([key, val]: [string, any]) => ({ id: key, ...val }))
      .reverse();
    callback(list);
  });
  return unsub;
}

// ─── User Preferences ────────────────────────────────────────────────────────

export async function saveUserPreferences(prefs: any): Promise<void> {
  try {
    await set(ref(db, 'userPreferences'), prefs);
  } catch (e) {
    console.warn('Failed to save preferences:', e);
  }
}

export async function getUserPreferences(): Promise<any | null> {
  try {
    const snapshot = await get(ref(db, 'userPreferences'));
    return snapshot.exists() ? snapshot.val() : null;
  } catch {
    return null;
  }
}

// ─── Clear All History ───────────────────────────────────────────────────────

export async function clearAllHistory(): Promise<void> {
  try {
    await remove(ref(db, 'washHistory'));
    await remove(ref(db, 'fabricAnalysis'));
    await remove(ref(db, 'notifications'));
  } catch (e) {
    console.warn('Failed to clear history:', e);
  }
}

// ─── Dynamic Analysis Parameters (Root-Level) ───────────────────────────────
// After each API-1 → API-2 analysis, write the predicted wash parameters
// to root-level Firebase keys so they update dynamically.

export async function saveAnalysisParams(washCycle: {
  detergent: number;
  soakTime: number;
  spinTime: number;
  washCycles: number;
  water: number;
  temperature?: number;
  loadWeight?: number;
}): Promise<void> {
  try {
    const updates: Record<string, any> = {
      detergent_amount: washCycle.detergent,
      soak_time: washCycle.soakTime,
      spin_time: washCycle.spinTime,
      wash_cycle: washCycle.washCycles,
      water_level: washCycle.water,
      status: false,
    };
    if (washCycle.temperature !== undefined) {
      updates.temperature = washCycle.temperature;
    }
    if (washCycle.loadWeight !== undefined) {
      updates.load_weight = washCycle.loadWeight;
    }
    await update(ref(db), updates);
    console.log('✅ Firebase root analysis params updated dynamically');
  } catch (e) {
    console.warn('Failed to save analysis params to Firebase:', e);
  }
}

// Reset analysis status after wash completes or is stopped
export async function resetAnalysisStatus(): Promise<void> {
  try {
    await update(ref(db), {
      status: false,
    });
  } catch (e) {
    console.warn('Failed to reset analysis status:', e);
  }
}

// ─── Listen to Root AI Params ───────────────────────────────────────────────
export function onRootParamsChange(callback: (params: any) => void): () => void {
  const rootRef = ref(db);
  const unsub = onValue(rootRef, (snapshot) => {
    if (!snapshot.exists()) return;
    const data = snapshot.val();
    callback({
      detergent_amount: data.detergent_amount,
      soak_time: data.soak_time,
      spin_time: data.spin_time,
      water_level: data.water_level,
      wash_cycle: data.wash_cycle,
      temperature: data.temperature,
      load_weight: data.load_weight,
    });
  });
  return unsub;
}
// ─── Arduino Logs (Real-time) ────────────────────────────────────────────────
export function onLogsChange(callback: (data: any) => void): () => void {
  const logsRef = ref(db, 'logs');
  const unsub = onValue(logsRef, (snapshot) => {
    const data = snapshot.val();
    if (data) callback(data);
  });
  return unsub;
}

// ─── Root Status (Real-time) ─────────────────────────────────────────────────
export function onRootStatusChange(callback: (status: boolean) => void): () => void {
  const statusRef = ref(db, 'status');
  const unsub = onValue(statusRef, (snapshot) => {
    const val = snapshot.val();
    callback(!!val); // Convert to explicit boolean
  });
  return unsub;
}
