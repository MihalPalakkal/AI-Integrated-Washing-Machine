import { Alert } from 'react-native';
import { MachineState, MachineStatus } from '../types/Machine';
import { FabricAnalysisResult, FabricDetection } from '../types/Fabric';
import { WashCycle, WashHistoryEntry, AppNotification } from '../types/Cycle';
import {
  saveMachineState,
  saveFabricAnalysis,
  saveWashHistory,
  getWashHistoryFromFirebase,
  getFabricAnalysisHistory,
  saveAnalysisParams,
  resetAnalysisStatus,
} from './firebase';

// ─── Backend Configuration (Dynamic cloud discovery) ─────────────────────────
import { getDatabase, ref, get } from 'firebase/database';

let LAPTOP_IP = '10.142.233.147'; // Fallback to last known local IP

const getRootUrl = () => `http://${LAPTOP_IP}:8001`;
const getApi1Url = () => `http://${LAPTOP_IP}:8000`;
const getApi2Url = () => `http://${LAPTOP_IP}:8002`;

// --- Latency Fix: Fetch with Timeout --- //
async function fetchWithTimeout(resource: string, options: any = {}, timeout = 10000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(resource, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

// Async discovery called once on startup (Firebase DB must be initialized)
export const discoverBackendIp = async () => {
  try {
    // 1. Try to see if localhost/127.0.0.1 is already working (USB cable / emulator)
    const localRes = await fetchWithTimeout(`http://127.0.0.1:8001/`, { method: 'GET' }, 2000).catch(() => null);
    if (localRes && localRes.ok) {
      LAPTOP_IP = '127.0.0.1';
      console.log('🔌 USB Cable/127.0.0.1 detected. Using local bridge.');
      return;
    }

    // 2. Otherwise try Firebase discovery
    const db = getDatabase();
    const snapshot = await get(ref(db, 'backend_ip'));
    if (snapshot.exists() && snapshot.val()) {
      LAPTOP_IP = snapshot.val();
      console.log('✅ Dynamic Cloud Discovery: Backend is at', LAPTOP_IP);
    }
  } catch (e) {
    console.warn('Backend auto-discovery failed, using fallback IP', e);
  }
};
// ─── Mock Data (demo mode when backends are unreachable) ──────────────────────

const MOCK_MACHINE_STATUS: MachineState = {
  status: 'idle',
  stage: 'Ready',
  timeRemaining: 0,
  waterUsage: 0,
  detergentUsage: 0,
  temperature: 0,
  loadWeight: 0,
  currentCycle: 'None',
  elapsedSeconds: 0
};

const MOCK_FABRIC_ANALYSIS: FabricAnalysisResult = {
  fabrics: [
    { name: 'Cotton', confidence: 0.7, fiberCategory: 'Natural', dirtLevel: 2, description: 'Soft, breathable natural fiber' },
    { name: 'Polyester', confidence: 0.2, fiberCategory: 'Synthetic', dirtLevel: 1, description: 'Durable synthetic material' },
  ],
  recommendedCycle: 'Gentle Cold Wash',
};

const MOCK_RECOMMENDED_CYCLE: WashCycle = {
  temperature: 30,
  spinTime: 10,
  duration: 45,
  detergent: 35,
  water: 20,
  soakTime: 10,
  washCycles: 1,
  agitationPattern: 'Normal',
  spinTimeOptions: [5, 10, 15, 20, 25],
};

const MOCK_HISTORY: WashHistoryEntry[] = [];

const MOCK_NOTIFICATIONS: AppNotification[] = [
  {
    id: '1',
    type: 'success',
    title: 'Wash Completed',
    message: 'Your Eco Wash cycle has finished.',
    timestamp: new Date().toISOString(),
  },
  {
    id: '2',
    type: 'warning',
    title: 'Detergent Low',
    message: 'Detergent level is below 20%. Please refill soon.',
    timestamp: new Date(Date.now() - 3600000).toISOString(),
  },
  {
    id: '3',
    type: 'info',
    title: 'Camera Cleaned',
    message: 'Vision sensor calibration successful.',
    timestamp: new Date(Date.now() - 86400000).toISOString(),
  },
];

// Mutable demo state
let demoMachineState = { ...MOCK_MACHINE_STATUS };
let currentWashStartTime: number | null = null;

// ─── Last AI Predicted Cycle (stored in memory) ─────────────────────────────
let lastPredictedCycle: WashCycle | null = null;

// ─── API Service ──────────────────────────────────────────────────────────────

export const api = {
  // ---- Last AI Cycle Getter ---- //
  getLastPredictedCycle: (): WashCycle | null => lastPredictedCycle,

  // ---- Root Backend (Machine Simulator) ---- //

  // GET /api/state → MachineState
  getMachineStatus: async (): Promise<MachineState> => {
    try {
      const res = await fetchWithTimeout(`${getRootUrl()}/api/state`);
      return await res.json();
    } catch {
      return { ...demoMachineState };
    }
  },

  // POST /api/start
  startWash: async (params?: any): Promise<MachineState> => {
    currentWashStartTime = Date.now();
    let result: MachineState;
    try {
      const res = await fetch(`${getRootUrl()}/api/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: params ? JSON.stringify(params) : undefined,
      });
      result = await res.json();
    } catch {
      demoMachineState.status = 'washing';
      demoMachineState.stage = 'Washing';
      demoMachineState.timeRemaining = params?.duration ? params.duration * 60 : 2700;
      demoMachineState.waterUsage = params?.water || 24;
      demoMachineState.detergentUsage = params?.detergent || 25;
      demoMachineState.temperature = params?.temperature || 30;
      demoMachineState.loadWeight = params?.loadWeight || 3.2;
      demoMachineState.currentCycle = params?.agitationPattern || 'Eco Wash';
      result = { ...demoMachineState };
    }
    // Save to Firebase
    saveMachineState(result).catch(() => {});
    return result;
  },

  // POST /api/pause
  pauseWash: async (): Promise<MachineState> => {
    try {
      const res = await fetchWithTimeout(`${getRootUrl()}/api/pause`, { method: 'POST' });
      return await res.json();
    } catch {
      demoMachineState = { ...MOCK_MACHINE_STATUS };
      return { ...demoMachineState };
    }
  },

  // POST /api/stop
  stopWash: async (): Promise<MachineState> => {
    let result: MachineState;
    try {
      const res = await fetchWithTimeout(`${getRootUrl()}/api/stop`, { method: 'POST' });
      result = await res.json();
    } catch {
      demoMachineState = { ...MOCK_MACHINE_STATUS };
      result = { ...demoMachineState };
    }
    // Save wash history to Firebase
    if (currentWashStartTime) {
      const duration = Math.round((Date.now() - currentWashStartTime) / 60000);
      saveWashHistory({
        date: new Date().toISOString(),
        fabricDetected: lastPredictedCycle?.agitationPattern || 'Unknown',
        cycleUsed: result.currentCycle || 'Custom',
        waterConsumed: result.waterUsage,
        detergentConsumed: result.detergentUsage,
        duration,
      }).catch(() => {});
    }
    currentWashStartTime = null;
    saveMachineState(result).catch(() => {});
    // Reset analysis status in Firebase
    resetAnalysisStatus().catch(() => {});
    return result;
  },

  // POST /api/config — apply wash settings
  updateConfig: async (config: any): Promise<MachineState> => {
    try {
      const res = await fetchWithTimeout(`${getRootUrl()}/api/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      return await res.json();
    } catch {
      return { ...demoMachineState };
    }
  },

  // GET /api/notifications
  getNotifications: async (): Promise<AppNotification[]> => {
    try {
      const res = await fetchWithTimeout(`${getRootUrl()}/api/notifications`);
      return await res.json();
    } catch (err: any) {
      // Suppression: Don't show loud warnings for non-critical background timeouts
      if (err.name === 'AbortError') {
         console.log('🕒 Background notifications fetch timed out (expected on startup)');
         return MOCK_NOTIFICATIONS;
      }
      console.warn('Warning: Could not fetch notifications.', err);
      return MOCK_NOTIFICATIONS;
    }
  },

  // GET /api/ai/insight
  getAIInsight: async () => {
    try {
      const res = await fetchWithTimeout(`${getRootUrl()}/api/ai/insight`);
      return await res.json();
    } catch {
      return {
        fabric_confidence: 0.92,
        color_confidence: 0.88,
        dirt_level: 'Medium',
        recommendation: 'Eco Wash recommended',
        explanation: 'Detected mixed fabrics.',
      };
    }
  },

  // ---- API-1 (Fabric Identifier) ---- //

  // POST /predict — upload image(s), get fabric detection results
  analyzeFabric: async (imageUris: string[]): Promise<FabricAnalysisResult> => {
    try {
      const formData = new FormData();
      imageUris.forEach((uri, index) => {
        formData.append('files', {
          uri: uri,
          type: 'image/jpeg',
          name: `clothing_${index}.jpg`,
        } as any);
      });

      const res = await fetch(`${getApi1Url()}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`API-1 returned ${res.status}`);
      }

      return await res.json();
    } catch (err) {
      console.error('API-1 analyzeFabric error:', err);
      return MOCK_FABRIC_ANALYSIS;
    }
  },

  // Get raw API-1 result for passing to API-2
  analyzeFabricRaw: async (imageUris: string[]): Promise<FabricAnalysisResult | null> => {
    try {
      const formData = new FormData();
      imageUris.forEach((uri, index) => {
        formData.append('files', {
          uri: uri,
          type: 'image/jpeg',
          name: `clothing_${index}.jpg`,
        } as any);
      });

      const res = await fetch(`${getApi1Url()}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`API-1 returned ${res.status}`);
      }

      return await res.json();
    } catch (err) {
      console.error('API-1 analyzeFabricRaw error:', err);
      return null;
    }
  },

  // ---- API-2 (Washing Parameter Predictor) ---- //

  // POST /predict — send one API-1 result item, get washing params
  getWashingParams: async (fabric: FabricDetection): Promise<WashCycle> => {
    try {
      const res = await fetch(`${getApi2Url()}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(fabric),
      });

      if (!res.ok) {
        throw new Error(`API-2 returned ${res.status}`);
      }

      const cycle = await res.json();
      
      // Add a dynamic load weight prediction in frontend (AI-2 doesn't predict it yet)
      // Dirt level 1-5 -> roughly 1.5kg to 5.0kg
      const dirtWeightMap: Record<number, number> = {
        1: 1.5, 2: 2.2, 3: 3.5, 4: 4.8, 5: 6.0
      };
      cycle.loadWeight = dirtWeightMap[fabric.dirtLevel] || 3.5;
      
      if (!cycle.spinTimeOptions) {
        cycle.spinTimeOptions = [5, 10, 15, 20, 25];
      }
      
      return cycle;
    } catch (err) {
      console.error('API-2 getWashingParams error:', err);
      return MOCK_RECOMMENDED_CYCLE;
    }
  },

  // ---- Full Pipeline: API-1 → API-2 ---- //

  // Upload an image, identify fabric, then predict washing params
  fullPipeline: async (imageUris: string[]): Promise<{
    fabricResult: FabricAnalysisResult;
    washCycle: WashCycle;
    rawAPI1: FabricAnalysisResult | null;
  }> => {
    // Step 1: Send images to API-1
    const rawAPI1 = await api.analyzeFabricRaw(imageUris);

    if (!rawAPI1 || rawAPI1.fabrics.length === 0) {
      return {
        fabricResult: MOCK_FABRIC_ANALYSIS,
        washCycle: MOCK_RECOMMENDED_CYCLE,
        rawAPI1: null,
      };
    }

    const fabricResult = rawAPI1;

    // Step 2: Take the first valid result and send to API-2
    const validResult = rawAPI1.fabrics.find((r: FabricDetection) => r.name !== 'Unknown');
    if (!validResult) {
      return {
        fabricResult,
        washCycle: MOCK_RECOMMENDED_CYCLE,
        rawAPI1,
      };
    }

    const washCycle = await api.getWashingParams(validResult);

    // Store the last AI predicted cycle in memory
    lastPredictedCycle = washCycle;

    // Push the predicted config to the root backend so the Dashboard updates
    try {
      await api.updateConfig({
        mode: 'CUSTOM',
        water_level: washCycle.water,
                spin_time: washCycle.spinTime,
        temperature: washCycle.temperature,
        duration: washCycle.duration,
        detergent_usage: washCycle.detergent,
        load_weight: washCycle.loadWeight,
        cycle_name: washCycle.agitationPattern,
      });
    } catch (e) {
      console.warn('Failed to push config to root backend:', e);
    }

    // 🔥 Dynamically update root-level Firebase keys with analysis results
    saveAnalysisParams({
      detergent: washCycle.detergent,
      soakTime: washCycle.soakTime,
      spinTime: washCycle.spinTime,
      washCycles: washCycle.washCycles,
      water: washCycle.water,
      temperature: washCycle.temperature,
      loadWeight: washCycle.loadWeight,
    }).catch(() => {});

    // Save fabric analysis to Firebase
    saveFabricAnalysis({
      timestamp: new Date().toISOString(),
      fabrics: fabricResult.fabrics,
      recommendedCycle: fabricResult.recommendedCycle,
      washCycle,
    }).catch(() => {});

    return {
      fabricResult,
      washCycle,
      rawAPI1,
    };
  },

  // ---- Legacy/Convenience ---- //

  getFabricAnalysis: async (): Promise<FabricAnalysisResult> => {
    return MOCK_FABRIC_ANALYSIS;
  },

  getRecommendedCycle: async (): Promise<WashCycle> => {
    return MOCK_RECOMMENDED_CYCLE;
  },

  overrideCycle: async (cycle: Partial<WashCycle>): Promise<MachineState> => {
    try {
      const config = {
        mode: 'CUSTOM',
        water_level: cycle.water || 20,
        spin_time: cycle.spinTime || 10,
        temperature: cycle.temperature || 30,
        extra_rinse: cycle.extraRinse || false,
        duration: cycle.duration || 45,
        detergent_usage: cycle.detergent || 35,
        load_weight: cycle.loadWeight || 3.5,
        cycle_name: cycle.agitationPattern || 'Custom Override',
      };
      const res = await fetch(`${getRootUrl()}/api/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      return await res.json();
    } catch {
      demoMachineState = {
        ...demoMachineState,
        temperature: cycle.temperature ?? demoMachineState.temperature,
        currentCycle: 'Custom Override',
      };
      return { ...demoMachineState };
    }
  },

  getWashHistory: async (): Promise<WashHistoryEntry[]> => {
    const fbHistory = await getWashHistoryFromFirebase();
    if (fbHistory.length > 0) return fbHistory;
    return MOCK_HISTORY;
  },
};
