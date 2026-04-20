export interface WashCycle {
  temperature: number;    // °C
    spinTime: number;      // minutes
  duration: number;       // minutes (soak_time + spin_time)
  detergent: number;      // ml
  water: number;          // litres
  soakTime: number;       // minutes
  washCycles: number;     // number of cycles
  agitationPattern?: string;
  spinTimeOptions?: number[];
  loadWeight?: number;    // kg
  extraRinse?: boolean;
}

export interface WashHistoryEntry {
  id: string;
  date: string;
  fabricDetected: string;
  cycleUsed: string;
  waterConsumed: number;
  detergentConsumed: number;
  duration: number;
}

export type NotificationType = 'info' | 'warning' | 'error' | 'success';

export interface AppNotification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: string;
}

// Raw API-2 POST /predict response shape
export interface API2WashingLogic {
  detergent_amount: number;  // ml
  soak_time: number;         // minutes
  spin_time: number;         // minutes
  water_level: number;       // litres
  wash_cycles: number;
  temperature_setting: number; // °C
  mechanical_action: string;   // "Gentle" | "Normal" | "Heavy Duty"
}

export interface API2PredictResponse {
  washing_logic: API2WashingLogic;
}
