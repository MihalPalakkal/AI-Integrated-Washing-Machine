export type MachineStatus =
  | 'idle'
  | 'analyzing'
  | 'filling'
  | 'washing'
  | 'rinsing'
  | 'spinning'
  | 'completed'
  | 'error';

export interface MachineState {
  status: MachineStatus;
  stage: string;
  timeRemaining: number; // seconds
  waterUsage: number;    // litres
  detergentUsage: number; // ml
  temperature: number;   // °C
  loadWeight: number;    // kg
  currentCycle: string;
  elapsedSeconds?: number;
}

// Raw root backend GET /api/state response shape
export interface RootBackendState {
  status: string;        // "IDLE", "WASHING", "RINSING", "SPINNING", "ERROR"
  current_load_size: string;
  detected_fabric: string;
  detected_color: string;
  time_remaining: number;  // minutes
  wash_config: {
    mode: string;
    water_level: number;
    spin_time: number;
    temperature: number;
    extra_rinse: boolean;
  };
  door_locked: boolean;
  water_supply_ok: boolean;
  current_phase: string | null;
}

// Raw root backend notification shape
export interface RootBackendNotification {
  id: string;
  type: string;
  message: string;
  timestamp: string;
}
