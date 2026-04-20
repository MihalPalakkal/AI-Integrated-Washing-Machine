from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime

class MachineStatus(str, Enum):
    IDLE = "IDLE"
    WASHING = "WASHING"
    RINSING = "RINSING"
    SPINNING = "SPINNING"
    ERROR = "ERROR"

class LoadSize(str, Enum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"

class WashMode(str, Enum):
    AI_AUTO = "AI_AUTO"
    QUICK_WASH = "QUICK_WASH"
    DELICATE = "DELICATE"
    HEAVY = "HEAVY"
    CUSTOM = "CUSTOM"

class WashConfig(BaseModel):
    mode: WashMode
    water_level: int = 50  # 0-100
    spin_time: int = 10  # minutes
    temperature: int = 40  # Celsius
    extra_rinse: bool = False
    duration: Optional[int] = None
    detergent_usage: int = 30
    load_weight: float = 3.5
    cycle_name: Optional[str] = None

# Exact match for frontend's MachineState
class MachineState(BaseModel):
    status: str = "idle"
    stage: str = "Ready"
    timeRemaining: int = 0
    waterUsage: int = 0
    detergentUsage: int = 0
    temperature: int = 0
    loadWeight: float = 0.0
    currentCycle: str = "None"
    elapsedSeconds: int = 0

# Internal tracking state for the simulator
class InternalState(BaseModel):
    status: MachineStatus = MachineStatus.IDLE
    current_load_size: LoadSize = LoadSize.MEDIUM
    time_remaining: int = 0  # In seconds
    wash_config: WashConfig
    door_locked: bool = False
    water_supply_ok: bool = True
    current_phase: Optional[str] = None
    elapsed_seconds: int = 0
    load_weight: float = 3.5

class AIInsight(BaseModel):
    fabric_confidence: float
    color_confidence: float
    dirt_level: str
    recommendation: str
    explanation: str

class Notification(BaseModel):
    id: str
    type: str  # 'info', 'warning', 'error', 'success'
    title: str
    message: str
    timestamp: str
