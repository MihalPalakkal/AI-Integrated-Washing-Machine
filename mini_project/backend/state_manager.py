import asyncio
import requests
from datetime import datetime, timedelta
from uuid import uuid4
from models import MachineState, InternalState, WashConfig, MachineStatus, AIInsight, Notification, WashMode, LoadSize

class StateManager:
    def __init__(self):
        self.state = InternalState(
            wash_config=WashConfig(mode=WashMode.AI_AUTO)
        )
        self.notifications: list[Notification] = []
        self._washing_task = None
        self._add_dummy_data()
        self.firebase_url = "https://ai-washing-machine-default-rtdb.asia-southeast1.firebasedatabase.app/machineState.json"

    def _broadcast_state(self):
        """Broadcast current state to Firebase RTDB in background."""
        asyncio.create_task(self._do_broadcast())

    async def _do_broadcast(self):
        try:
            state = self.get_state()
            state_dict = {
                "status": state.status,
                "stage": state.stage,
                "timeRemaining": state.timeRemaining,
                "waterUsage": state.waterUsage,
                "detergentUsage": state.detergentUsage,
                "temperature": state.temperature,
                "loadWeight": state.loadWeight,
                "currentCycle": state.currentCycle,
                "elapsedSeconds": state.elapsedSeconds
            }
            await asyncio.to_thread(requests.put, self.firebase_url, json=state_dict, timeout=5)
        except Exception as e:
            print(f"Firebase Broadcast Error: {e}")

    def _sync_root_status(self, is_running: bool):
        """Sync root status in background."""
        asyncio.create_task(self._do_sync_root_status(is_running))

    async def _do_sync_root_status(self, is_running: bool):
        """Sync the root 'status' flag to Firebase for hardware control."""
        try:
            root_url = self.firebase_url.replace("machineState.json", ".json")
            resp = await asyncio.to_thread(requests.patch, root_url, json={"status": is_running}, timeout=5)
            if resp.status_code != 200:
                print(f"Warning: Firebase Root Sync Status {resp.status_code} - {resp.text}")
            else:
                print(f"Success: Firebase Root Sync: status={is_running}")
        except Exception as e:
            print(f"Error: Firebase Root Sync Error: {e}")
    def _add_dummy_data(self):
        # Initial Dummy Data
        now = datetime.now()
        
        # Recent Notifications
        self.add_notification("info", "System Initialized", "Startup Context")
        self.add_notification("warning", "Check drum balance (Previous Cycle)", "Drum Balance")
        
        # Historical Logs (mocked as notifications for now)
        self.notifications.append(Notification(
            id=str(uuid4()), 
            type="success", 
            title="Wash Completed",
            message="Wash Cycle Completed - Heavy Duty", 
            timestamp=(now - timedelta(days=1, hours=3)).isoformat()
        ))
        self.notifications.append(Notification(
            id=str(uuid4()), 
            type="info", 
            title="Smart Diagnosis",
            message="No issues found", 
            timestamp=(now - timedelta(days=2)).isoformat()
        ))
        
        # Sort by timestamp (newest last, but UI reverses it)
        self.notifications.sort(key=lambda x: x.timestamp)

    def get_state(self) -> MachineState:
        status_map = {
            MachineStatus.IDLE: 'idle',
            MachineStatus.WASHING: 'washing',
            MachineStatus.RINSING: 'rinsing',
            MachineStatus.SPINNING: 'spinning',
            MachineStatus.ERROR: 'error'
        }
        
        status_str = status_map.get(self.state.status, 'idle')
        stage = self.state.current_phase
        if not stage:
            stage = "Ready" if status_str == 'idle' else status_str.capitalize()
            
        load_weight_map = {
            LoadSize.SMALL: 2.0,
            LoadSize.MEDIUM: 3.5,
            LoadSize.LARGE: 5.0
        }
        
        current_cycle_str = self.state.wash_config.cycle_name if (self.state.wash_config.mode == WashMode.CUSTOM and self.state.wash_config.cycle_name) else self.state.wash_config.mode.replace('_', ' ').title()
        
        return MachineState(
            status=status_str,
            stage=stage,
            timeRemaining=self.state.time_remaining,
            waterUsage=self.state.wash_config.water_level,
            detergentUsage=self.state.wash_config.detergent_usage,
            temperature=self.state.wash_config.temperature,
            loadWeight=self.state.wash_config.load_weight,
            currentCycle=current_cycle_str,
            elapsedSeconds=self.state.elapsed_seconds
        )

    def update_config(self, config: WashConfig):
        if self.state.status != MachineStatus.IDLE:
            raise ValueError("Cannot update config while machine is running")
        self.state.wash_config = config
        self.state.time_remaining = self._calculate_time(config)
        self._broadcast_state()

    def start_wash(self, custom_params: dict = None):
        if self.state.status != MachineStatus.IDLE:
            return
        
        if not self.state.water_supply_ok:
            self.add_notification("error", "Water supply issue detected!")
            return

        if custom_params:
            self.state.wash_config.mode = WashMode.CUSTOM
            self.state.wash_config.cycle_name = custom_params.get("agitationPattern", "Custom Wash")
            self.state.wash_config.water_level = custom_params.get("water", self.state.wash_config.water_level)
            self.state.wash_config.temperature = custom_params.get("temperature", self.state.wash_config.temperature)
            self.state.wash_config.detergent_usage = custom_params.get("detergent", 30)
            self.state.wash_config.load_weight = custom_params.get("loadWeight", 3.5)
            if "duration" in custom_params:
                self.state.wash_config.duration = custom_params["duration"]

        if not self.state.door_locked:
             # Auto lock
             self.state.door_locked = True
        
        self.state.status = MachineStatus.WASHING
        self.state.time_remaining = self._calculate_time(self.state.wash_config)
        self.state.elapsed_seconds = 0
        
        # Start background wash simulation
        self._washing_task = asyncio.create_task(self._simulate_wash_cycle())
        self.add_notification("info", f"Wash started: {self.state.wash_config.mode}")
        self._broadcast_state()
        self._sync_root_status(True)

    def pause_wash(self):
        if self.state.status in [MachineStatus.WASHING, MachineStatus.RINSING, MachineStatus.SPINNING]:
            if self._washing_task:
                self._washing_task.cancel()
            self.state.status = MachineStatus.IDLE 
            self.state.door_locked = False
            self.add_notification("info", "Wash paused")
            self._broadcast_state()
            self._sync_root_status(False)

    def stop_wash(self):
         if self._washing_task:
            self._washing_task.cancel()
         self.state.status = MachineStatus.IDLE
         self.state.time_remaining = 0
         self.state.current_phase = None
         self.state.door_locked = False
         self.state.elapsed_seconds = 0
         # We keep wash_config so the dashboard shows the last programmed values
         self.add_notification("info", "Wash stopped")
         self._broadcast_state()
         self._sync_root_status(False)

    def get_ai_insight(self) -> AIInsight:
        return AIInsight(
            fabric_confidence=0.92,
            color_confidence=0.88,
            dirt_level="Medium",
            recommendation="Eco Wash recommended for Mixed load",
            explanation="Detected mixed fabrics (Cotton/Synthetic). Lower temperature (30°C) is safer."
        )

    def add_notification(self, type: str, message: str, title: str = None):
        if not title:
            title = type.capitalize()
        notif = Notification(
            id=str(uuid4()),
            type=type,
            title=title,
            message=message,
            timestamp=datetime.now().isoformat()
        )
        self.notifications.append(notif)
        if len(self.notifications) > 20:
            self.notifications.pop(0)

    def get_notifications(self) -> list[Notification]:
        return self.notifications

    def _calculate_time(self, config: WashConfig) -> int:
        if config.duration is not None:
            return config.duration * 60 # Convert minutes to seconds
            
        base_time_mins = 30
        if config.mode == WashMode.HEAVY: base_time_mins = 90
        elif config.mode == WashMode.QUICK_WASH: base_time_mins = 15
        elif config.mode == WashMode.DELICATE: base_time_mins = 45
        
        if config.extra_rinse: base_time_mins += 15
        return base_time_mins * 60

    async def _simulate_wash_cycle(self):
        try:
            total_time = self.state.time_remaining
            phases = [
                ("Water Fill", 0.1),
                ("Wash", 0.4),
                ("Rinse", 0.3),
                ("Spin", 0.2)
            ]
            
            for phase_name, duration_share in phases:
                self.state.current_phase = phase_name
                phase_seconds = int(total_time * duration_share)
                if phase_seconds < 1: phase_seconds = 1
                
                for _ in range(phase_seconds):
                    await asyncio.sleep(1) 
                    if self.state.time_remaining > 0:
                        self.state.time_remaining -= 1
                    self.state.elapsed_seconds += 1
                    self._broadcast_state()
                    
                    if self.state.status == MachineStatus.WASHING and phase_name == "Rinse":
                         self.state.status = MachineStatus.RINSING
                    elif self.state.status == MachineStatus.RINSING and phase_name == "Spin":
                         self.state.status = MachineStatus.SPINNING

            self.state.status = MachineStatus.IDLE
            self.state.current_phase = None
            self.state.door_locked = False
            self.state.custom_detergent = 0
            self.add_notification("info", "Wash failed? No, Wash completed successfully!")
            self.add_notification("info", "Wash Cycle Completed")
            self._broadcast_state()
            self._sync_root_status(False)

        except asyncio.CancelledError:
            self._sync_root_status(False)
            pass
