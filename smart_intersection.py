import carla
import math
import time
import config
import utils

class SmartIntersection:
    def __init__(self, junction_id, lights, world):
        self.junction_id = junction_id
        self.world = world
        self.light_groups = {'N': [], 'S': [], 'E': [], 'O': []}
        self.phase_to_dir = {0: 'N', 1: 'S', 2: 'E', 3: 'O'}
        self.vehicle_stop_times = {}  # {vehicle_id: timestamp_start_waiting}
        self.is_blocked = False  # Stato occupazione incrocio

        x_coords = [tl.get_location().x for tl in lights]
        y_coords = [tl.get_location().y for tl in lights]
        self.center = carla.Location(x=sum(x_coords) / len(x_coords), y=sum(y_coords) / len(y_coords), z=0.5)

        # Calcolo raggio dinamico del Box Junction (distanza minima semaforo dal centro * fattore sicurezza)
        # Usiamo 0.7 per stare SICURAMENTE lontani dalle linee di arresto
        min_dist = 10000.0
        mapped_count = 0
        for tl in lights:
            d = tl.get_location().distance(self.center)
            if d < min_dist: min_dist = d

            raw_yaw = utils.get_yaw(tl)
            direction = utils.get_direction_label(raw_yaw)

            if direction in self.light_groups:
                self.light_groups[direction].append(tl)
                mapped_count += 1
                tl_loc = tl.get_location()
                self.world.debug.draw_string(
                    tl_loc + carla.Location(z=2.5),
                    f"{direction}",
                    draw_shadow=False,
                    color=carla.Color(255, 255, 0),
                    life_time=3600.0
                )
        
        # Raggio ridotto per prendere solo chi è DAVVERO in mezzo
        self.box_radius = max(5.0, min_dist * 0.6) # Ridotto da 0.7 a 0.6 per sicurezza
        
        self.state = 'GREEN'
        self.current_phase_idx = 0
        self.next_phase_idx = 0
        self.last_change_time = 0.0
        self._apply_lights_hard(self.current_phase_idx)

        # Disegno area rilevamento
        radius = config.EMERGENCY_DETECTION_DIST
        num_segments = 24
        angle_step = 360.0 / num_segments
        color = carla.Color(200, 0, 0)
        for i in range(num_segments):
            angle1 = math.radians(i * angle_step)
            angle2 = math.radians((i + 1) * angle_step)
            p1_loc = carla.Location(x=self.center.x + math.cos(angle1) * radius,
                                    y=self.center.y + math.sin(angle1) * radius, z=1.0)
            p2_loc = carla.Location(x=self.center.x + math.cos(angle2) * radius,
                                    y=self.center.y + math.sin(angle2) * radius, z=1.0)
            world.debug.draw_line(p1_loc, p2_loc, thickness=0.2, color=color, life_time=3600.0)

        # Debug visivo Box Junction (Giallo)
        debug_z = 0.5
        for i in range(num_segments):
            angle1 = math.radians(i * angle_step)
            angle2 = math.radians((i + 1) * angle_step)
            p1 = carla.Location(self.center.x + math.cos(angle1) * self.box_radius, 
                                self.center.y + math.sin(angle1) * self.box_radius, debug_z)
            p2 = carla.Location(self.center.x + math.cos(angle2) * self.box_radius, 
                                self.center.y + math.sin(angle2) * self.box_radius, debug_z)
            world.debug.draw_line(p1, p2, thickness=0.3, color=carla.Color(255, 255, 0), life_time=3600.0)

        utils.log(f"Incrocio {junction_id} inizializzato. Box Radius: {self.box_radius:.1f}m. Mappati {mapped_count}.")

    def _apply_lights_hard(self, phase_idx):
        green_dir = self.phase_to_dir.get(phase_idx, 'N')
        for direction, lights in self.light_groups.items():
            state = carla.TrafficLightState.Green if direction == green_dir else carla.TrafficLightState.Red
            for tl in lights:
                tl.set_state(state)
                tl.freeze(True)

    def _set_physical_lights(self, phase_idx, state):
        target_dir = self.phase_to_dir.get(phase_idx, 'N')
        for direction, lights in self.light_groups.items():
            if direction == target_dir:
                for tl in lights: tl.set_state(state)
            else:
                if state == carla.TrafficLightState.Green:
                    for tl in lights: tl.set_state(carla.TrafficLightState.Red)

    def update(self, ai_requested_phase, now):
        """
        Gestisce la transizione dinamica.
        L'IA decide la fase, ma l'incrocio gestisce la sicurezza e i tempi minimi.
        """
        elapsed = now - self.last_change_time
        
        # --- FASE VERDE (Estensibile dall'IA) ---
        if self.state == 'GREEN':
            # Se l'IA chiede la STESSA fase, estendiamo il verde (Tempo Dinamico)
            if ai_requested_phase == self.current_phase_idx:
                return # Rimaniamo verdi, lasciamo scorrere il traffico
            
            # Se l'IA chiede di cambiare:
            # Rispettiamo un minimo vitale (es. 5s) per non fare sfarfallio
            if elapsed >= config.MIN_GREEN_TIME:
                self.next_phase_idx = ai_requested_phase
                self._set_physical_lights(self.current_phase_idx, carla.TrafficLightState.Yellow)
                self.state = 'YELLOW'
                self.last_change_time = now
        
        # --- FASE GIALLA (Fissa per sicurezza) ---
        elif self.state == 'YELLOW':
            if elapsed >= config.YELLOW_TIME:
                self._set_physical_lights(self.current_phase_idx, carla.TrafficLightState.Red)
                self.state = 'ALL_RED'
                self.last_change_time = now
        
        # --- FASE TUTTO ROSSO (Dinamica Anti-Blocco) ---
        elif self.state == 'ALL_RED':
            # 1. È passato il tempo tecnico di sicurezza?
            if elapsed >= config.ALL_RED_TIME:
                # 2. CONTROLLO CRITICO: L'incrocio è fisicamente libero?
                # Se c'è un veicolo nel mezzo (is_blocked), estendiamo il rosso!
                if self.is_blocked:
                    # Rimaniamo in All Red. Non facciamo nulla.
                    # Questo permette all'incrocio di svuotarsi prima di far entrare altri.
                    pass 
                else:
                    # Via libera alla nuova fase scelta dall'IA
                    self.current_phase_idx = self.next_phase_idx
                    self._set_physical_lights(self.current_phase_idx, carla.TrafficLightState.Green)
                    self.state = 'GREEN'
                    self.last_change_time = now

    def is_min_green_respected(self, now):
        """
        Ritorna True se l'incrocio è VERDE e sono passati almeno MIN_GREEN_TIME secondi.
        """
        if self.state != 'GREEN':
            return False
        return (now - self.last_change_time) >= config.MIN_GREEN_TIME

    def get_sensors_data(self, vehicles_list, now):
        queues = {'N': 0, 'S': 0, 'E': 0, 'O': 0}
        wait_times = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'O': 0.0}
        emerg_wait_times = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'O': 0.0} # [NEW]
        emergencies = {'N': False, 'S': False, 'E': False, 'O': False}
        emerg_keywords = ['police', 'ambulance', 'firetruck']

        current_stopped_ids = set()
        
        # Reset stato blocco
        self.is_blocked = False
        blocked_by_actor = None

        for item in vehicles_list:
            v = item['actor']
            if not v.is_alive: continue
            loc = v.get_location()
            dist = loc.distance(self.center)
            
            # CHECK BOX JUNCTION: Se un veicolo è DENTRO il raggio critico e va piano, blocca l'incrocio.
            # Nota: consideriamo "bloccato" se il veicolo è nell'area. Anche se si muove, è pericoloso cambiare.
            if dist < self.box_radius:
                self.is_blocked = True
                blocked_by_actor = v.id

            if dist > config.EMERGENCY_DETECTION_DIST: continue

            vec = self.center - loc
            dx, dy = vec.x, vec.y

            # Logica geometrica (Indipendente dai semafori)
            direction = ''
            if abs(dx) > abs(dy):
                direction = 'E' if dx < 0 else 'O'
            else:
                direction = 'N' if dy < 0 else 'S'

            is_emerg_vehicle = any(kw in v.type_id for kw in emerg_keywords)
            if is_emerg_vehicle:
                fwd = v.get_transform().get_forward_vector()
                dot = vec.x * fwd.x + vec.y * fwd.y
                if dist > 0.1 and (dot / dist) > 0.5:
                    emergencies[direction] = True

            speed = utils.vehicle_speed(v)
            if speed < 4.0:
                queues[direction] += 1
                current_stopped_ids.add(v.id)
                # Calcolo tempo attesa
                if v.id not in self.vehicle_stop_times:
                    self.vehicle_stop_times[v.id] = now
                
                waited = now - self.vehicle_stop_times[v.id]
                if waited > wait_times[direction]:
                    wait_times[direction] = waited
                
                # [NEW] Se è un veicolo di emergenza, tracciamo il suo tempo specifico
                if is_emerg_vehicle:
                     if waited > emerg_wait_times[direction]:
                        emerg_wait_times[direction] = waited

        # Pulizia veicoli che non sono più fermi in zona
        to_remove = [vid for vid in self.vehicle_stop_times if vid not in current_stopped_ids]
        for vid in to_remove:
            del self.vehicle_stop_times[vid]

        return {
            'queues': queues,
            'wait_times': wait_times,
            'emerg_wait_times': emerg_wait_times,  # [NEW] Tempo attesa specifico emergenze
            'emergencies': emergencies,
            'current_phase': self.current_phase_idx
        }

    def get_emergency_state(self, vehicles_data):
        """
        Controlla rapidamente se ci sono veicoli di emergenza in avvicinamento
        su una delle 4 direzioni (N, S, E, O).
        Restituisce un dizionario: {'N': False, 'S': True, ...}
        """
        emerg_state = {'N': False, 'S': False, 'E': False, 'O': False}
        
        # Se non abbiamo dati sui veicoli, ritorniamo tutto False
        if not vehicles_data:
            return emerg_state

        for data in vehicles_data:
            vehicle = data['actor']
            if not vehicle.is_alive: 
                continue

            # 1. Filtro: È un veicolo di emergenza?
            # Controlla se 'police', 'ambulance' o 'firetruck' sono nell'ID del blueprint
            if not any(k in vehicle.type_id for k in ['police', 'ambulance', 'firetruck']):
                continue

            # 2. Controllo Distanza
            veh_loc = vehicle.get_location()
            dist = veh_loc.distance(self.center)
            
            # Consideriamo solo emergenze entro 50 metri
            if dist > 50.0: 
                continue

            # 3. Controllo Direzione (si sta avvicinando o allontanando?)
            # Calcoliamo il vettore dal veicolo al centro dell'incrocio
            to_center = carla.Vector3D(self.center.x - veh_loc.x, self.center.y - veh_loc.y, 0)
            fwd = vehicle.get_transform().get_forward_vector()
            
            # Prodotto scalare: se > 0, il veicolo sta guardando verso il centro
            dot = fwd.x * to_center.x + fwd.y * to_center.y
            
            if dot < 0: 
                continue # Si sta allontanando dall'incrocio, ignoralo

            # 4. Determina geometricamente su quale braccio si trova (N, S, E, O)
            dx = veh_loc.x - self.center.x
            dy = veh_loc.y - self.center.y

            direction = ''
            # Se la differenza X è maggiore della Y, è sulla strada orizzontale
            if abs(dx) > abs(dy):
                # Se dx > 0 è a Est, se dx < 0 è a Ovest (O)
                direction = 'E' if dx > 0 else 'O'
            else:
                # Strada verticale
                # Se dy > 0 è a Sud, se dy < 0 è a Nord
                direction = 'S' if dy > 0 else 'N'
            
            if direction in emerg_state:
                emerg_state[direction] = True

        return emerg_state