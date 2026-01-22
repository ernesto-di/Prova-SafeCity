import carla
import config
import utils

class SmartIntersection:
    def __init__(self, id, lights, world):
        self.id = id
        self.world = world
        self.lights = lights
        
        # Calcolo centro incrocio (media posizioni semafori)
        if len(lights) > 0:
            avg_x = sum([l.get_location().x for l in lights]) / len(lights)
            avg_y = sum([l.get_location().y for l in lights]) / len(lights)
            self.center = carla.Location(x=avg_x, y=avg_y, z=0.0)
        else:
            self.center = carla.Location(0,0,0)

        # Raggruppa semafori per direzione
        self.light_groups = {'N': [], 'S': [], 'E': [], 'O': []}
        for tl in lights:
            yaw = utils.get_yaw(tl)
            d = utils.get_direction_label(yaw)
            self.light_groups[d].append(tl)

        self.phases = ['N', 'S', 'E', 'O']
        self.current_phase_idx = 0 
        self.next_phase_idx = 0
        self.state = 'GREEN' # GREEN, YELLOW, RED_TRANSITION
        self.last_change_time = 0.0
        
        # Inizializza semafori
        self._apply_lights_hard(self.current_phase_idx)
        
        # Memoria tempi di attesa per veicoli
        self.vehicle_stop_times = {} 

    def update(self, action_idx, now):
        """Gestisce la macchina a stati del semaforo (Giallo -> Rosso -> Verde)"""
        if self.state == 'GREEN':
            # Se l'azione è diversa dalla fase attuale E il tempo minimo è passato
            if action_idx != self.current_phase_idx and self.is_min_green_respected(now):
                self.next_phase_idx = action_idx
                self.state = 'YELLOW'
                self.last_change_time = now
                self._set_lights(self.current_phase_idx, carla.TrafficLightState.Yellow)
        
        elif self.state == 'YELLOW':
            if now - self.last_change_time > config.YELLOW_TIME:
                self.state = 'RED_TRANSITION'
                self.last_change_time = now
                self._set_lights(self.current_phase_idx, carla.TrafficLightState.Red)
        
        elif self.state == 'RED_TRANSITION':
            if now - self.last_change_time > config.ALL_RED_TIME:
                self.current_phase_idx = self.next_phase_idx
                self.state = 'GREEN'
                self.last_change_time = now
                self._apply_lights_hard(self.current_phase_idx)

    def _apply_lights_hard(self, green_idx):
        """Forza i semafori allo stato desiderato (senza transizione)"""
        green_dir = self.phases[green_idx]
        for d, group in self.light_groups.items():
            state = carla.TrafficLightState.Green if d == green_dir else carla.TrafficLightState.Red
            for tl in group:
                tl.set_state(state)
    
    def _set_lights(self, phase_idx, state):
        """Imposta lo stato di un gruppo specifico"""
        direction = self.phases[phase_idx]
        for tl in self.light_groups[direction]:
            tl.set_state(state)

    def is_min_green_respected(self, now):
        return (now - self.last_change_time) > config.MIN_GREEN_TIME

    def get_sensors_data(self, vehicles, now):
        """Scansiona veicoli vicini per calcolare code e attese"""
        data = {
            'queues': {'N':0, 'S':0, 'E':0, 'O':0},
            'wait_times': {'N':0.0, 'S':0.0, 'E':0.0, 'O':0.0},
            'emergencies': {'N':False, 'S':False, 'E':False, 'O':False},
            'current_phase': self.current_phase_idx
        }
        
        # Rilevamento raggio d'azione dell'incrocio (es. 60 metri)
        DETECTION_DIST = 60.0

        for v_data in vehicles:
            actor = v_data['actor']
            if not actor.is_alive: continue
            
            # Filtro distanza
            loc = actor.get_location()
            if loc.distance(self.center) > DETECTION_DIST: 
                continue 
            
            # Identifica direzione veicolo
            v_yaw = utils.get_yaw(actor)
            v_dir = utils.get_direction_label(v_yaw)
            
            # Check Emergenza
            type_id = actor.type_id
            is_emergency = ('ambulance' in type_id or 'police' in type_id or 'firetruck' in type_id)

            # Check se fermo (Stop Time)
            vel = utils.vehicle_speed(actor)
            wait = 0.0
            if vel < 0.5: # Considerato fermo
                if actor.id not in self.vehicle_stop_times:
                    self.vehicle_stop_times[actor.id] = now
                wait = now - self.vehicle_stop_times[actor.id]
            else:
                # Se si muove, resetta il timer
                self.vehicle_stop_times.pop(actor.id, None)
            
            # Aggiorna statistiche
            data['queues'][v_dir] += 1
            if wait > data['wait_times'][v_dir]:
                 data['wait_times'][v_dir] = wait
            if is_emergency:
                data['emergencies'][v_dir] = True

        return data
        