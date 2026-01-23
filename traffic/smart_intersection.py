import carla
import config
import env.utils as utils

class SmartIntersection:
    def __init__(self, id, lights, world):
        self.id = id
        self.world = world
        self.lights = lights
        
        # 1. Calcolo Centro
        if len(lights) > 0:
            avg_x = sum([l.get_location().x for l in lights]) / len(lights)
            avg_y = sum([l.get_location().y for l in lights]) / len(lights)
            self.center = carla.Location(x=avg_x, y=avg_y, z=0.0)
        else:
            self.center = carla.Location(0,0,0)

        # 2. CONFIGURAZIONE RETTANGOLI
        STOP_LINE_OFFSET = 12.0
        LANE_SHIFT = 5.0  
        BOX_LENGTH = 55.0
        BOX_WIDTH = 8.0
        BOX_HEIGHT = 4.0
        half_len = BOX_LENGTH / 2.0
        half_width = BOX_WIDTH / 2.0
        
        # --- NUOVO: BOX CENTRALE (INCROCIO) ---
        # Area centrale (es. 14x14m) per rilevare se l'incrocio è fisicamente bloccato
        JUNCTION_EXTENT = 12.0 
        self.junction_box = carla.BoundingBox(self.center, carla.Vector3D(JUNCTION_EXTENT, JUNCTION_EXTENT, BOX_HEIGHT))
        
        # Calcolo Centri dei Box Corsie
        c_nord  = self.center + carla.Location(x=LANE_SHIFT, y=(STOP_LINE_OFFSET + half_len))
        c_sud   = self.center + carla.Location(x=-LANE_SHIFT, y=-(STOP_LINE_OFFSET + half_len))
        c_est   = self.center + carla.Location(x=-(STOP_LINE_OFFSET + half_len), y=LANE_SHIFT)
        c_ovest = self.center + carla.Location(x=(STOP_LINE_OFFSET + half_len), y=-LANE_SHIFT)

        self.detection_boxes = {
            'N': carla.BoundingBox(c_nord, carla.Vector3D(half_width, half_len, BOX_HEIGHT)),
            'S': carla.BoundingBox(c_sud, carla.Vector3D(half_width, half_len, BOX_HEIGHT)),
            'O': carla.BoundingBox(c_est, carla.Vector3D(half_len, half_width, BOX_HEIGHT)), 
            'E': carla.BoundingBox(c_ovest, carla.Vector3D(half_len, half_width, BOX_HEIGHT)) 
        }

        # 3. Setup Semafori
        self.light_groups = {'N': [], 'S': [], 'E': [], 'O': []}
        for tl in lights:
            yaw = utils.get_yaw(tl)
            d = utils.get_direction_label(yaw)
            self.light_groups[d].append(tl)

        self.phases = ['N', 'S', 'E', 'O']
        self.current_phase_idx = 0 
        self.next_phase_idx = 0
        self.state = 'GREEN'
        self.last_change_time = 0.0
        self.vehicle_stop_times = {} 
        
        # DATI PER DASHBOARD
        self.current_queues = {'N':0, 'S':0, 'E':0, 'O':0}
        self.current_waits  = {'N':0.0, 'S':0.0, 'E':0.0, 'O':0.0}
        self.current_emergencies = {'N':False, 'S':False, 'E':False, 'O':False}
        self.junction_occupied = False # Nuovo Stato

        self._apply_lights_hard(self.current_phase_idx)

    def draw_setup_debug(self):
        """Disegna box geometrici, timer, AVVISO EMERGENZA e STATO INCROCIO"""
        z_txt = 8.0
        # AUMENTIAMO LA DURATA: Da 0.15 a 0.5 per evitare sfarfallii
        DEBUG_LIFE_TIME = 0.5 
        
        visual_cfg = {
            'N': {'col': carla.Color(0, 255, 0), 'label': f"NORD {self.id+1}"},
            'S': {'col': carla.Color(255, 0, 0), 'label': f"SUD {self.id+1}"},
            'O': {'col': carla.Color(255, 255, 0), 'label': f"EST {self.id+1}"},
            'E': {'col': carla.Color(0, 0, 255), 'label': f"OVEST {self.id+1}"}
        }
        
        curr_p = self.phases[self.current_phase_idx]
        col_p = carla.Color(0, 255, 0) if self.state == 'GREEN' else carla.Color(255, 255, 0)
        label_phase = visual_cfg[curr_p]['label'].split(' ')[0]
        
        self.world.debug.draw_string(self.center + carla.Location(z=z_txt+2), 
                                     f"INCROCIO {self.id+1} | VERDE: {label_phase}", 
                                     color=col_p, life_time=DEBUG_LIFE_TIME)

        id_rot = carla.Rotation()
        
        # --- DISEGNO BOX CENTRALE ---
        # Rosso se occupato, Blu trasparente se libero
        jun_col = carla.Color(255, 0, 0) if self.junction_occupied else carla.Color(0, 0, 255)
        
        # FIX VISIBILITÀ: Spessore aumentato a 0.5 e life_time aumentato
        self.world.debug.draw_box(self.junction_box, id_rot, thickness=0.5, color=jun_col, life_time=DEBUG_LIFE_TIME)
        
        if self.junction_occupied:
             self.world.debug.draw_string(self.center, "BLOCCATO", color=carla.Color(255,0,0), life_time=DEBUG_LIFE_TIME)

        # Disegno Box Corsie
        for key, box in self.detection_boxes.items():
            cfg = visual_cfg[key]
            # FIX VISIBILITÀ: Spessore aumentato
            self.world.debug.draw_box(box, id_rot, thickness=0.5, color=cfg['col'], life_time=DEBUG_LIFE_TIME)
            
            q = self.current_queues.get(key, 0)
            w = self.current_waits.get(key, 0.0)
            is_emerg = self.current_emergencies.get(key, False)
            
            txt = f"{cfg['label']}: {q}"
            if w > 0: txt += f" | {w:.1f}s"
            
            txt_col = cfg['col'] if q > 0 else carla.Color(200,200,200)
            if w > 60: txt_col = carla.Color(255, 0, 0)
            
            self.world.debug.draw_string(box.location + carla.Location(z=z_txt), txt, 
                                         draw_shadow=True, color=txt_col, life_time=DEBUG_LIFE_TIME)
            
            if is_emerg:
                self.world.debug.draw_string(box.location + carla.Location(z=z_txt + 3.0), 
                                             "!!! EMERGENZA !!!", 
                                             draw_shadow=True, color=carla.Color(255, 0, 0), life_time=DEBUG_LIFE_TIME)

    def update(self, action_idx, now, vehicles=None):
        if self.state == 'RED_TRANSITION':
            if now - self.last_change_time > config.ALL_RED_TIME:
                self.current_phase_idx = self.next_phase_idx
                self.state = 'GREEN'
                self.last_change_time = now
                self._apply_lights_hard(self.current_phase_idx)
        elif self.state == 'GREEN':
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

    def _apply_lights_hard(self, green_idx):
        green_dir = self.phases[green_idx]
        for d, group in self.light_groups.items():
            st = carla.TrafficLightState.Green if d == green_dir else carla.TrafficLightState.Red
            for tl in group: tl.set_state(st)
    
    def _set_lights(self, phase_idx, state):
        direction = self.phases[phase_idx]
        for tl in self.light_groups[direction]: tl.set_state(state)

    def is_min_green_respected(self, now):
        return (now - self.last_change_time) > config.MIN_GREEN_TIME

    def get_sensors_data(self, vehicles, now):
        data = {
            'queues': {'N':0, 'S':0, 'E':0, 'O':0},
            'wait_times': {'N':0.0, 'S':0.0, 'E':0.0, 'O':0.0},
            'emergencies': {'N':False, 'S':False, 'E':False, 'O':False},
            'current_phase': self.current_phase_idx,
            'junction_occupied': False 
        }
        identity = carla.Transform()
        
        # Reset stato interno
        junction_busy = False

        for v_data in vehicles:
            actor = v_data['actor']
            if not actor.is_alive: continue
            loc = actor.get_location()
            
            # 1. Check Box Centrale (Anti-Gridlock)
            if self.junction_box.contains(loc, identity):
                junction_busy = True
                # Linea Blu per indicare che questo veicolo sta bloccando il centro
                self.world.debug.draw_line(loc, self.junction_box.location, thickness=0.1, color=carla.Color(0,0,255), life_time=0.1)

            # 2. Check Detection Boxes (Corsie)
            for key, box in self.detection_boxes.items():
                if box.contains(loc, identity):
                    data['queues'][key] += 1
                    
                    vel = utils.vehicle_speed(actor)
                    wait = 0.0
                    if vel < 1.0: 
                        if actor.id not in self.vehicle_stop_times:
                            self.vehicle_stop_times[actor.id] = now
                        wait = now - self.vehicle_stop_times[actor.id]
                    else:
                        self.vehicle_stop_times.pop(actor.id, None)
                    
                    if wait > data['wait_times'][key]: data['wait_times'][key] = wait
                    
                    type_id = actor.type_id
                    if 'ambulance' in type_id or 'police' in type_id or 'firetruck' in type_id:
                        data['emergencies'][key] = True
                    
                    # Linea bianca debug
                    self.world.debug.draw_line(loc, box.location, thickness=0.1, color=carla.Color(255,255,255), life_time=0.1)
                    break 

        data['junction_occupied'] = junction_busy
        
        self.current_queues = data['queues']
        self.current_waits  = data['wait_times']
        self.current_emergencies = data['emergencies']
        self.junction_occupied = junction_busy 
        
        return data
    