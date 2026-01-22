import carla
import config
import utils
import torch
import numpy as np
from smart_intersection import SmartIntersection
from network import DualHeadDQN  # Importiamo la rete dal nuovo file

class TrafficLightManager:
    def __init__(self, world):
        self.world = world
        self.intersections = []
        self.last_print_time = 0
        
        # --- CARICAMENTO AI ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualHeadDQN(input_dim=32, output_dim=4).to(self.device)
        self.use_ai = False
        
        try:
            # Tenta di caricare i pesi addestrati
            self.model.load_state_dict(torch.load("modello.pth", map_location=self.device))
            self.model.eval() # Imposta in modalità inferenza (no training)
            self.use_ai = True
            utils.log("MODELLO AI CARICATO CORRETTAMENTE! Uso il cervello neurale.")
        except Exception as e:
            utils.log(f"ATTENZIONE: Modello non trovato o errore ({e}). Uso logica di base (ai_logic).")

        self._scan_intersections()

    def _scan_intersections(self):
        # 1. Recupera TUTTI i semafori della mappa
        all_tls = self.world.get_actors().filter('traffic.traffic_light')
        box_tls = []
        outside_tls = []

        # 2. Separali in base alla Bounding Box
        for tl in all_tls:
            if utils.is_point_in_box(tl.get_location()):
                box_tls.append(tl)
            else:
                outside_tls.append(tl)

        # 3. GESTIONE SEMAFORI ESTERNI: Imposta SEMPRE VERDE
        count_out = 0
        for tl in outside_tls:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
            count_out += 1
        
        # 4. GESTIONE SEMAFORI INTERNI
        groups = []
        processed = set()
        GROUP_DISTANCE = 45.0

        for tl in box_tls:
            if tl.id in processed: continue
            current_group = [tl]
            processed.add(tl.id)
            tl_loc = tl.get_location()
            for other in box_tls:
                if other.id not in processed:
                    if tl_loc.distance(other.get_location()) < GROUP_DISTANCE:
                        current_group.append(other)
                        processed.add(other.id)
            groups.append(current_group)

        idx_count = 0
        for grp in groups:
            if len(grp) >= config.MIN_LIGHTS_PER_JUNCTION:
                self.intersections.append(SmartIntersection(idx_count, grp, self.world))
                idx_count += 1
            else:
                for tl in grp:
                    tl.set_state(carla.TrafficLightState.Green)
                    tl.freeze(True)
        utils.log(f"Manager: Attivati {len(self.intersections)} incroci intelligenti.")

    def _get_state_vector(self, world_state):
        """Converte lo stato del mondo in un tensore per la rete neurale"""
        state_vec = []
        # Assumiamo che il modello si aspetti dati per 2 incroci (come nel training)
        target_intersections = 2
        
        for i in range(target_intersections):
            if i < len(world_state):
                data = world_state[i]
                phase_idx = data['current_phase']
            else:
                # Padding se manca un incrocio
                data = {'queues': {'N':0,'S':0,'E':0,'O':0}, 
                        'wait_times': {'N':0,'S':0,'E':0,'O':0}, 
                        'emergencies': {'N':False,'S':False,'E':False,'O':False},
                        'current_phase': 0}
                phase_idx = 0

            # 1. Phase (One-Hot)
            phase_vec = [0.0] * 4
            phase_vec[phase_idx] = 1.0
            state_vec.extend(phase_vec)

            # 2. Wait Times (normalizzati)
            state_vec.extend([data['wait_times'][d] / 60.0 for d in ['N','S','E','O']])

            # 3. Emergenze
            state_vec.extend([1.0 if data['emergencies'][d] else 0.0 for d in ['N','S','E','O']])

            # 4. Code (normalizzate)
            state_vec.extend([data['queues'][d] / 20.0 for d in ['N','S','E','O']])

        return np.array(state_vec, dtype=np.float32)

    def run_step(self, vehicles_list, now):
        world_state = []
        for incrocio in self.intersections:
            world_state.append(incrocio.get_sensors_data(vehicles_list, now))

        ai_decisions = []
        if self.use_ai:
            state_vec = self._get_state_vector(world_state)
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out0, out1 = self.model(state_t)
                ai_decisions = [out0.argmax().item(), out1.argmax().item()]
        else:
            from ai_logic import ai_brain_decision
            ai_decisions = ai_brain_decision(world_state)

        for i, incrocio in enumerate(self.intersections):
            if i < len(ai_decisions):
                # MODIFICA QUI: passiamo vehicles_list all'update!
                incrocio.update(ai_decisions[i], now, vehicles_list)

        # Report Console (ogni 5s)
        if now - self.last_print_time > 5.0:
            print("-" * 50)
            mode_str = "AI NEURALE" if self.use_ai else "LOGICA BASE"
            print(f"[REPORT 5s] Stato Traffico - Modalità: {mode_str}")
            dir_names = ['N', 'S', 'E', 'O']
            for i, data in enumerate(world_state):
                q = data['queues']
                curr = dir_names[data['current_phase']]
                emerg = " [!!! EMERGENZA !!!]" if any(data['emergencies'].values()) else ""
                print(f"  > Incrocio {i} (Verde su {curr}): Code N={q['N']} S={q['S']} E={q['E']} O={q['O']}{emerg}")
            print("-" * 50)
            self.last_print_time = now