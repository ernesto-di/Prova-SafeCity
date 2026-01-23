import carla
import config
import env.utils as utils
import torch
import numpy as np
from traffic.smart_intersection import SmartIntersection
from network.network import DualHeadDQN

class TrafficLightManager:
    def __init__(self, world, model):
        self.world = world
        self.model = model                              # ### CHANGED ###
        self.device = next(model.parameters()).device   # ### CHANGED ###

        self.intersections = []

        self._scan_intersections()

    def _scan_intersections(self):
        # 1. Recupera TUTTI i semafori
        all_tls = self.world.get_actors().filter('traffic.traffic_light')
        box_tls = []
        outside_tls = []

        # 2. Filtra interni/esterni usando la bounding box globale
        for tl in all_tls:
            if utils.is_point_in_box(tl.get_location()):
                box_tls.append(tl)
            else:
                outside_tls.append(tl)

        # 3. Esterni -> Verde Fisso
        for tl in outside_tls:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
        
        # 4. Raggruppamento per Incrocio
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

        # ORDINAMENTO: Ordina i gruppi da Sinistra a Destra (X crescente)
        groups.sort(key=lambda g: sum([t.get_location().x for t in g])/len(g))

        idx_count = 0
        for grp in groups:
            if len(grp) >= config.MIN_LIGHTS_PER_JUNCTION:
                self.intersections.append(SmartIntersection(idx_count, grp, self.world))
                idx_count += 1
            else:
                for tl in grp:
                    tl.set_state(carla.TrafficLightState.Green)
                    tl.freeze(True)
        utils.log(f"Manager: Attivati {len(self.intersections)} incroci intelligenti (Ordinati SX->DX).")

    def _get_state_vector(self, world_state):
        """Converte lo stato del mondo in un tensore per la rete neurale"""
        state_vec = []
        target_intersections = 2
        dirs = ['N', 'S', 'E', 'O']
        
        for i in range(target_intersections):
            if i < len(world_state):
                data = world_state[i]
                phase_idx = data['current_phase']
            else:
                data = {'queues': {d:0 for d in dirs}, 
                        'wait_times': {d:0 for d in dirs}, 
                        'emergencies': {d:False for d in dirs},
                        'current_phase': 0}
                phase_idx = 0

            # 1. Phase (One-Hot)
            phase_vec = [0.0] * 4
            phase_vec[phase_idx] = 1.0
            state_vec.extend(phase_vec)

            # 2. Wait Times
            state_vec.extend([data['wait_times'][d] / 60.0 for d in dirs])

            # 3. Emergenze
            state_vec.extend([1.0 if data['emergencies'][d] else 0.0 for d in dirs])

            # 4. Code
            state_vec.extend([data['queues'][d] / 20.0 for d in dirs])

        return np.array(state_vec, dtype=np.float32)

    def run_step(self, vehicles_list, now):
        world_state = []

        for incrocio in self.intersections:
            data = incrocio.get_sensors_data(vehicles_list, now)
            world_state.append(data)
            
            # --- VISUALIZZAZIONE DEBUG INTEGRATA ---
            # Disegna i box anche qui, per sicurezza, se il loop principale non lo fa
            incrocio.draw_setup_debug()

        state_vec = self._get_state_vector(world_state)
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        
        out0, out1 = self.model(state_t)

        action = [
            out0.argmax(dim=1).item(),
            out1.argmax(dim=1).item()
        ]

        for i, incrocio in enumerate(self.intersections):
            if i < len(action):
                incrocio.update(action[i], now, vehicles_list)

        return world_state, action