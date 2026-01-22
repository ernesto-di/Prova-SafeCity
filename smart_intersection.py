import carla
import config
import utils
from smart_intersection import SmartIntersection
from ai_logic import ai_brain_decision

class TrafficLightManager:
    def __init__(self, world):
        self.world = world
        self.intersections = []
        self.last_print_time = 0
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

        utils.log(f"DEBUG: {count_out} semafori FUORI dal box impostati su VERDE permanente.")
        utils.log(f"DEBUG: Trovati {len(box_tls)} semafori DENTRO l'area da gestire.")

        # 4. GESTIONE SEMAFORI INTERNI (Raggruppamento per Incroci)
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
                # Anche i gruppi spuri DENTRO il box vengono messi a verde
                for tl in grp:
                    tl.set_state(carla.TrafficLightState.Green)
                    tl.freeze(True)
        utils.log(f"Manager: Attivati {len(self.intersections)} incroci intelligenti.")

    def run_step(self, vehicles_list, now):
        world_state = []
        for incrocio in self.intersections:
            world_state.append(incrocio.get_sensors_data(vehicles_list, now))

        ai_decisions = ai_brain_decision(world_state)

        for i, incrocio in enumerate(self.intersections):
            incrocio.update(ai_decisions[i], now)

        if now - self.last_print_time > 5.0:
            print("-" * 50)
            print(f"[REPORT 5s] Stato Traffico (Tempo: {now % 100:.1f}s)")
            dir_names = ['N', 'S', 'E', 'O']
            for i, data in enumerate(world_state):
                q = data['queues']
                w = data['wait_times']
                curr_phase = dir_names[data['current_phase']]

                # Info Logica per capire cosa sta succedendo
                logic_msg = ""
                if config.INVERT_DIRECTION_LOGIC:
                    logic_msg = f"(Logica Inversa: Verde su {curr_phase} serve coda opposta)"

                emerg_info = ""
                for d, active in data['emergencies'].items():
                    if active: emerg_info += f" [EMERGENZA DA {d}!]"
                
                # Format string for queues + waits
                q_str = f"N={q['N']}({w['N']:.1f}s) S={q['S']}({w['S']:.1f}s) E={q['E']}({w['E']:.1f}s) O={q['O']}({w['O']:.1f}s)"

                print(
                    f"  > Incrocio {i} {logic_msg} | Code: {q_str}{emerg_info}")
            print("-" * 50)
            self.last_print_time = now
