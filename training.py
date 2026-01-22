import carla
import random
import time
import math
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config
import utils
from traffic_light_manager import TrafficLightManager
import matplotlib
matplotlib.use('Agg') # Backend non-interattivo per evitare finestre vuote
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIG & MODES
# ==============================================================================
TRAINING_MODE = 'SHORT'  # Options: 'SHORT', 'LONG'

# ==============================================================================
# DQN MODEL
# ==============================================================================
class DualHeadDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DualHeadDQN, self).__init__()
        # Input: 32 (16 per intersection * 2 intersections)
        
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Head per Incrocio 0
        self.head0 = nn.Linear(64, output_dim)
        # Head per Incrocio 1
        self.head1 = nn.Linear(64, output_dim)

    def forward(self, x):
        features = self.common(x)
        out0 = self.head0(features)
        out1 = self.head1(features)
        return out0, out1

# ==============================================================================
# ENVIRONMENT
# ==============================================================================
# ==============================================================================
# ENVIRONMENT COMPLETO (FIXED)
# ==============================================================================
class TrafficEnv:
    def __init__(self):
        self.client = carla.Client(config.HOST, config.PORT)
        self.client.set_timeout(config.CLIENT_TIMEOUT)

        # FIX: Carica la mappa UNA SOLA VOLTA all'inizio
        print("Inizializzazione Environment: Caricamento mondo Town05...")
        self.client.load_world('Town05')

        self.world = self.client.get_world()

        # Setup visuale debug
        utils.draw_bounding_box(self.world)
        utils.draw_spawn_points(self.world)

        # Configurazione TM
        self.tm_port = 8005
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_global_distance_to_leading_vehicle(3.0)
        self.tm.set_synchronous_mode(config.SYNCHRONOUS_MODE)
        self.tm.set_respawn_dormant_vehicles(False) # Prevent Carla from destroying stuck vehicles

        # Configurazione Sincrona Mondo
        if config.SYNCHRONOUS_MODE:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = config.FIXED_DELTA_SECONDS
            if config.NO_RENDERING_MODE:
                settings.no_rendering_mode = True
            self.world.apply_settings(settings)

        # Gestore semafori
        self.manager = TrafficLightManager(self.world)
        if len(self.manager.intersections) < 2:
            print("WARNING: Trovati meno di 2 incroci. Il training richiede 2 incroci.")

        # Blueprints spawn
        bp_lib = self.world.get_blueprint_library()
        self.emerg_bps = [x for x in bp_lib.filter('vehicle.*') if any(k in x.id for k in ['police', 'ambulance', 'firetruck'])]
        self.norm_bps = [x for x in bp_lib.filter('vehicle.*') if
                         int(x.get_attribute('number_of_wheels')) == 4 and 
                         not any(k in x.id for k in ['police', 'ambulance', 'firetruck'])]

        # Inizializzazione variabili stato
        self.vehicles_data = []
        self.spawn_timers = {}
        self.spawn_counter = 0
        self.cars_finished = 0
        self.last_cleanup_time = 0.0
        self.last_stuck_check_time = 0.0 # [OPT] Timer per ottimizzazione check pesanti
        self.last_sensor_data = []  # Cache per il reward

    def reset(self):
        print("Reset dell'ambiente (Soft Reset)...")

        # 1. PULIZIA ATTORI ESISTENTI (Senza ricaricare la mappa)
        if self.vehicles_data:
            # Filtra solo quelli ancora vivi per evitare errori
            active_actors = [x['actor'] for x in self.vehicles_data if x['actor'].is_alive]
            if active_actors:
                batch = [carla.command.DestroyActor(x) for x in active_actors]
                self.client.apply_batch(batch)

        # Pulizia completa di sicurezza (rimuove eventuali veicoli orfani)
        all_vehicles = self.world.get_actors().filter('vehicle.*')
        if len(all_vehicles) > 0:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in all_vehicles])

        # Reset variabili
        self.vehicles_data = []
        self.spawn_timers = {}
        self.spawn_counter = 0
        self.cars_finished = 0

        # 2. RESET SEMAFORI
        for incrocio in self.manager.intersections:
            incrocio.state = 'GREEN'
            incrocio.current_phase_idx = 0
            incrocio.next_phase_idx = 0
            incrocio._apply_lights_hard(0)
            incrocio.last_change_time = 0.0
            incrocio.vehicle_stop_times = {}

        # Attesa tecnica per propagazione distruzioni
        for _ in range(10): 
            if config.SYNCHRONOUS_MODE: 
                self.world.tick()
            else:
                time.sleep(0.05)

        # Ritorna lo stato iniziale
        self.prev_wait_times = [] # Reset storico tempi attesa
        return self._get_state()

    def step(self, action_tuple):
        """
        action_tuple: (action_idx_int0, action_idx_int1)
        Esegue un passo di simulazione SINCRONIZZATO con i semafori.
        Il sistema avanza finché TUTTI gli incroci non hanno completato il loro ciclo 
        (cioè sono VERDI da almeno 7 secondi).
        """
        accumulated_reward = 0.0
        step_count = 0
        MAX_WAIT_STEPS = 2000 # Safety breakout (200 secondi)

        # 1. Applica azioni (decisione AI)
        # Nota: L'azione viene applicata subito. Se comporta un cambio fase,
        # l'incrocio passerà a YELLOW -> RED -> GREEN.
        now = 0.0
        if config.SYNCHRONOUS_MODE:
            now = self.world.get_snapshot().timestamp.elapsed_seconds
        else:
            now = time.time()

        for i, incrocio in enumerate(self.manager.intersections):
            if i < len(action_tuple):
                incrocio.update(action_tuple[i], now)

        # 2. Loop di attesa dinamica
        # Continuiamo a simouvare finché non siamo pronti per la prossima decisione.
        # Condizione: TUTTI gli incroci devono essere GREEN e aver rispettato il MIN_GREEN_TIME.
        
        while True:
            # Aggiorna tempo corrente
            if config.SYNCHRONOUS_MODE:
                snapshot = self.world.get_snapshot()
                now = snapshot.timestamp.elapsed_seconds
            else:
                now = time.time()

            # Gestione interna semafori (transizioni G->Y->R->G)
            # Nota: update viene chiamato con l'azione CORRENTE (che non cambia durante l'attesa)
            # Ma serve per far avanzare la macchina a stati finiti interna (es. scattare da Yellow a Red)
            running_transition = False
            for i, incrocio in enumerate(self.manager.intersections):
                # Passiamo -1 o l'indice corrente? 
                # In realtà update controlla il cambio solo se siamo in GREEN.
                # Se siamo in YELLOW/RED, avanza da solo.
                # Per sicurezza ripassiamo l'azione corrente, ma se siamo in transizione verrà ignorata.
                current_action = action_tuple[i] if i < len(action_tuple) else 0
                incrocio.update(current_action, now)

                if not incrocio.is_min_green_respected(now):
                    running_transition = True

            # Gestione Spawn e Cleanup
            cars_finished_before = self.cars_finished
            self._manage_traffic(now)
            delta_finished_cars = self.cars_finished - cars_finished_before

            # Step Fisico
            if config.SYNCHRONOUS_MODE:
                self.world.tick()
            else:
                time.sleep(config.AI_STEP_INTERVAL)
            
            # Check Stuck e Calcolo Reward Parziale
            dt = config.FIXED_DELTA_SECONDS if config.SYNCHRONOUS_MODE else config.AI_STEP_INTERVAL
            self._check_stuck_vehicles(dt, now)
            
            # Calcolo reward per questo singolo tick
            # Nota: _get_state aggiorna self.last_sensor_data che serve a _compute_reward
            # Dobbiamo chiamarlo per aggiornare i dati interni, anche se lo stato ritornato non lo usiamo qui.
            self._get_state() 
            step_reward = self._compute_reward(delta_finished_cars)
            accumulated_reward += step_reward

            step_count += 1

            # BREAK CONDITION: Se tutti gli incroci sono stabili (Verde > 7s), usciamo.
            if not running_transition:
                break
            
            # SAFETY BREAK
            if step_count >= MAX_WAIT_STEPS:
                print("WARNING: Max wait steps reached in synchronization loop!")
                break

        # 4. Ottieni stato finale e reward totale accumulato
        next_state = self._get_state()
        done = False 

        return next_state, accumulated_reward, done, {}

    def _manage_traffic(self, now):
        # Update lista veicoli vivi
        try:
            # Controlla solo attori noti per efficienza
            self.vehicles_data = [d for d in self.vehicles_data if d['actor'].is_alive]
        except:
            self.vehicles_data = []

        # Cleanup globale periodico
        if now - self.last_cleanup_time > config.GLOBAL_CLEANUP_INTERVAL:
            removed_ids = utils.perform_global_cleanup(self.world, self.client, self.tm_port)
            if removed_ids:
                self.cars_finished += len(removed_ids)
                # Rimozione immediata dalla lista locale
                removed_set = set(removed_ids)
                self.vehicles_data = [d for d in self.vehicles_data if d['actor'].id not in removed_set]
            
            self.last_cleanup_time = now

        # Spawning
        if len(self.vehicles_data) < config.MAX_CONCURRENT_VEHICLES and self.spawn_counter < config.MAX_TOTAL_VEHICLES:
            indices = [i for i in range(len(config.FIXED_SPAWN_POINTS)) if now > self.spawn_timers.get(i, 0)]
            if indices:
                idx = random.choice(indices)
                sp = config.FIXED_SPAWN_POINTS[idx]

                # Logica emergenza: ogni 30 veicoli, uno è emergenza (se > 0)
                is_emerg = (self.spawn_counter % 30 == 0) and (self.spawn_counter > 0)
                bp = random.choice(self.emerg_bps) if is_emerg else random.choice(self.norm_bps)

                veh = self.world.try_spawn_actor(bp, sp)
                if veh:
                    veh.set_autopilot(True, self.tm_port)
                    if is_emerg:
                        self.tm.ignore_lights_percentage(veh, 0.0) # [MOD] Ora rispettano i semafori per evitare deadlock
                        veh.set_light_state(carla.VehicleLightState.Special1)
                    else:
                        self.tm.ignore_lights_percentage(veh, 0.0)

                    self.vehicles_data.append({
                        'actor': veh,
                        'stuck_timer': 0.0,
                        'offset_active': False
                    })
                    self.spawn_timers[idx] = now + config.SPAWN_RETRY_TIME
                    self.spawn_counter += 1
                else:
                    self.spawn_timers[idx] = now + 2.0

    def _check_stuck_vehicles(self, dt, now):
        """
        Risoluzione Deadlock: Se un veicolo è fermo ma teoricamente potrebbe muoversi 
        (es. verde o centro incrocio), applica un offset laterale per "schivare" ostacoli.
        
        [OPTIMIZATION] I check pesanti (_has_vehicle_ahead, _is_potential_collision) sono O(N^2).
        Li eseguiamo solo ogni 0.5s e usiamo il risultato cachato.
        """
        STUCK_VEL = 0.5   # m/s
        STUCK_LIM = 4.0   # secondi
        OFFSET_VAL = 1.3  # metri a destra
        CHECK_INTERVAL = 0.5 # secondi

        run_heavy_checks = (now - self.last_stuck_check_time) > CHECK_INTERVAL
        if run_heavy_checks:
            self.last_stuck_check_time = now

        for data in self.vehicles_data:
            actor = data['actor']
            if not actor.is_alive:
                continue
            
            # Calcolo velocità
            vel = actor.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2)

            # Controlla se è fermo a un semaforo ROSSO (in quel caso è OK stare fermi)
            is_stopped_red = False
            if actor.is_at_traffic_light():
                state = actor.get_traffic_light_state()
                if state == carla.TrafficLightState.Red or state == carla.TrafficLightState.Yellow:
                    is_stopped_red = True
            
            # Fallback: Se Carla dice False, controlliamo se c'è un semaforo vicino (es. 15m -> 35m) che è Rosso/Giallo
            # Esteso a 35m per supportare semafori posti "dall'altra parte" dell'incrocio (Town05)
            if not is_stopped_red:
                actor_loc = actor.get_location()
                actor_fwd = actor.get_transform().get_forward_vector()
                
                for incrocio in self.manager.intersections:
                    if actor_loc.distance(incrocio.center) > 50.0: 
                        continue
                    
                    for direction, lights in incrocio.light_groups.items():
                        for tl in lights:
                            tl_loc = tl.get_location()
                            dist = tl_loc.distance(actor_loc)
                            
                            if dist < 35.0:
                                # CHECK DIREZIONALE: Il semaforo deve essere "davanti" (+- 60 gradi)
                                vec_to_tl = tl_loc - actor_loc
                                # Normalizzo
                                if dist > 0.1:
                                    vec_to_tl = carla.Vector3D(vec_to_tl.x / dist, vec_to_tl.y / dist, 0)
                                    dot = actor_fwd.x * vec_to_tl.x + actor_fwd.y * vec_to_tl.y
                                    
                                    if dot > 0.5: # Angolo < 60 gradi circa
                                         if tl.get_state() in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
                                            is_stopped_red = True
                                            # print(f"DEBUG: Veicolo {actor.id} rilevato fermo al semaforo (FALLBACK). Dist: {dist:.1f}m")
                                            break
                        if is_stopped_red: break
                    if is_stopped_red: break

            # [OPTIMIZATION]
            # 1. "Completamente fermo": Speed < 0.1 (quasi 0)
            # 2. "Non ha auto davanti": Check _has_vehicle_ahead
            # 3. Solo se queste condizioni sono vere, potremmo fare altri check, ma in realtà:
            #    Se è fermo, non è al rosso, e NON ha auto davanti -> È STUCK (strada libera ma fermo).
            #    Se ha auto davanti -> È in CODA (non stuck).
            
            should_check_stuck = (speed < 0.1) and (not is_stopped_red)

            # Aggiornamento cache condizioni pesanti
            if run_heavy_checks:
                if should_check_stuck:
                    has_lead = self._has_vehicle_ahead(actor, max_dist=8.0)
                    
                    if has_lead:
                         # Se ha auto davanti, è in una normale coda -> NON attivare offset
                         data['cached_stuck_cond'] = False
                    else:
                         # Se NON ha auto davanti, ma è fermo:
                         # Controlliamo se c'è una collisione "fantasma" (es. bus incrociati).
                         # Solo in questo caso attiviamo l'offset.
                         data['cached_stuck_cond'] = self._is_potential_collision(actor)
                else:
                    data['cached_stuck_cond'] = False # Si muove o è al rosso -> OK

            # Recupera condizione da cache
            stuck_condition_met = data.get('cached_stuck_cond', False)

            # Timer incrementa solo se siamo VERAMENTE fermi e la condizione di blocco persiste
            if should_check_stuck and stuck_condition_met:
                data['stuck_timer'] = data.get('stuck_timer', 0.0) + dt
            else:
                data['stuck_timer'] = 0.0
                # Reset offset se riparte
                if data.get('offset_active', False):
                    self.tm.vehicle_lane_offset(actor, 0.0)
                    data['offset_active'] = False
            
            # Applicazione Offset
            if data['stuck_timer'] > STUCK_LIM:
                if not data.get('offset_active', False):
                    loc = actor.get_location()
                    print(f"DEBUG: Unstucking Vehicle {actor.id} at ({loc.x:.1f}, {loc.y:.1f}) (Offset {OFFSET_VAL}m)")
                    self.tm.vehicle_lane_offset(actor, OFFSET_VAL)
                    data['offset_active'] = True

    def _is_potential_collision(self, actor, dist_margin=1.0): # Margine ridotto da 2.0 a 1.0
        try:
            loc = actor.get_location()
            fwd = actor.get_transform().get_forward_vector()
            # extent.x è metà lunghezza, extent.y è metà larghezza
            extent_x = actor.bounding_box.extent.x 
            extent_y = actor.bounding_box.extent.y
        except:
            return False

        for other_data in self.vehicles_data:
            other = other_data['actor']
            if other.id == actor.id or not other.is_alive: continue
            
            try:
                other_loc = other.get_location()
                other_fwd = other.get_transform().get_forward_vector()
                dist = loc.distance(other_loc)
                
                # Calcoliamo l'angolo tra i due veicoli (Dot Product dei vettori Forward)
                # 1.0 = Paralleli (stessa direzione), 0.0 = Perpendicolari, -1.0 = Opposti
                alignment = fwd.x * other_fwd.x + fwd.y * other_fwd.y
                
                other_extent_x = other.bounding_box.extent.x
                other_extent_y = other.bounding_box.extent.y

                # LOGICA MIGLIORATA:
                # Se sono paralleli (si stanno affiancando o seguendo), usiamo la larghezza (Y)
                # Se sono perpendicolari (incrocio), usiamo la lunghezza (X)
                
                is_parallel = abs(alignment) > 0.7 
                
                if is_parallel:
                    # Se sono paralleli, il pericolo è scontrarsi lateralmente
                    # Usiamo la larghezza + un margine piccolissimo
                    collision_dist = extent_y + other_extent_y + 0.5 
                    
                    # Controllo laterale specifico (vettori perpendicolari)
                    # (Semplificazione: usiamo distanza euclidea ma con soglia ridotta)
                    if dist < collision_dist:
                        return True
                else:
                    # Se sono perpendicolari (incrocio), serve spazio per passare
                    collision_dist = extent_x + other_extent_x + dist_margin
                    if dist < collision_dist:
                        # Controlla se è davanti
                        vec = other_loc - loc
                        dot = vec.x * fwd.x + vec.y * fwd.y
                        if dot > 0: return True

            except:
                continue
        return False

    def _has_vehicle_ahead(self, actor, max_dist=6.0):
        """
        Controlla se c'è un veicolo davanti a breve distanza (simulazione sensore)
        """
        try:
            p1 = actor.get_location()
            fwd = actor.get_transform().get_forward_vector()
        except RuntimeError:
            return False

        for other_data in self.vehicles_data:
            other = other_data['actor']
            if other.id == actor.id: # Skip self
                continue
            
            if not other.is_alive:
                continue

            try:
                p2 = other.get_location()
            except RuntimeError:
                continue
            
            # Distanza euclidea grezza
            dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            
            if dist < max_dist:
                # Calcola se è "davanti" usando il dot product
                # Vettore verso l'altro veicolo
                vec_to_other = carla.Vector3D(p2.x - p1.x, p2.y - p1.y, 0)
                
                # Dot product posizionale: > 0 se l'altro è nel semispazio davanti
                dot_pos = fwd.x * vec_to_other.x + fwd.y * vec_to_other.y
                
                if dot_pos > 0:
                    # È geometricamente davanti.
                    # ORA CONTROLLIAMO L'ORIENTAMENTO: È una coda o traffico opposto?
                    fwd_other = other.get_transform().get_forward_vector()
                    
                    # Dot product direzionale: 
                    # ~1 (stessa direzione), ~-1 (direzione opposta), ~0 (perpendicolare)
                    dot_dir = fwd.x * fwd_other.x + fwd.y * fwd_other.y
                    
                    # Se dot_dir > 0.0, vanno grossomodo nella stessa direzione (angolo < 90°)
                    # Se dot_dir < 0.0, viene incontro o è molto angolato -> Non è coda, è l'ostacolo!
                    if dot_dir > 0.3: # Soglia di tolleranza (circa 70 gradi)
                        return True # È un veicolo della mia coda
        return False

    def _get_state(self):
        if config.SYNCHRONOUS_MODE:
             now = self.world.get_snapshot().timestamp.elapsed_seconds
        else:
             now = time.time()
        state_vec = []

        target_intersections = 2
        current_data_list = []

        for i in range(target_intersections):
            if i < len(self.manager.intersections):
                incrocio = self.manager.intersections[i]
                data = incrocio.get_sensors_data(self.vehicles_data, now)
                phase_idx = data['current_phase']
            else:
                data = {
                    'queues': {'N': 0, 'S': 0, 'E': 0, 'O': 0},
                    'wait_times': {'N': 0, 'S': 0, 'E': 0, 'O': 0},
                    'emergencies': {'N': False, 'S': False, 'E': False, 'O': False},
                    'current_phase': 0
                }
                phase_idx = 0

            # --- COSTRUZIONE VETTORE ---
            dirs = ['N', 'S', 'E', 'O']

            # 1. Stato (Phase One-Hot)
            phase_vec = [0.0] * 4
            phase_vec[phase_idx] = 1.0
            state_vec.extend(phase_vec)

            # 2. Wait Times (norm)
            state_vec.extend([data['wait_times'][d] / 60.0 for d in dirs])

            # 3. Veicoli Emergenza
            state_vec.extend([1.0 if data['emergencies'][d] else 0.0 for d in dirs])

            # 4. Lunghezza Coda (norm)
            state_vec.extend([data['queues'][d] / 20.0 for d in dirs])

            current_data_list.append(data)

        self.last_sensor_data = current_data_list
        return np.array(state_vec, dtype=np.float32)
    
    def _compute_reward(self, delta_finished_cars):
        total_reward = 0.0
        
        # --- CONFIGURAZIONE PESI ---
        W_THROUGHPUT = 80.0       # Premio per ogni auto uscita
        W_EMERGENCY  = 5000.0     # PRIORITÀ ASSOLUTA (Valore enorme)
        W_PRESSURE   = 10.0       # Peso della logica code
        
        # --- 1. Throughput ---
        total_reward += delta_finished_cars * W_THROUGHPUT

        # --- Analisi Situazione ---
        current_data_list = self.last_sensor_data 

        for data in current_data_list:
            queues = data['queues']
            wait_times = data['wait_times']
            emergencies = data['emergencies']
            current_phase_idx = data['current_phase']
            
            phase_to_dir = {0: 'N', 1: 'S', 2: 'E', 3: 'O'}
            green_dir = phase_to_dir.get(current_phase_idx, 'N')
            
            # --- 2. GESTIONE EMERGENZE (Override Totale) ---
            any_emergency = False
            for d, is_active in emergencies.items():
                if is_active:
                    any_emergency = True
                    if d == green_dir:
                        # Bravo! Hai dato verde all'ambulanza
                        total_reward += 200.0 
                    else:
                        # DISASTRO! Ambulanza ferma al rosso.
                        # Punizione tale da azzerare qualsiasi altro ragionamento.
                        total_reward -= W_EMERGENCY
            
            # Se c'è un'emergenza gestita male, saltiamo la logica code (la priorità è solo quella)
            if any_emergency:
                continue

            # --- 3. LOGICA DI PRESSIONE (Gestione Code) ---
            # Calcoliamo quante auto stiamo servendo (Outgoing) vs quante aspettano (Incoming)
            
            # Auto che stanno passando (sulla corsia verde)
            # Se la coda è 0, stiamo servendo "il vuoto" -> Efficienza bassa
            cars_served = queues[green_dir]
            
            # Auto che stanno aspettando (la peggiore delle corsie rosse)
            max_waiting_red = 0
            for d in ['N', 'S', 'E', 'O']:
                if d != green_dir:
                    if queues[d] > max_waiting_red:
                        max_waiting_red = queues[d]
            
            # PRESSIONE = (Chi aspetta) - (Chi passa)
            # Esempio Sbagliato: Aspettano 10, Passano 0. Pressione = 10 (Punizione alta)
            # Esempio Giusto: Aspettano 2, Passano 15. Pressione = -13 (Premio)
            pressure = max_waiting_red - cars_served
            
            # Applichiamo la punizione basata sulla pressione
            total_reward -= (pressure * W_PRESSURE)

            # --- 4. SPRECO DI TEMPO (Verde Vuoto) ---
            # Se ho il verde ma non c'è coda, e altrove c'è gente, punizione extra immediata
            if cars_served == 0 and max_waiting_red > 0:
                total_reward -= 50.0

            # --- 5. LIMITE ATTESA MASSIMA ---
            # Anche se la pressione suggerisce di non cambiare, non lasciare nessuno > 90s
            for d in ['N', 'S', 'E', 'O']:
                if wait_times[d] > 90.0:
                    total_reward -= 100.0 # Punizione per "dimenticanza"

        return total_reward / 100.0
# ==============================================================================
# REPLAY BUFFER & AGENT
# ==============================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, input_dim, output_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DualHeadDQN(input_dim, output_dim).to(self.device)
        self.target_net = DualHeadDQN(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(10000)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.985

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3), random.randint(0, 3)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            out0, out1 = self.policy_net(state_t)
            return out0.argmax().item(), out1.argmax().item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state_batch = torch.FloatTensor(np.array(state)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state)).to(self.device)
        reward_batch = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # Action è una lista di tuple (a0, a1). Dobbiamo separarle
        a0_list = [a[0] for a in action]
        a1_list = [a[1] for a in action]
        action0_batch = torch.LongTensor(a0_list).unsqueeze(1).to(self.device)
        action1_batch = torch.LongTensor(a1_list).unsqueeze(1).to(self.device)

        # Q(s, a) corrente
        curr_out0, curr_out1 = self.policy_net(state_batch)
        q_val0 = curr_out0.gather(1, action0_batch)
        q_val1 = curr_out1.gather(1, action1_batch)

        # V(s') dal target
        with torch.no_grad():
            next_out0, next_out1 = self.target_net(next_state_batch)
            next_val0 = next_out0.max(1)[0].unsqueeze(1)
            next_val1 = next_out1.max(1)[0].unsqueeze(1)
        
        # Loss combinata (media delle due loss)
        # Target Q
        exp_q0 = reward_batch + (1 - done_batch) * self.gamma * next_val0
        exp_q1 = reward_batch + (1 - done_batch) * self.gamma * next_val1
        
        loss0 = F.mse_loss(q_val0, exp_q0)
        loss1 = F.mse_loss(q_val1, exp_q1)
        loss = loss0 + loss1
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
def train():
    env = TrafficEnv()
    agent = Agent(input_dim=32, output_dim=4)
    
    if TRAINING_MODE == 'SHORT':
        print(f"!!! TRAINING MODE: SHORT (Veloce) !!!")
        num_episodes = 30
        finished_cars_target = 200
    else:
        print(f"!!! TRAINING MODE: LONG (Completo) !!!")
        num_episodes = 200
        num_episodes = 200
        finished_cars_target = config.TARGET_FINISHED_CARS # Default 500

    print(f"Target veicoli per epoca: {finished_cars_target}")
    
    # Adattamento Epsilon Decay per SHORT mode
    if TRAINING_MODE == 'SHORT':
        # Vogliamo scendere da 1.0 a 0.05 in circa 30 epoche
        # 0.05 = 1.0 * (decay)^30  -> decay = 0.05^(1/30) ~= 0.905
        agent.epsilon_decay = 0.90
        print(f"Epsilon Decay adattato a: {agent.epsilon_decay}")
    
    print(f"Inizio training su {agent.device}...")
    
    episode_rewards = [] # Track rewards for plotting

    try:
        for e in range(num_episodes):
            epoch_start = time.time()
            state = env.reset()
            total_reward = 0
            
            step_count = 0
            while env.cars_finished < finished_cars_target:
                step_count += 1
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                agent.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                agent.optimize_model()
                
                # Print meno frequente dato che lo step ora dura molto di più (7s min)
                if step_count % 5 == 0:
                    print(f"  Ep {e} Dec {step_count} Reward: {reward:.2f} [Cars Finished: {env.cars_finished}]")
                
                # Safety break (adattato ai nuovi step lunghi)
                if step_count > 1000: # 1000 decisioni * 7s = 7000s = ~2 ore simulate
                    print("  WARNING: Max steps reached for this episode.")
                    break

            agent.update_target()
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            
            episode_rewards.append(total_reward) # Save reward
            
            epoch_duration = time.time() - epoch_start
            m, s = divmod(epoch_duration, 60)
            print(f"Episodio {e} completato. Total Reward: {total_reward:.2f}. Epsilon: {agent.epsilon:.2f}")
            print(f"Tempo Reale Epoca: {int(m)}m {int(s)}s")
            
            # Plot progressivo
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards, marker='o')
            plt.title('Training Rewards per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            plt.savefig('training_rewards.png')
            plt.close() # Chiudi la figura per liberare memoria
            
    except KeyboardInterrupt:
        print("Training interrotto.")
    finally:
        print("Salvataggio modello...")
        torch.save(agent.policy_net.state_dict(), "modello.pth")
        
        # Pulizia finale
        print("Pulizia environment...")
        batch = [carla.command.DestroyActor(x) for x in env.world.get_actors().filter('vehicle.*')]
        env.client.apply_batch(batch)

if __name__ == "__main__":
    train()
