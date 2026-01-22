import carla
import math
import time
import config

def log(msg):
    print(f"[AI-TRAFFIC] {time.time() % 100:.2f}: {msg}", flush=True)

def vehicle_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

def is_point_in_box(loc):
    return (config.MIN_X <= loc.x <= config.MAX_X) and (config.MIN_Y <= loc.y <= config.MAX_Y)

def get_yaw(actor):
    return actor.get_transform().rotation.yaw

def normalize_angle(angle):
    while angle < 0: angle += 360
    while angle >= 360: angle -= 360
    return angle

def get_direction_label(yaw):
    y = normalize_angle(yaw + config.TRAFFIC_LIGHT_CORRECTION)
    if 45 <= y < 135: return 'N'
    if 135 <= y < 225: return 'O'
    if 225 <= y < 315: return 'S'
    return 'E'

def draw_bounding_box(world):
    bbox_color = carla.Color(255, 0, 0)
    for i, v in enumerate(config.BOUNDING_BOX_VERTICES):
        p1 = carla.Location(x=v[0], y=v[1], z=0.5)
        v_next = config.BOUNDING_BOX_VERTICES[(i + 1) % len(config.BOUNDING_BOX_VERTICES)]
        p2 = carla.Location(x=v_next[0], y=v_next[1], z=0.5)
        world.debug.draw_line(p1, p2, 0.3, bbox_color, 3600.0)

def draw_spawn_points(world):
    spawn_color = carla.Color(0, 255, 0)
    for i, sp in enumerate(config.FIXED_SPAWN_POINTS):
        start_loc = sp.location + carla.Location(z=0.5)
        fwd_vec = sp.get_forward_vector()
        end_loc = start_loc + fwd_vec * 3.0
        world.debug.draw_arrow(start_loc, end_loc, thickness=0.1, arrow_size=0.3, color=spawn_color, life_time=3600.0)
        world.debug.draw_string(start_loc + carla.Location(z=1.0), f"SP {i}", draw_shadow=False, color=spawn_color,
                                life_time=3600.0)

def perform_global_cleanup(world, client, tm_port):
    all_actors = world.get_actors().filter('vehicle.*')
    batch_cmds = []
    batch_ids = []
    count = 0
    for actor in all_actors:
        try:
            loc = actor.get_location()
        except:
            continue
        should_destroy = False
        should_destroy = False
        if not is_point_in_box(loc):
            # FIX: Se il veicolo è fuori ma sta ENTRANDO (o è in coda per entrare), non distruggerlo.
            # Calcolo versore veicolo
            fwd = actor.get_transform().get_forward_vector()
            # Calcolo vettore dal veicolo al centro del box (0,0) - approssimato
            # Il centro del box è circa ( (MIN_X+MAX_X)/2, (MIN_Y+MAX_Y)/2 )
            center_x = (config.MIN_X + config.MAX_X) / 2
            center_y = (config.MIN_Y + config.MAX_Y) / 2
            vec_to_center = carla.Vector3D(center_x - loc.x, center_y - loc.y, 0)
            
            # Normalizzazione
            dist = math.sqrt(vec_to_center.x**2 + vec_to_center.y**2)
            if dist > 0:
                vec_to_center.x /= dist
                vec_to_center.y /= dist
            
            # Dot Product
            dot = fwd.x * vec_to_center.x + fwd.y * vec_to_center.y
            
            # Se dot > 0 guarda verso il centro (entrata) -> KEEP
            # Se dot < 0 guarda fuori (uscita) -> DESTROY
            if dot < -0.2: # Tolleranza
                 should_destroy = True
            else:
                 should_destroy = False # Salvato!

        elif loc.z < -10.0 or loc.z > 50.0:
            should_destroy = True
        if should_destroy:
            actor.set_autopilot(False, tm_port)
            batch_cmds.append(carla.command.DestroyActor(actor))
            batch_ids.append(actor.id)
            count += 1
    if batch_cmds:
        client.apply_batch(batch_cmds)
    return batch_ids
