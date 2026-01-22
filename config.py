import carla

# ====================================================
# CONFIGURAZIONE GENERALE
# ====================================================
HOST = '127.0.0.1'
PORT = 2000
CLIENT_TIMEOUT = 60.0

# --- PARAMETRI SIMULAZIONE ---
MAX_CONCURRENT_VEHICLES = 50
MAX_TOTAL_VEHICLES = 10000
SPAWN_RETRY_TIME = 1.0
GLOBAL_CLEANUP_INTERVAL = 2.0

# --- CONFIGURAZIONE AI ---
ENABLE_AI_CONTROL = True
AI_STEP_INTERVAL = 0.5
EMERGENCY_DETECTION_DIST = 65.0

# --- CONFIGURAZIONE TRAINING OTTIMIZZATO ---
SYNCHRONOUS_MODE = True
FIXED_DELTA_SECONDS = 0.1
TARGET_FINISHED_CARS = 500
NO_RENDERING_MODE = False

# --- CONFIGURAZIONE SEMAFORI (FIX ALLINEAMENTO) ---
TRAFFIC_LIGHT_CORRECTION = -90

# --- FIX LOGICA SPECCHIO ---
INVERT_DIRECTION_LOGIC = True

# TEMPI SEMAFORO
MIN_GREEN_TIME = 7.0
YELLOW_TIME = 4.0
ALL_RED_TIME = 2

# --- AREA DI INTERESSE ---
MIN_X, MAX_X = -110.0, 90.0
MIN_Y, MAX_Y = -75.0, 75.0

BOUNDING_BOX_VERTICES = [
    (MAX_X, MAX_Y), (MAX_X, MIN_Y), (MIN_X, MIN_Y), (MIN_X, MAX_Y)
]

MIN_LIGHTS_PER_JUNCTION = 3

# --- SPAWN POINTS ---
FIXED_SPAWN_POINTS = [
    carla.Transform(carla.Location(x=-105, y=3, z=0.5), carla.Rotation(yaw=0)),
    carla.Transform(carla.Location(x=-105, y=6.5, z=0.5), carla.Rotation(yaw=0)),
    carla.Transform(carla.Location(x=85, y=-5, z=0.5), carla.Rotation(yaw=180)),
    carla.Transform(carla.Location(x=85, y=-1.5, z=0.5), carla.Rotation(yaw=180)),
    carla.Transform(carla.Location(x=-47.2, y=70, z=0.5), carla.Rotation(yaw=270)),
    carla.Transform(carla.Location(x=-43.7, y=70, z=0.5), carla.Rotation(yaw=270)),
    carla.Transform(carla.Location(x=35.3, y=70, z=0.5), carla.Rotation(yaw=270)),
    carla.Transform(carla.Location(x=31.8, y=70, z=0.5), carla.Rotation(yaw=270)),
    carla.Transform(carla.Location(x=-54.8, y=-70, z=0.5), carla.Rotation(yaw=90)),
    carla.Transform(carla.Location(x=-51.3, y=-70, z=0.5), carla.Rotation(yaw=90)),
    carla.Transform(carla.Location(x=29, y=-70, z=0.5), carla.Rotation(yaw=90)),
    carla.Transform(carla.Location(x=25.5, y=-70, z=0.5), carla.Rotation(yaw=90)),
]
