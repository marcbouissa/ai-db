"""Tunables of the scene plugin (kept out of ai_db.constants: core knows nothing of 3D)."""

DEFAULT_JUDGE_MODEL = "claude-opus-5-5"
SCENE_MEMORY_PATH = "~/.ai-db/scene_memory.db"  # override: AI_DB_SCENE_MEMORY

# --- Scene layout solver (ai_db_scene/layout.py) ---
LAYOUT_SEED = 0
LAYOUT_EPS_M = 1e-4  # geometric slack for exact contact (walls, stacking)
LAYOUT_REL_TOL_M = 0.03  # tolerance when verifying gap/contact/centering relations
LAYOUT_FACING_TOL_DEG = 20.0
LAYOUT_CANDIDATES = 300  # sampled candidates per object (greedy pass)
LAYOUT_RETRY_CANDIDATES = 2000  # one denser retry before declaring a conflict
LAYOUT_LOCAL_ITERS = 400  # local-search proposals after the greedy pass
LAYOUT_LOCAL_STEP_M = 0.25
LAYOUT_LOCAL_YAW_DEG = 15.0
LAYOUT_DEFAULT_IN_FRONT_GAP_M = 0.4
LAYOUT_DEFAULT_BESIDE_GAP_M = 0.15
LAYOUT_DEFAULT_AROUND_GAP_M = 0.3
LAYOUT_DEFAULT_ABOVE_GAP_M = 0.25  # "above": B's top to A's base (mirror over a sink, board over a desk)
LAYOUT_FLAT_MAX_H_M = 0.03  # objects at most this tall (rugs) do not block others
LAYOUT_BREATHING_M = 1.0  # soft preference for free space around objects
LAYOUT_FACING_ATTEMPTS = 12  # extra seeds tried while a facing relation is still missed (≈50 ms each)
LAYOUT_FACING_MISS_COST = 2.0  # extra cost once facing is beyond LAYOUT_FACING_TOL_DEG (validate_scene blocks it)
LAYOUT_MAX_COUNT = 50  # max instances per ObjectSpec
LAYOUT_GRID_GAP_M = (0.5, 0.8)  # default [between columns, between rows] of a grid (aisles)
LAYOUT_MAX_BOUND_M = 500.0
LAYOUT_RESTARTS = 4  # greedy passes (fresh random draws) before reporting a conflict
LAYOUT_DOOR_HEIGHT_M = 2.1  # inner doors of multi-room buildings: the lintel starts here
LAYOUT_DOOR_CLEAR_M = 0.9  # free zone (door width x this deep) kept on both sides of each inner door

SCENE_SUN_ENERGY = 4.0  # W/m² of the daylight sun apply_layout adds (AgX view: a bright interior)
SCENE_EXPOSURE_INDOOR = 0.5  # stops for a reference room (SCENE_REF_FLOOR_M2): windows light it well
SCENE_EXPOSURE_OUTDOOR = -0.7  # open air gets the full sun and sky: +1 stop washed scenes out
SCENE_REF_FLOOR_M2 = 20.0  # floor area the indoor exposure is tuned for
SCENE_EXPOSURE_PER_DOUBLING = 0.35  # stops added each time the floor area doubles (deeper rooms)
SCENE_EXPOSURE_RANGE = (0.2, 1.5)  # indoor exposure clamp, stops
SCENE_FILL_W_PER_M2 = 8.0  # ceiling area fill: W per m² of floor (≈ an evenly lit room under AgX)
SCENE_FILL_W_RANGE = (100.0, 2500.0)
SCENE_FILL_SIZE_FRACTION = 0.5  # fill light size as a fraction of the room footprint (soft shadows)
SCENE_FILL_CEILING_GAP_M = 0.05  # below the ceiling plane, so it is inside the room
SCENE_LAMP_W = {"floor_lamp": 80.0, "table_lamp": 40.0, "pendant_lamp": 100.0, "ceiling_light": 150.0,
                "wall_sconce": 40.0, "street_lamp": 300.0, "lamp_post_double": 500.0}  # a point light in each lamp object
SCENE_CAMERA_INSET = 0.12  # interior camera: corner inset as a fraction of the room size
SCENE_CAMERA_HEIGHT_M = 1.6  # eye height indoors (outdoors: 2.5 m)
REPAIR_GROWTH = 1.2  # build_scene: room x/y growth per repair step
REPAIR_MAX_GROWTH = 1.8  # build_scene: never more than this times the requested size
BUILD_MAX_REPAIRS = 6  # build_scene: spec edits before giving up
BUILD_FIX_ROUNDS = 2  # build_scene: rounds of applying validate_scene's fix actions
BUILD_POLY_BUDGET: int | None = None  # build_scene: default total-face budget (None: no budget)
BUILD_DECIMATE_MIN_RATIO = 0.1  # never decimate an object below this fraction of its faces
BUILD_JUDGE_RESOLUTION = (640, 360)  # build_scene judge: the probe render (the build's eye-level camera)
BUILD_JUDGE_SAMPLES = 16
SCENE_EXPORT_DIR = "~/ai-db-exports"  # export_scene's folder; override: plugins.scene.export_dir, AI_DB_SCENE_EXPORT_DIR
SCENE_EXPORT_FORMATS = {"glb": ".glb", "gltf": ".gltf", "usd": ".usd", "usdz": ".usdz"}
SCENE_EXPORT_TIMEOUT_S = 300.0

# --- Blender (addon socket server) ---
BLENDER_OX_HOST = "127.0.0.1"
BLENDER_OX_PORT = 9877  # override: AI_DB_BLENDER_PORT
BLENDER_OX_TIMEOUT_S = 15.0

# --- Asset providers (ai_db_scene/providers.py; off unless configured) ---
ASSET_FILE_TYPES = (".glb", ".gltf", ".blend")  # local library files indexed by filename tokens
POLYHAVEN_API = "https://api.polyhaven.com"
POLYHAVEN_USER_AGENT = "ai-db-scene/0.1 (scene pipeline asset provider)"  # the API rejects requests without one
POLYHAVEN_RESOLUTIONS = ("1k", "2k", "4k")
POLYHAVEN_TIMEOUT_S = 30.0
POLYHAVEN_INDEX_TTL_S = 24 * 3600  # the model list is cached on disk this long
POLYHAVEN_CACHE_DIR = "ai-db/polyhaven"  # under $XDG_CACHE_HOME (~/.cache)
