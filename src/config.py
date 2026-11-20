"""Central configuration: paths, constants and tunable parameters.

Everything that an examiner might ask "why this value?" about lives here so it
is easy to find, change and justify.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
RES_DIR = OUT_DIR / "results"
CLEAN_CSV = DATA_DIR / "traffic_cleaned.csv"
RAW_CSV = DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"

RANDOM_STATE = 42

# ---- chronological split (no shuffling) -----------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15          # test = remaining 15 %
SPLIT_GAP_HOURS = 1      # rows dropped at each boundary (target of the last
                         # train row is the first val-row's current volume)

# ---- domain constants -------------------------------------------------------
DIRECTIONS = ["North", "South", "East", "West"]
QUEUE_COLS = {"North": "queue_north", "South": "queue_south",
              "East": "queue_east", "West": "queue_west"}
WEATHER_CATEGORIES = ["Clear", "Clouds", "Drizzle", "Fog", "Mist", "Rain",
                      "Snow", "Thunderstorm"]
HOLIDAY_LABEL = "National Holiday"

# Congestion bands. They were *recovered from the data* (congestion_level is a
# pure threshold function of traffic_volume: Low < 2500 <= Medium <= 4500 < High).
LOW_MAX = 2500
HIGH_MIN = 4500

# ---- model selection --------------------------------------------------------
# A simpler model is preferred when its validation RMSE is within this fraction
# of the best model's RMSE (stated *before* seeing results; see docs).
SELECTION_TOLERANCE = 0.02
SIMPLICITY_RANK = {"Linear Regression": 0, "Random Forest": 1,
                   "Gradient Boosting": 2}

# ---- signal control constraints (simulation assumptions) -------------------
SIGNAL = dict(
    cycle_s=120,            # reference / fixed-time cycle length
    cycle_low_s=60,         # adaptive cycle when predicted volume is Low   (chosen on VALIDATION days,
                            # see outputs/results/calibration_cycle_low_s.csv)
    cycle_high_s=120,       # adaptive cycle when predicted volume is High (longer cycles 135/150 were WORSE on
                            # validation days, see calibration_cycle_high_s.csv)
    clearance_s=4,          # yellow(3) + all-red(1) after every green
    adverse_extra_clearance_s=2,
    min_green_s=10,
    max_green_s=60,
    incident_max_green_s=45,
    emergency_green_s=30,
    ambulance_lead_s=15,    # detection happens this long before stop-line
    ambulance_yield_s=3,    # vehicles ahead pull aside; extra crossing delay
    startup_lost_s=2,       # first seconds of green with no discharge
    saturation_veh_per_s=1.0,   # 2 lanes x 1800 veh/h per approach
    adverse_flow_factor=0.85,   # discharge slows in bad weather
    incident_flow_factor=0.5,   # blocked lane halves one approach's discharge
    demand_scale=0.4,       # share of the dataset's hourly volume that reaches
                            # THIS intersection (site calibration constant)
)
