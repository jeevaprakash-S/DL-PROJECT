"""Configuration for World Model IDS Version 2.

The defaults are deliberately small enough for a Mac CPU sanity run.  Change
FULL_TRAIN_STEPS / EPOCHS only after inspecting label_audit.json.
"""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"

STAGES = ("Benign", "Reconnaissance", "Delivery", "Exploitation", "C2", "ActionsOnObjectives")
STAGE_TO_IDX = {name: i for i, name in enumerate(STAGES)}
NUM_STAGES = len(STAGES)
# All six stages are learned.  The primary reported metric excludes stages
# that the strict chronological test stream cannot support sufficiently.
PRIMARY_EVALUATION_STAGES = ("Benign", "Reconnaissance", "C2", "ActionsOnObjectives")
PRIMARY_EVALUATION_INDICES = tuple(STAGE_TO_IDX[name] for name in PRIMARY_EVALUATION_STAGES)
MIN_RELIABLE_TEST_SUPPORT = 100

# Rules are reviewed in order after labels are normalized.  This is a
# project-defined kill-chain taxonomy, not a native CICIDS label column.
LABEL_TO_STAGE_RULES = (
    (("benign", "normal"), "Benign"),
    (("portscan", "port scan", "scan", "probe"), "Reconnaissance"),
    (("ftp-patator", "ssh-patator", "brute force", "bruteforce", "infiltration"), "Delivery"),
    (("web attack", "sql injection", "sql", "xss", "heartbleed", "eternalblue", "exploit"), "Exploitation"),
    (("botnet", "bot", "backdoor", "command and control", "c2"), "C2"),
    (("ddos", "dos", "goldeneye", "slowloris", "slowhttptest", "hulk", "exfiltration"), "ActionsOnObjectives"),
)

LABEL_COLUMN_CANDIDATES = ("Label", "label", "Class", "class", "Attack", "attack")
TIMESTAMP_CANDIDATES = ("Timestamp", "timestamp", "Flow Start Time", "Flow Start Time (UTC)")
# Per-destination-IP grouping is supported, but CICIDS rare web-exploit flows
# can occur only at the tail of individual host streams.  Use source-file
# streams for the default reproducible six-stage evaluation; switch this to
# "Destination IP" only after label_audit.json confirms every stage has
# non-zero validation and test support.
GROUP_BY_COLUMN = None
GROUP_COLUMN_CANDIDATES = ("Destination IP", "Dst IP", "DestinationIP", "dst_ip")

WINDOW_SIZE = 12
FORECAST_HORIZON = 1
STRIDE = 1
TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION = 0.70, 0.15, 0.15
MIN_GROUP_ROWS = WINDOW_SIZE + FORECAST_HORIZON + 3
MIN_NUMERIC_FRACTION = 0.80

# Smaller than V1: practical default for CPU / MPS.  The architecture remains
# an encoder-only Transformer with joint forecasting and stage heads.
D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT = 64, 4, 2, 128, 0.15
BATCH_SIZE = 128
EPOCHS = 12
SANITY_TRAIN_STEPS = 300
FULL_TRAIN_STEPS = 2500  # sampled, class-aware updates per epoch; None = one sampled epoch
LEARNING_RATE, WEIGHT_DECAY = 8e-4, 1e-4
FORECAST_LOSS_WEIGHT, CLASS_LOSS_WEIGHT = 0.35, 1.0
FOCAL_GAMMA = 1.5
EFFECTIVE_NUMBER_BETA = 0.9999
MAX_ALPHA = 8.0
EARLY_STOPPING_PATIENCE = 4
SEED = 42
ALERT_STAGE_PROB_THRESHOLD = 0.60
ALERT_FORECAST_ERROR_PERCENTILE = 95

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def ensure_dirs() -> None:
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
