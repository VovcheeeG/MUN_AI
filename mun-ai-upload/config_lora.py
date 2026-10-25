# config_lora.py
# ============================================================
#  Настройки для локального запуска Муна
# ============================================================

# ---------- ПУТИ (ЛОКАЛЬНЫЕ) ----------
BASE_MODEL_PATH = './qwen_parasite'
OUTPUT_DIR      = './lora_mun'
DATASET_PATH    = './dataset/mun_identity_final.jsonl'

# ---------- LoRA ----------
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj',
]

# ---------- ОБУЧЕНИЕ (не используется при инференсе) ----------
MAX_SEQ_LEN    = 1024
BATCH_SIZE     = 1
GRAD_ACCUM     = 8
LEARNING_RATE  = 2e-4
NUM_EPOCHS     = 3
WARMUP_STEPS   = 15
LR_SCHEDULER   = 'cosine'
OPTIMIZER      = 'paged_adamw_8bit'
LOGGING_STEPS  = 10
SAVE_STEPS     = 100
SAVE_TOTAL_LIMIT = 3

# ---------- КВАНТИЗАЦИЯ (для инференса на 16 ГБ RAM) ----------
LOAD_IN_4BIT          = True
BNB_4BIT_QUANT_TYPE   = 'nf4'
BNB_4BIT_DOUBLE_QUANT = True
BNB_4BIT_COMPUTE_DTYPE = 'bfloat16'

# ---------- СИСТЕМА ----------
SEED = 42