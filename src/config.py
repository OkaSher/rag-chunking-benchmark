"""Configuration and hyperparameter specification for the TechQA Chunking Benchmark.

Assignment 4 Experiment Protocol Freeze settings:
- Dataset: IBM TechQA (CDLA-Permissive 1.0)
- Baseline Strategy: Fixed-token chunking (window <= 256 MPNet tokens, 0 overlap)
- Proposed Strategy: Adjacent-sentence semantic chunking (128 target min, 320 hard max, 80th percentile distance)
- Embedding Model: sentence-transformers/all-mpnet-base-v2
- Generator Model: Qwen/Qwen2.5-1.5B-Instruct
- Retrieval: In-memory exact cosine similarity / FAISS FlatIP over 50 candidate docs
- Context Assembly: Top-5 chunks, <= 1280 tokens (Qwen tokenizer)
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "extracted"
OUTPUTS_DIR = BASE_DIR / "outputs" / "assignment_4"

TRAINING_QA_PATH = RAW_DATA_DIR / "training_Q_A.json"
DEV_QA_PATH = RAW_DATA_DIR / "dev_Q_A.json"
TECHNOTES_PATH = RAW_DATA_DIR / "training_dev_technotes.json"
SMOKE_FIXTURE_PATH = DATA_DIR / "smoke_fixture.json"

# Models and pinned revisions
ENCODER_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
ENCODER_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"

GENERATOR_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GENERATOR_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

# Chunking specifications
FIXED_CHUNK_SIZE = 256         # max tokens per chunk
FIXED_CHUNK_OVERLAP = 0        # zero overlap

SEMANTIC_TARGET_MIN = 128      # minimum tokens before allowing semantic boundary split
SEMANTIC_HARD_MAX = 320        # hard cap forcing chunk split
SEMANTIC_DISTANCE_PERCENTILE = 80.0  # boundary threshold percentile (1 - cosine)

# Retrieval and Budget
CANDIDATES_PER_QUERY = 50      # IBM TechQA supplied candidate documents pool
TOP_K_CHUNKS = 5               # Top-k retrieved chunks
CONTEXT_TOKEN_BUDGET = 1280    # Total budget in Qwen tokenizer tokens for delivered context

# Generator inference parameters
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0              # Greedy decoding for exact reproducibility
DO_SAMPLE = False

# Statistical testing parameters
RANDOM_SEED = 42
BOOTSTRAP_ROUNDS = 10000
ALPHA = 0.05
