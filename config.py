# ==============================
# PolicyPal Configuration File
# (Gemini version, drop-in replacement)
# ==============================

# ========= Step 3: Parsing & Chunking =========


INPUT_PDF_DIR = "data/sample_policies"


OUTPUT_CHUNKS_PATH = "storage/parsed_chunks.json"

# Token-based chunking
TOKEN_CHUNK_SIZE = 800        
TOKEN_CHUNK_OVERLAP = 100
TOKEN_ENCODING_NAME = "cl100k_base"

# 
MIN_CHUNK_CHARS = 300


# ===== Step 4: Embeddings (Gemini) =====
# Gemini text embedding model
# NOTE:
# - google-genai SDK expects Gemini model IDs (e.g., "gemini-embedding-001")
# - Ensure your environment has GEMINI_API_KEY or GOOGLE_API_KEY
EMBEDDING_MODEL = "gemini-embedding-001"

# Retrieval test
RETRIEVAL_TOP_K = 3


# ===== Step 5: RAG Answer Generation (Gemini) =====
# Use a fast/cheap Gemini model for chat completion
# You can swap to "gemini-2.5-flash" if you want stronger quality (usually higher cost/latency).
CHAT_MODEL = "gemini-2.5-flash"

MAX_CONTEXT_CHARS = 12000   
RAG_TOP_K = 3
RAG_DISTANCE_THRESHOLD = 1.2  


# ===== Step 6: Intent Classification (Gemini) =====
# Intent classification can use the same model as chat
INTENT_MODEL = "gemini-2.5-flash-lite"
ENABLE_INTENT_ROUTER = True
