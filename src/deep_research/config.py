from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "research.db"


class Settings(BaseModel):
    zai_api_key: str = os.getenv("ZAI_API_KEY", "")
    zai_base_url: str = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
    glm_model: str = os.getenv("GLM_MODEL", "glm-5")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    semantic_scholar_api_key: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    default_depth: str = os.getenv("RESEARCH_DEPTH", "standard")
    max_sub_questions: int = 5
    max_search_results: int = 5
    max_scrape_pages: int = 3

    # Multi-model routing: override per pipeline stage (empty = use glm_model)
    model_decompose: str = os.getenv("MODEL_DECOMPOSE", "")
    model_analyze: str = os.getenv("MODEL_ANALYZE", "")
    model_synthesize: str = os.getenv("MODEL_SYNTHESIZE", "")
    model_gap_analysis: str = os.getenv("MODEL_GAP_ANALYSIS", "")


settings = Settings()
