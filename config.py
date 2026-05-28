"""
Configuration for SmartScrape.
All secrets are read from environment variables (see .env.example).
NEVER hardcode tokens or passwords here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- FITLAYOUT CONNECTION ---
API_BASE_URL = os.getenv("FITLAYOUT_API_URL", "https://layout.fit.vutbr.cz/api")
TARGET_REPOSITORY_NAME = os.getenv("FITLAYOUT_REPO_NAME", "IE AI")

# --- SERVICE IDENTIFIERS ---
SERVICE_RENDER_ID = os.getenv("FITLAYOUT_RENDER_SERVICE", "FitLayout.Puppeteer")
SERVICE_SEGMENTATION_ID = os.getenv("FITLAYOUT_SEGMENTATION_SERVICE", "FitLayout.VIPS")

# Token is NEVER hardcoded — always read from environment
SERVICE_AUTH_TOKEN = os.getenv("FITLAYOUT_TOKEN", "")
if not SERVICE_AUTH_TOKEN:
    import warnings
    warnings.warn(
        "[SmartScrape] FITLAYOUT_TOKEN is not set. "
        "Create a .env file from .env.example and add your token.",
        stacklevel=1,
    )

# --- ML SETTINGS ---
EMBEDDING_DIM = 128
HIDDEN_DIM = 64

# --- CONSTRAINT SETTINGS (aligned with dissertation Section 6.3) ---
# Footer exclusion: nodes in the bottom 20% are excluded from Title/Price.
# Paper reference: Section 6.3, Constraint 2 (Geometry / Footer Trap).
FOOTER_THRESHOLD = 0.80

# Product-zone exclusion threshold for books.toscrape.com sample pages (paper Γ3).
PRODUCT_ZONE_MAX_PX = float(os.getenv("SMARTSCRAPE_PRODUCT_ZONE_MAX_PX", "500.0"))

# Stability score threshold for drift detection (Section 6.4)
STABILITY_THRESHOLD = 0.60

# GNN model checkpoint path
MODEL_CHECKPOINT = os.getenv("SMARTSCRAPE_MODEL_PATH", "model.pt")
