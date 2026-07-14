"""Environment-driven settings."""

import os
import shutil
from pathlib import Path

NVCC_PATH = os.getenv("NVCC_PATH", shutil.which("nvcc") or "/opt/cuda/bin/nvcc")
NVCC_ARCH = os.getenv("NVCC_ARCH", "native")
JUDGE_COMPILE_TIMEOUT_S = int(os.getenv("JUDGE_COMPILE_TIMEOUT_S", "120"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

SMTP_ADDRESS = os.getenv("SMTP_ADDRESS", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
MAILER_FROM_ADDRESS = os.getenv("MAILER_FROM_ADDRESS", "noreply@example.com")
