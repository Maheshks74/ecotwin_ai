"""EcoTwin: Industrial Digital Twin with RL Controller, Preference Learning & LLM Advisor."""
import os

# Prevent OpenMP runtime duplicate error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

__version__ = "0.1.0"

