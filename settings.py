import os

import dotenv

dotenv.load_dotenv()


CONFIG = "configs/self_forcing_server_14b.yaml"
HF_HOME = os.getenv("HF_HOME")
REL_MODEL_FOLDER = "models--Wan-AI--Wan2.1-T2V-14B/snapshots/a064a6c71f5be440641209c07bf2a5ce7a2ff5e4"
JOB_ID = os.getenv("SLURM_JOB_ID")
USER_ID = os.getenv("USER")
TMP_DIR = f"/scratch/tmp.{JOB_ID}.{USER_ID}"
MODEL_FOLDER = os.getenv("MODEL_FOLDER", f"{TMP_DIR}/huggingface/hub/{REL_MODEL_FOLDER}")
COMPILE_SHAPES = [(832, 480), (480, 832)]
