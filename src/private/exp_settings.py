from pathlib import Path

# ! Servers
LINUX_HOME = str(Path.home())
CONDA_ENV_NAME = 'ct'
CONDA_PATH = f'{LINUX_HOME}/miniconda/envs/{CONDA_ENV_NAME}'
NV_HTOP_FILE = f"{CONDA_PATH}/bin/nvidia-htop.py"
SV_INIT_CMDS = [
    f'source {LINUX_HOME}/miniconda/etc/profile.d/conda.sh;conda activate {CONDA_ENV_NAME}',
    f'alias tr_lm="python src/models/LMs/trainLM.py"',
]

# ! Git settings
GIT_ACCOUNT = 'AndyJZhao'
GIT_TOKEN = 'GHSAT0AAAAAABRRWTPVFBISNLHINRXUMIWKYSKKV3Q'

# ! Wandb settings
# Reuse the machine's existing `wandb login` credentials instead of hard-coding a
# (stale) key. WANDB_ENTITY=None -> logs under the logged-in account's default
# entity. conf_utils.wandb_init sets os.environ['WANDB_API_KEY'] = WANDB_API_KEY,
# so we read the key from ~/.netrc (written by `wandb login`), falling back to the
# WANDB_API_KEY env var.
import os as _os
import netrc as _netrc

WANDB_DIR = 'wandb'
WANDB_PROJ = 'GLEM-TAG'
WANDB_ENTITY = None
try:
    WANDB_API_KEY = _netrc.netrc().authenticators('api.wandb.ai')[2]
except Exception:
    WANDB_API_KEY = _os.environ.get('WANDB_API_KEY', '')
