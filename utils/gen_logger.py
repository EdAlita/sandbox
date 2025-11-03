"""
gen_logger.py
"""
import logging
from pathlib import Path
from tqdm import tqdm
import os

def loggen(name: str):
        cwd = Path(os.getcwd())
        Path(cwd / "logs").mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(name)
        fhandler = logging.FileHandler(filename=Path( cwd / "logs" / f'{name}.log'), mode='a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        handler = logging.StreamHandler()
        handler.setStream(tqdm)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger