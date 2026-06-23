from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rbf_benchmark.config import read_config
from rbf_benchmark.training import run_dataset
run_dataset("iris", read_config())
