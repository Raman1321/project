from setuptools import setup, find_packages

setup(
    name="ml_fairness_monitoring",
    version="0.1",
    packages=find_packages(),
)
from models.baseline_model import train_baseline
from fairness.metrics import compute_fairness
