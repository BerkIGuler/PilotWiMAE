"""
PilotWiMAE: Temporal Wireless Masked Autoencoder Package

This package provides the PilotWiMAE model for 3D wireless channel modeling.
"""

__version__ = "0.1.0"
__author__ = "Berkay Guler"

from .models import PilotWiMAE
from .training import PilotWiMAETrainer

__all__ = [
    "PilotWiMAE",
    "PilotWiMAETrainer",
]
