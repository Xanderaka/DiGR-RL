"""Recurrence Plot Transform.

References:
    J.-P Eckmann, S. Oliffson Kamphorst and D Ruelle, “Recurrence Plots of
        Dynamical Systems”. Europhysics Letters (1987).

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import numpy as np
from pyts.image.recurrence import RecurrencePlot
from scipy.signal import resample

from src.signal_to_image.base import SignalToImageTransformer


class RP(SignalToImageTransformer):
    def __init__(self,
                 dimension: int | float = 1,
                 time_delay: int | float = 50,
                 threshold: any = 'point',
                 percentage: int | float = 10,
                 flatten: bool = False):
        super().__init__()
        self.rp = RecurrencePlot(dimension, time_delay, threshold, percentage,
                                 flatten)

    def __str__(self):
        return "RecurrencePlotTransform()"

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # Input shape: (time, channels, antennas)

        # Step 1: Flatten to shape (time, channels * antennas)
        x = x.reshape((x.shape[0], -1))  # shape = (2000, 540)

        # Step 2: Average over channels (optional, helps reduce size & noise)
        x = x.mean(axis=1)  # shape = (2000,)

        # Step 3: Downsample (critical for size/memory)
        x = resample(x, 128)  # shape = (500,)

        # Step 4: Reshape to expected input (1, time)
        x = x[np.newaxis, :]  # shape = (1, 500)

        x = x.astype(np.float32)

        # Step 5: Generate RP
        rp_image = self.rp.transform(x)  # shape = (1, 500, 500)

        return rp_image  # or rp_image[0] if you want just the 2D array