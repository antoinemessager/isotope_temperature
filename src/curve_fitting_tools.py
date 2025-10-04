import numpy as np
import scipy.optimize
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import pandas as pd

from src.passey_model_core import run_passey_inverse_model

def sine_model(
    x: NDArray[np.float64],
    amplitude: float,
    vertical_shift: float,
    period: float,
    phase_shift: float
) -> NDArray[np.float64]:
    """
    Defines a sinusoidal model function.

    Note: The 'amplitude' here represents the peak-to-peak amplitude.

    Args:
        x: The independent variable (e.g., time or position).
        amplitude: The peak-to-peak amplitude of the sine wave (A).
        vertical_shift: The vertical offset of the sine wave (k).
        period: The period of the sine wave (T).
        phase_shift: The phase offset of the sine wave (phi).

    Returns:
        The calculated values of the sine wave at each point x.
    """
    return -amplitude / 2 * np.sin((2 * np.pi / period) * x - phase_shift) + vertical_shift

def sum_of_squared_errors(
    params: tuple[float, float, float, float],
    x_data: NDArray[np.float64],
    y_data: NDArray[np.float64]
) -> float:
    """
    Objective function to minimize. Calculates the sum of squared errors.

    This function is vectorized with NumPy for high efficiency, avoiding slow Python loops.
    
    Args:
        params: A tuple containing the model parameters (amplitude, vertical_shift, period, phase_shift).
        x_data: The x-coordinates of the data points.
        y_data: The y-coordinates of the data points.

    Returns:
        The total squared error between the model and the data.
    """
    amplitude, vertical_shift, period, phase_shift = params
    
    # Calculate the model's predictions for all x_data points at once
    y_pred = sine_model(x_data, amplitude, vertical_shift, period, phase_shift)
    
    # Calculate and return the sum of squared differences (residuals)
    return np.sum((y_pred - y_data)**2)

def fit_sine_curve(x: NDArray[np.float64], y: NDArray[np.float64]) -> dict[str, float]:
    """
    Fits a sinusoidal curve to the provided data points (x, y).

    This function estimates initial parameters, sets reasonable bounds, and then
    uses numerical optimization to find the best-fit sine curve.

    Args:
        x: The x-coordinates of the data points.
        y: The y-coordinates of the data points.

    Returns:
        A dictionary containing the optimized parameters of the sine model:
        {'amplitude', 'vertical_shift', 'period', 'phase_shift'}.
    """
    # --- 1. Estimate Initial Guess Parameters ---
    # These initial guesses are crucial for helping the optimizer converge to a good solution.
    initial_amplitude = max(y) - min(y)
    initial_vertical_shift = np.mean(y)
    # A simple guess for the period is the span of the x-data.
    # Note: This may be inaccurate if the data doesn't cover a full cycle.
    initial_period = max(x) - min(x)
    # Estimate phase by averaging the phases calculated from the min and max y points.
    phase_at_min = (np.pi * (2 / initial_period * x[np.argmin(y)] - 1/2))
    phase_at_max = (np.pi * (2 / initial_period * x[np.argmax(y)] - 3/2))
    initial_phase_shift = (phase_at_min % (2 * np.pi) + phase_at_max % (2 * np.pi)) / 2
    
    initial_params = np.array([initial_amplitude, initial_vertical_shift, initial_period, initial_phase_shift])

    # --- 2. Define Bounds for the Optimization ---
    # Bounding the parameters prevents the optimizer from finding unrealistic solutions.
    bounds = (
        (0.8 * initial_amplitude, 1.2 * initial_amplitude),  # Amplitude
        (initial_vertical_shift - 0.2 * abs(initial_vertical_shift), initial_vertical_shift + 0.2 * abs(initial_vertical_shift)), # Vertical shift
        (0.5 * initial_period, 1.5 * initial_period),      # Period
        (-np.inf, np.inf)                                    # Phase shift (unbounded)
    )

    # --- 3. Run the Optimization ---
    # We use scipy.optimize.minimize to find the parameters that minimize our error function.
    result = scipy.optimize.minimize(
        fun=sum_of_squared_errors,
        x0=initial_params,
        args=(x, y), # Additional arguments passed to the error function
        bounds=bounds
    )
    
    # Extract the optimized parameters from the result object
    opt_amplitude, opt_vertical_shift, opt_period, opt_phase_shift = result.x

    # --- 4. Return the Results ---
    fit_params = {
        "amplitude": opt_amplitude,
        "vertical_shift": opt_vertical_shift,
        "period": opt_period,
        "phase_shift": opt_phase_shift
    }

    return fit_params

