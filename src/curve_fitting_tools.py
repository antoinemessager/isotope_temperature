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

def plot_seasonal_model(
    x_distance: NDArray[np.float64],
    measurements: NDArray[np.float64],
    estimated_signal: NDArray[np.float64],
    estimated_std: NDArray[np.float64],
    title_label: str
) -> None:
    """
    Fits a sinusoidal model and plots the results on a seasonal axis.
    This corrected version matches the logic of the working example.
    """
    # --- 1. Fit the Sinusoidal Model ---
    fit_params = fit_sine_curve(x_distance, estimated_signal)
    period_in_distance_units = fit_params["period"]
    
    # --- 2. Transform the X-axis to Months (following the example's logic) ---
    # Convert the distance into "raw months" where one period = 12 months.
    x_months = x_distance * 12 / period_in_distance_units
    
    # Generate the fitted curve on the original distance axis to find its minimum.
    fitted_sine_on_dist_axis = sine_model(x_distance, **fit_params)
    
    # Find the position of the minimum (coldest point) in raw months.
    coldest_point_in_months = x_months[np.argmin(fitted_sine_on_dist_axis)]
    
    # Shift the axis so this cold point corresponds to month 13 (January of the 2nd year).
    shift = 13 - coldest_point_in_months
    x_months_shifted = x_months + shift

    # --- 3. Create the Plot ---
    plt.figure(figsize=(12, 6))
    
    # Plot the original measurements on the new month axis.
    plt.plot(x_months_shifted, measurements, 'o', label='Original Observations')
    
    # Plot the inverted signal and its uncertainty band.
    line_plot = plt.plot(x_months_shifted, estimated_signal, '--', label='Inverted Signal')
    plt.fill_between(
        x_months_shifted,
        estimated_signal - estimated_std,
        estimated_signal + estimated_std,
        color=line_plot[0].get_color(),
        alpha=0.2
    )
    
    # Plot an IDEALIZED sinusoidal curve, as in the working example.
    # This curve uses the amplitude and vertical shift from the fit but has a
    # perfect 12-month period and a phase calculated to place its minimum at x=13.
    x_sine_plot = np.linspace(x_months_shifted.min(), x_months_shifted.max(), 200)
    
    # Calculate the phase `phi` so that the minimum of `sine_model` is at x=13 when T=12.
    # For the model y = -A/2 * sin(2*pi/T * x - phi) + k, the minimum is when sin(...) = 1.
    # (2*pi/12)*13 - phi = pi/2  => phi = 5*pi/3
    ideal_phase = 5 * np.pi / 3
    
    y_sine_plot = sine_model(
        x_sine_plot, 
        amplitude=fit_params["amplitude"], 
        vertical_shift=fit_params["vertical_shift"], 
        period=12,  # A perfect 12-month period
        phase_shift=ideal_phase
    )
    plt.plot(x_sine_plot, y_sine_plot, label='Idealized Seasonal Model', color='k', linewidth=2)

    # --- 4. Format the Axes ---
    min_month_val = int(np.floor(x_months_shifted.min()))
    max_month_val = int(np.ceil(x_months_shifted.max()))
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] * 3
    tick_labels = month_names[min_month_val - 1 : max_month_val]
    
    plt.xticks(ticks=np.arange(min_month_val, max_month_val + 1), labels=tick_labels, rotation=45)
    plt.ylabel('Temperature (°C)')
    plt.title(f'Seasonal Model for {title_label}')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

def analyze_and_save_horse_plot(
    full_df: pd.DataFrame, 
    horse_name: str, 
    output_dir: str
) -> None:
    """
    Runs the full analysis and plotting pipeline for a single horse and saves the plot.

    This function performs all steps in one place:
    1. Prepares the data for a specific horse.
    2. Runs the Passey inverse model.
    3. Creates a seasonal plot of the results.
    4. Saves the plot to a PDF file.

    Args:
        full_df: The DataFrame containing data for all horses.
        horse_name: The name of the horse to process.
        output_dir: The directory where the output PDF will be saved.
    """
    try:
        # --- 1. Prepare Data ---
        horse_df = (
            full_df.loc[full_df['horse'] == horse_name]
            .sort_values(by='index_tooth')
            .copy()  # Use .copy() to avoid SettingWithCopyWarning
        )
        
        # Create necessary columns
        horse_df['depth'] = 10.5
        horse_df['length'] = horse_df['hauteur_sample'].diff(-1) * 10
        horse_df = horse_df.dropna()

        # Check for sufficient data points
        if len(horse_df) < 5:
            print(f"Skipping {horse_name}: not enough data points after preparation.")
            return

        # Extract data as NumPy arrays
        measurements = horse_df['temperature'].to_numpy(dtype=float)
        segment_lengths = horse_df['length'].to_numpy(dtype=int)
        sampling_depths = horse_df['depth'].to_numpy(dtype=int)

        # --- 2. Run Inverse Model ---
        # This calls the high-level function defined previously
        mEst_mat = run_passey_inverse_model(
            measurements=measurements,
            segment_lengths=segment_lengths,
            sampling_depths=sampling_depths,
            verbose=False  # Disable verbose output to keep the loop clean
        )
        
        # --- 3. Create Seasonal Plot ---
        # Prepare data for the plotting function
        x_distance = np.cumsum(segment_lengths) / 10
        estimated_signal = np.mean(mEst_mat, axis=0)
        estimated_std = np.std(mEst_mat, axis=0)

        # Use the seasonal plotting function we created
        plot_seasonal_model(
            x_distance=x_distance,
            measurements=measurements,
            estimated_signal=estimated_signal,
            estimated_std=estimated_std,
            title_label=horse_name
        )

        # --- 4. Save the Figure ---
        safe_horse_name = horse_name.replace('/', '-')
        save_path = f"{output_dir}/Temperature_horse_{safe_horse_name}.pdf"
        plt.savefig(save_path)
        plt.close() # Crucial: closes the figure to free up memory

    except Exception as e:
        print(f"Failed to process horse {horse_name}. Error: {e}")