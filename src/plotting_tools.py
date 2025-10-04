import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np
import pandas as pd

from src.curve_fitting_tools import fit_sine_curve, sine_model
from src.passey_model_core import run_passey_inverse_model

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


def plot_inverse_model_results(
    segment_lengths: NDArray[np.float64],
    measurements: NDArray[np.float64],
    estimated_matrix: NDArray[np.float64],
    title_label: str,
    distance_scale_factor: float = 10.0
) -> None:
    """
    Creates a plot comparing original measurements with the inverse model results.

    This function displays:
    1. The original measurements as scatter points.
    2. The mean estimated signal from the model as a line.
    3. The standard deviation of the model's estimates as a shaded uncertainty band.

    Args:
        segment_lengths: The length of each segment in the profile.
        measurements: The original measured values (e.g., temperature).
        estimated_matrix: The output matrix from the inverse model (num_trials x num_samples).
        title_label: A label to include in the plot's title (e.g., the sample name).
        distance_scale_factor: The factor to divide the cumulative length by for the x-axis unit.
    """
    # --- 1. Prepare data for plotting ---
    # Calculate the x-axis (distance) by taking the cumulative sum of segment lengths.
    x_distance = np.cumsum(segment_lengths) / distance_scale_factor
    
    # Calculate the mean and standard deviation of the model's predictions across all trials.
    mean_signal = np.mean(estimated_matrix, axis=0)
    std_signal = np.std(estimated_matrix, axis=0)
    
    # --- 2. Create the plot ---
    # Plot the original measurements as points.
    plt.plot(x_distance, measurements, 'o', label='Observations')
    
    # Plot the mean estimated signal as a line. We capture the plot object to reuse its color.
    line_plot = plt.plot(x_distance, mean_signal, label='Estimated Signal Source')
    
    # Add a shaded region for the uncertainty (±1 standard deviation).
    plt.fill_between(
        x_distance,
        mean_signal - std_signal,
        mean_signal + std_signal,
        color=line_plot[0].get_color(), # Match the line color
        alpha=0.2 # Make the shaded region semi-transparent
    )
    
    # --- 3. Add labels and finalize the plot ---
    plt.legend()
    plt.title(f'Inverse Model Results for {title_label}')
    plt.ylabel('Temperature (°C)')
    plt.xlabel(f'Distance from Enamel Junction (mm)')
    
    return