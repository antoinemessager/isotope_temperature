import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from numpy.typing import NDArray


def calculate_mean_monte_carlo_error(
    measurements: NDArray[np.float64],
    segment_lengths: NDArray[np.float64],
    num_trials: int,
    analysis_error_std: float,
    length_error_std: float,
    depth_error_std: float,
    depth_shift: int
) -> np.float64:
    """
    Performs a Monte Carlo simulation to estimate the mean total squared error.

    This function simulates multiple error sources (analysis, length, depth)
    on a series of measurements to evaluate their combined impact.

    Args:
        measurements: 1D array of the initial measurement values.
        segment_lengths: 1D array of the lengths between each measurement.
        num_trials: Number of simulation iterations (trials).
        analysis_error_std: Standard deviation of the analysis error.
        length_error_std: Standard deviation of the length-related error.
        depth_error_std: Standard deviation of the depth-related error.
        depth_shift: Shift amount used for the depth error calculation.

    Returns:
        The mean total squared error over all trials.
    """
    num_samples = len(measurements)
    total_squared_error_per_trial = np.zeros(num_trials)

    # Interpolation can be pre-calculated once as it doesn't depend on the random trials.
    # 1. Create cumulative coordinates for interpolation
    cumulative_length = np.cumsum(segment_lengths)
    # Insert the starting point (0, 0) for correct interpolation
    x_coords = np.insert(cumulative_length, 0, 0)
    y_coords = np.insert(measurements, 0, 0)

    # 2. Create the interpolation object
    spline_interpolator = CubicSpline(x_coords, y_coords)

    # 3. Calculate the interpolated measurement difference (Delta-delta)
    # Create a fine-grained grid for interpolation
    interp_x_coords = np.arange(np.max(x_coords) + 1)
    interp_measurements = spline_interpolator(interp_x_coords)
    
    # Shift the interpolated data to compute the difference
    padded_interp = np.pad(interp_measurements, (depth_shift, 0), mode='edge')
    shifted_interp = padded_interp[:-depth_shift]
    
    # Dd is the difference across the entire interpolated grid
    Dd = shifted_interp - interp_measurements
    # Sample this difference at the original measurement points
    delta_delta_at_samples = Dd[x_coords[1:].astype(int)] # We exclude the 0-th point

    for j in range(num_trials):
        # --- 1. Calculate LengthError ---
        
        # Vectorized calculation of the "local slope" (Isoslope)
        # Calculate deltas and sums over adjacent segments
        delta_meas = np.abs(np.diff(measurements))
        sum_len = segment_lengths[:-1] + segment_lengths[1:]
        
        # Slope contribution from the right and left of each point
        slope_contrib = delta_meas / sum_len
        
        local_slope = np.zeros(num_samples)
        local_slope[1:-1] = slope_contrib[:-1] + slope_contrib[1:]
        local_slope[0] = 2 * slope_contrib[0]
        local_slope[-1] = 2 * slope_contrib[-1]
        
        # Generate the length error
        length_random_component = length_error_std * np.random.randn(num_samples)
        length_error = local_slope * length_random_component
        measurements_with_length_error = measurements + length_error

        # Vectorized clamping of the values
        # Ensure the noisy value doesn't exceed the min/max of its direct neighbors
        padded_meas = np.pad(measurements, 1, mode='edge')
        lower_bounds = np.minimum(padded_meas[:-2], padded_meas[2:])
        upper_bounds = np.maximum(padded_meas[:-2], padded_meas[2:])
        
        clamped_measurements = np.clip(measurements_with_length_error, lower_bounds, upper_bounds)
        
        # Final length error after clamping
        final_length_error = clamped_measurements - measurements

        # --- 2. Calculate DepthError ---
        depth_random_component = (depth_error_std * np.random.randn(num_samples)) / depth_shift
        depth_error = delta_delta_at_samples * depth_random_component
        
        # --- 3. Calculate AnalysisError ---
        analysis_error = analysis_error_std * np.random.randn(num_samples)
        
        # --- 4. Combine errors ---
        # Direct sum of squared errors is more efficient than taking the square root (sqrt) 
        # and then squaring it again.
        total_squared_error = np.sum(final_length_error**2 + depth_error**2 + analysis_error**2)
        
        total_squared_error_per_trial[j] = total_squared_error

    return np.mean(total_squared_error_per_trial)

def find_optimal_regularization_factor(
    num_trials: int,
    measurements: NDArray[np.float64],
    segment_lengths: NDArray[np.int64],
    sampling_depths: NDArray[np.int64],
    open_ended_index: int,
    analysis_error_std: float,
    length_error_std: float,
    depth_error_std: float,
    lag_depth: int,
    mixing_length: int,
    initial_fraction: float,
    min_ratio: float,
    max_ratio: float,
    ratio_stdev: float,
    min_depth: int,
    min_length: int,
    max_length: int,
    mean_estimated_error: float,
    plot_results: bool = False,
    verbose: bool = True
) -> float:
    """
    Finds the optimal regularization factor (df) by simulating a forward model.

    This function models a physical process where material is sampled over a profile,
    mixed, and measured. It introduces noise and uses regularization to find the
    best 'df' that explains the data without overfitting.

    Args:
        num_trials: Number of Monte Carlo simulations to run.
        measurements: Measured values at each sample point.
        segment_lengths: The length of each segment in the profile.
        sampling_depths: The depth of sampling at each point.
        open_ended_index: Index marking the start of the open-ended section.
        analysis_error_std: Std dev for the analytical measurement error.
        length_error_std: Std dev for noise added to segment lengths.
        depth_error_std: Std dev for noise added to sampling depths.
        lag_depth: Lag depth for sampling before the profile starts.
        mixing_length: The length over which material is mixed.
        initial_fraction: The fraction of initial material in the mix.
        min_ratio, max_ratio: Bounds for the underlying true ratio.
        ratio_stdev: Std dev for the random walk of the true ratio.
        min_depth, min_length, max_length: Physical constraints for the simulation.
        mean_estimated_error: The baseline mean error from a previous estimation.
        plot_results: If True, plots the DPE error vs. regularization factor.
        verbose: If True, shows a progress bar.

    Returns:
        The optimal regularization factor (df).
    """
    num_samples = len(measurements)
    df_candidates = np.logspace(-4, 1, 50)
    dpe_errors = np.zeros((num_trials, len(df_candidates)))

    # Main Monte Carlo simulation loop
    for i in tqdm(range(num_trials), disable=not verbose, desc="Running simulations"):
        # --- 1. Simulate noisy physical parameters ---
        avg_length = np.mean(segment_lengths)
        
        # Add noise to segment lengths and clip to physical bounds
        length_noise = length_error_std * np.random.randn(num_samples)
        noisy_lengths = np.round(length_noise).astype(int) + segment_lengths
        noisy_lengths = np.clip(noisy_lengths, min_length, max_length)

        # Add noise to sampling depths and clip to bounds
        depth_noise = np.round(depth_error_std * np.random.randn(num_samples))
        noisy_depths = (depth_noise + sampling_depths).astype(int)
        noisy_depths = np.clip(noisy_depths, min_depth, lag_depth)

        # --- 2. Construct the forward model matrix (A) ---
        
        # Define segments before and after the main profile for edge effects
        num_before = int(np.ceil(lag_depth / avg_length))
        num_after = int(np.ceil((mixing_length - open_ended_index) / avg_length)) + 1
        
        lengths_before = np.full(num_before, avg_length)
        lengths_after = np.full(num_after, avg_length)
        
        # Full profile lengths including extensions
        full_lengths = np.concatenate([lengths_before, noisy_lengths, lengths_after, [mixing_length]]).astype(int)
        total_profile_length = np.sum(full_lengths)
        num_total_segments = len(full_lengths) - 1

        # Build B, a matrix mapping each point in the profile to its segment
        B = np.zeros((total_profile_length, num_total_segments))
        current_pos = 0
        for m in range(num_total_segments):
            segment_len = full_lengths[m]
            B[current_pos : current_pos + segment_len, m] = 1
            current_pos += segment_len
        
        # Set non-contributing rows (beyond the open-ended index) to zero
        B[np.sum(full_lengths[:-1]) - open_ended_index:, :] = 0
        
        # Build AB, the mixing matrix
        final_fraction = 1 - initial_fraction
        AB = np.zeros((total_profile_length - mixing_length, num_total_segments))
        for o in range(total_profile_length - mixing_length):
            # Calculate the mixed contribution over the mixing_length window
            mixed_slice = B[o : o + mixing_length, :]
            weighted_avg = initial_fraction * B[o, :] + final_fraction * np.mean(mixed_slice, axis=0)
            
            # Normalize the contribution vector
            total_sum = np.sum(weighted_avg)
            if total_sum > 0:
                AB[o, :] = weighted_avg / total_sum
        
        # Build A, the final forward matrix that maps true ratios to measurements
        cumulative_lengths = np.cumsum(full_lengths).astype(int)
        A = np.zeros((num_samples, num_total_segments))
        
        for k in range(num_samples):
            start_segment_idx = num_before + k
            depth = noisy_depths[k]
            
            # Get the block of AB corresponding to the sampling depth and location
            start_row = cumulative_lengths[start_segment_idx - 1] - depth + 1
            end_row = cumulative_lengths[start_segment_idx] - depth
            sampling_block = AB[start_row:end_row, :]
            
            # The resulting measurement is the mean of this block
            A[k, :] = np.mean(sampling_block, axis=0)

        # --- 3. Generate a "true" underlying signal (mm) ---
        # This signal is a constrained random walk
        true_underlying_ratio = np.zeros(num_total_segments)
        true_underlying_ratio[0] = min_ratio
        for x in range(1, num_total_segments):
            next_val = true_underlying_ratio[x-1] + ratio_stdev * np.random.randn()
            # Re-sample if outside bounds
            while not (min_ratio <= next_val <= max_ratio):
                next_val = true_underlying_ratio[x-1] + ratio_stdev * np.random.randn()
            true_underlying_ratio[x] = next_val

        # --- 4. Invert the model for each regularization factor ---
        identity_matrix = np.eye(num_samples)
        noisy_measurements = measurements + analysis_error_std * np.random.randn(num_samples)
        
        # A_T * (A * A_T + df*I)^-1
        A_transpose = A.T
        term_to_invert_base = A @ A_transpose
        
        for df_idx, df in enumerate(df_candidates):
            term_to_invert = term_to_invert_base + df * identity_matrix
            inv_term = np.linalg.inv(term_to_invert)
            
            # m_est = m_true + A^T * (A*A^T + df*I)^-1 * (d_meas - A*m_true)
            estimated_ratio = true_underlying_ratio + A_transpose @ inv_term @ (noisy_measurements - A @ true_underlying_ratio)
            
            predicted_measurements = A @ estimated_ratio
            prediction_error = predicted_measurements - measurements
            dpe_errors[i, df_idx] = np.sum(prediction_error**2)

    # --- 5. Find the optimal df ---
    mean_dpe_errors = np.mean(dpe_errors, axis=0)
    
    if plot_results:
        plt.figure()
        plt.plot(df_candidates, mean_dpe_errors, label='Mean Prediction Error (DPE)')
        plt.axhline(mean_estimated_error, color='k', linestyle='--', label='Mean Estimated Error (Edist)')
        plt.xscale('log')
        plt.xlabel('Regularization Factor (df)')
        plt.ylabel('Sum of Squared Errors')
        plt.legend()
        plt.title('Model Error vs. Regularization')
        plt.show()

    # Find where the prediction error curve intersects the baseline estimated error
    spline = CubicSpline(df_candidates, mean_dpe_errors - mean_estimated_error)
    roots = spline.roots()
    
    # The optimal df is the largest positive root
    positive_roots = roots[roots > 0]
    if len(positive_roots) == 0:
        # Fallback if no intersection is found, return df with the minimum error
        return df_candidates[np.argmin(mean_dpe_errors)]
        
    optimal_df = np.max(positive_roots)

    return optimal_df

def _build_forward_model(
    noisy_lengths: NDArray[np.int64],
    noisy_depths: NDArray[np.int64],
    avg_length: float,
    lag_depth: int,
    mixing_length: int,
    open_ended_index: int,
    initial_fraction: float
) -> tuple[NDArray[np.float64], int, int]:
    """
    Constructs the forward model matrix (A) for the simulation.
    
    This is a helper function that encapsulates the complex matrix construction logic.
    """
    num_samples = len(noisy_lengths)
    
    # Define segments before and after the main profile for edge effects
    num_before = int(np.ceil(lag_depth / avg_length))
    num_after = int(np.ceil((mixing_length - open_ended_index) / avg_length)) + 1
    
    lengths_before = np.full(num_before, avg_length)
    lengths_after = np.full(num_after, avg_length)
    
    # Full profile lengths including extensions
    full_lengths = np.concatenate([lengths_before, noisy_lengths, lengths_after, [mixing_length]]).astype(int)
    total_profile_length = np.sum(full_lengths)
    num_total_segments = len(full_lengths) - 1

    # Build B, a matrix mapping each point in the profile to its segment
    B = np.zeros((total_profile_length, num_total_segments))
    current_pos = 0
    for m in range(num_total_segments):
        segment_len = full_lengths[m]
        B[current_pos : current_pos + segment_len, m] = 1
        current_pos += segment_len
    
    # Set non-contributing rows (beyond the open-ended index) to zero
    B[np.sum(full_lengths[:-1]) - open_ended_index:, :] = 0
    
    # Build AB, the mixing matrix
    final_fraction = 1 - initial_fraction
    AB = np.zeros((total_profile_length - mixing_length, num_total_segments))
    for o in range(total_profile_length - mixing_length):
        mixed_slice = B[o : o + mixing_length, :]
        weighted_avg = initial_fraction * B[o, :] + final_fraction * np.mean(mixed_slice, axis=0)
        
        total_sum = np.sum(weighted_avg)
        if total_sum > 0:
            AB[o, :] = weighted_avg / total_sum
    
    # Build A, the final forward matrix that maps true ratios to measurements
    cumulative_lengths = np.cumsum(full_lengths).astype(int)
    A = np.zeros((num_samples, num_total_segments))
    
    for k in range(num_samples):
        start_segment_idx = num_before + k
        depth = noisy_depths[k]
        
        start_row = cumulative_lengths[start_segment_idx - 1] - depth + 1
        end_row = cumulative_lengths[start_segment_idx] - depth
        sampling_block = AB[start_row:end_row, :]
        
        A[k, :] = np.mean(sampling_block, axis=0)
        
    return A, num_total_segments, num_before

def run_inverse_prediction(
    num_trials: int,
    measurements: NDArray[np.float64],
    segment_lengths: NDArray[np.int64],
    sampling_depths: NDArray[np.int64],
    open_ended_index: int,
    analysis_error_std: float,
    length_error_std: float,
    depth_error_std: float,
    lag_depth: int,
    mixing_length: int,
    initial_fraction: float,
    min_ratio: float,
    max_ratio: float,
    ratio_stdev: float,
    min_depth: int,
    min_length: int,
    max_length: int,
    regularization_factor: float,
    verbose: bool = True
) -> NDArray[np.float64]:
    """
    Performs an inverse prediction for an underlying ratio from noisy measurements.

    This function runs a Monte Carlo simulation. In each trial, it:
    1. Simulates noisy physical parameters (length, depth).
    2. Constructs a forward model mapping a true underlying ratio to expected measurements.
    3. Generates a synthetic "true" underlying ratio.
    4. Applies a regularized inverse formula to estimate the ratio from noisy measurements.
    
    Args:
        (See 'find_optimal_regularization_factor' for parameter descriptions)
        regularization_factor: The optimal 'df' value to use for the inversion.

    Returns:
        A matrix (num_trials x num_samples) containing the estimated underlying
        ratio for the core profile segment in each simulation trial.
    """
    num_samples = len(measurements)
    estimated_ratio_matrix = np.zeros((num_trials, num_samples))
    
    # Main Monte Carlo simulation loop
    for i in tqdm(range(num_trials), disable=not verbose, desc="Running predictions"):
        # --- 1. Simulate noisy physical parameters ---
        avg_length = np.mean(segment_lengths)
        
        length_noise = length_error_std * np.random.randn(num_samples)
        noisy_lengths = np.clip(np.round(length_noise).astype(int) + segment_lengths, min_length, max_length)

        depth_noise = np.round(depth_error_std * np.random.randn(num_samples))
        noisy_depths = np.clip((depth_noise + sampling_depths).astype(int), min_depth, lag_depth)

        # --- 2. Construct the forward model using the helper function ---
        A, num_total_segments, num_before = _build_forward_model(
            noisy_lengths, noisy_depths, avg_length, lag_depth,
            mixing_length, open_ended_index, initial_fraction
        )

        # --- 3. Generate a "true" underlying signal (mm) ---
        true_underlying_ratio = np.zeros(num_total_segments)
        true_underlying_ratio[0] = min_ratio
        for x in range(1, num_total_segments):
            next_val = true_underlying_ratio[x-1] + ratio_stdev * np.random.randn()
            while not (min_ratio <= next_val <= max_ratio):
                next_val = true_underlying_ratio[x-1] + ratio_stdev * np.random.randn()
            true_underlying_ratio[x] = next_val

        # --- 4. Apply the regularized inverse formula ---
        identity_matrix = np.eye(num_samples)
        noisy_measurements = measurements + analysis_error_std * np.random.randn(num_samples)
        
        # A_T * (A * A_T + df*I)^-1
        A_transpose = A.T
        term_to_invert = A @ A_transpose + regularization_factor * identity_matrix
        inv_term = np.linalg.inv(term_to_invert)
        
        # m_est = m_true + A^T * (A*A^T + df*I)^-1 * (d_meas - A*m_true)
        estimated_ratio = true_underlying_ratio + A_transpose @ inv_term @ (noisy_measurements - A @ true_underlying_ratio)
        
        # Store the part of the estimated ratio that corresponds to the actual samples
        estimated_ratio_matrix[i, :] = estimated_ratio[num_before : num_samples + num_before]

    return estimated_ratio_matrix

def run_passey_inverse_model(
    measurements: NDArray[np.float64],
    segment_lengths: NDArray[np.int64],
    sampling_depths: NDArray[np.int64],
    num_trials: int = 100,
    analysis_error_std: float = 0.3,
    length_error_std: float = 1.0,
    depth_error_std: float = 2.0,
    lag_depth: int = 15,
    mixing_length: int = 250,
    min_length: int = 3,
    max_length: int = 40,
    min_depth: int = 5,
    open_ended_index: int = 1,
    initial_fraction: float = 0.25,
    verbose: bool = True
) -> NDArray[np.float64]:
    """
    Applies the full Passey et al. (2005) inverse model to an input signal.

    This high-level function orchestrates a three-step Monte Carlo process:
    1.  Estimates the baseline squared error of the measurements (`mean_Edist`).
    2.  Finds the optimal regularization factor (`df`) that balances model fit
        and complexity.
    3.  Runs the final inverse prediction using the optimal `df` to generate a
        set of possible "true" underlying signals.

    Args:
        measurements: The list of observed values (e.g., temperature, isotopes).
        segment_lengths: The height or length of each measured segment.
        sampling_depths: The depth of each measurement.
        num_trials: The number of Monte Carlo simulations to run.
        (Other parameters are model-specific settings as described in Passey et al., 2005)

    Returns:
        A matrix of size (num_trials x num_samples), where each row is a
        possible "true" signal (e.g., paleo-temperature profile) inverted from
        the measurements.
    """
    # --- 1. Initial Parameter Setup ---
    # Derive key statistical parameters from the input measurements.
    min_ratio = np.quantile(measurements, 0.1)
    max_ratio = np.quantile(measurements, 0.9)
    # The standard deviation for the random walk of the true signal
    ratio_stdev = max_ratio - min_ratio

    # --- 2. Step 1: Compute Mean Estimated Error (Edist) ---
    if verbose:
        print("Step 1/3: Computing mean baseline error (Edist)...")
    
    mean_estimated_error = calculate_mean_monte_carlo_error(
        measurements=measurements,
        segment_lengths=segment_lengths,
        num_trials=num_trials,
        analysis_error_std=analysis_error_std,
        length_error_std=length_error_std,
        depth_error_std=depth_error_std,
        depth_shift=lag_depth, # 'la' corresponds to 'depth_shift' in this function
    )

    # --- 3. Step 2: Find the Optimal Regularization Factor (df) ---
    if verbose:
        print("Step 2/3: Finding optimal regularization factor (df)...")

    optimal_df = find_optimal_regularization_factor(
        num_trials=num_trials,
        measurements=measurements,
        segment_lengths=segment_lengths,
        sampling_depths=sampling_depths,
        open_ended_index=open_ended_index,
        analysis_error_std=analysis_error_std,
        length_error_std=length_error_std,
        depth_error_std=depth_error_std,
        lag_depth=lag_depth,
        mixing_length=mixing_length,
        initial_fraction=initial_fraction,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        ratio_stdev=ratio_stdev,
        min_depth=min_depth,
        min_length=min_length,
        max_length=max_length,
        mean_estimated_error=mean_estimated_error,
        plot_results=False, # Set to True to see the diagnostic plot
        verbose=verbose
    )

    # --- 4. Step 3: Run the Final Inverse Prediction ---
    if verbose:
        print("Step 3/3: Running final inverse prediction...")

    estimated_ratio_matrix = run_inverse_prediction(
        num_trials=num_trials,
        measurements=measurements,
        segment_lengths=segment_lengths,
        sampling_depths=sampling_depths,
        open_ended_index=open_ended_index,
        analysis_error_std=analysis_error_std,
        length_error_std=length_error_std,
        depth_error_std=depth_error_std,
        lag_depth=lag_depth,
        mixing_length=mixing_length,
        initial_fraction=initial_fraction,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        ratio_stdev=ratio_stdev,
        min_depth=min_depth,
        min_length=min_length,
        max_length=max_length,
        regularization_factor=optimal_df,
        verbose=verbose
    )

    if verbose:
        print("Inverse modeling complete.")
        
    return estimated_ratio_matrix

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