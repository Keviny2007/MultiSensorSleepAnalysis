import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, resample

# This code is based off the Axivity data processing pipeline described in the paper: https://findresearcher.sdu.dk/ws/files/145634666/Generating_ActiGraph_Counts_from_Raw_Acceleration_Recorded_by_an_Alternative_Monitor.pdf
# The purpose of this file is to generate activity counts from raw Axivity data. This is the expected data format for sleep analysis algorithms
# The pipeline consists of the following steps:

def butter_lowpass(cutoff, fs, order=4):  # Nyquist theorem states that a signal must be sampled at a frequency at least twice the highest frequency present in the signal to avoid aliasing
    nyquist = fs / 2                      # See paper for more information
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=4):  # Bandpass filter is used to remove noise from the signal per paper instructions
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def process_axivity_data(input_csv, output_csv):
    # Step 1: Load data
    print("Loading data...")
    data = pd.read_csv(input_csv)
    timestamps = data['dataTimestamp']
    x, y, z = data['axis1'], data['axis2'], data['axis3']

    # Step 2: Resample to 30 Hz if necessary
    original_rate = 100  # Adjust if your input sampling rate is different
    target_rate = 30
    resample_factor = target_rate / original_rate
    num_samples = int(len(x) * resample_factor)
    print(f"Resampling to {target_rate} Hz...")
    x_resampled = resample(x, num_samples)
    y_resampled = resample(y, num_samples)
    z_resampled = resample(z, num_samples)

    # Step 3: Apply aliasing low-pass filter (cutoff = 15 Hz)
    print("Applying low-pass filter...")
    b, a = butter_lowpass(cutoff=14.9, fs=target_rate) # Slightly lower than 15 Hz b/c dividing by 1 raises errors in lowpass filter
    x_filtered = filtfilt(b, a, x_resampled)
    y_filtered = filtfilt(b, a, y_resampled)
    z_filtered = filtfilt(b, a, z_resampled)

    # Step 4: Apply band-pass filter (0.29–1.63 Hz)
    print("Applying band-pass filter...")
    b, a = butter_bandpass(lowcut=0.29, highcut=1.63, fs=target_rate) # Constants for upper and lower bound specified in paper.
    x_bandpassed = filtfilt(b, a, x_filtered)
    y_bandpassed = filtfilt(b, a, y_filtered)
    z_bandpassed = filtfilt(b, a, z_filtered)

    # Step 5: Resample to 10 Hz
    print("Resampling to 10 Hz...")
    target_rate = 10
    resample_factor = target_rate / 30  # From 30 Hz to 10 Hz
    num_samples = int(len(x_bandpassed) * resample_factor)
    x_downsampled = resample(x_bandpassed, num_samples)
    y_downsampled = resample(y_bandpassed, num_samples)
    z_downsampled = resample(z_bandpassed, num_samples)

    # Step 6: Calculate vector magnitude
    print("Calculating vector magnitude...")
    vm = np.sqrt(x_downsampled**2 + y_downsampled**2 + z_downsampled**2)

    # Step 7: Apply dead-band threshold
    print("Applying dead-band threshold...")
    x_downsampled[x_downsampled < 0.068] = 0
    y_downsampled[y_downsampled < 0.068] = 0
    z_downsampled[z_downsampled < 0.068] = 0
    vm[vm < 0.068] = 0

    # Step 8: Cap at 2.13 g
    print("Capping at 2.13 g...")
    x_downsampled[x_downsampled > 2.13] = 2.13
    y_downsampled[y_downsampled > 2.13] = 2.13
    z_downsampled[z_downsampled > 2.13] = 2.13
    vm[vm > 2.13] = 2.13

    # Step 9: Convert to 8-bit resolution
    print("Converting to 8-bit resolution...")
    x_scaled = (x_downsampled / 2.13) * 128
    y_scaled = (y_downsampled / 2.13) * 128
    z_scaled = (z_downsampled / 2.13) * 128
    vm_scaled = (vm / 2.13) * 128

    x_scaled = np.round(x_scaled).astype(int)
    y_scaled = np.round(y_scaled).astype(int)
    z_scaled = np.round(z_scaled).astype(int)
    vm_scaled = np.round(vm_scaled).astype(int)

    # Step 10: Aggregate into 60-second epochs (sum of 600 samples): This is based off https://github.com/dipetkov/actigraph.sleepr. Must convert data to 60-second epochs.
    print("Aggregating into 60-second epochs...")       
    num_epochs = len(vm_scaled) // 600
    x_scaled = x_scaled[:num_epochs * 600].reshape(-1, 600)
    y_scaled = y_scaled[:num_epochs * 600].reshape(-1, 600)
    z_scaled = z_scaled[:num_epochs * 600].reshape(-1, 600)
    vm_scaled = vm_scaled[:num_epochs * 600].reshape(-1, 600)

    x_epoch_counts = np.sum(x_scaled, axis=1)
    y_epoch_counts = np.sum(y_scaled, axis=1)
    z_epoch_counts = np.sum(z_scaled, axis=1)
    vm_epoch_counts = np.sum(vm_scaled, axis=1)

    # Step 11: Match timestamps to 60-second epochs
    print("Matching timestamps to epochs...")
    initial_timestamp = pd.to_datetime(timestamps.iloc[0])  # Start time
    epoch_duration = pd.Timedelta(seconds=60)  # Each epoch is 60 seconds

    epoch_timestamps = [
        initial_timestamp + i * epoch_duration for i in range(num_epochs)
    ]
    # Step 12: Save results to CSV
    print("Saving results...")
    output_df = pd.DataFrame({
        'dataTimestamp': epoch_timestamps,
        'axis1': x_epoch_counts,
        'axis2': y_epoch_counts,
        'axis3': z_epoch_counts,
        'vm_epoch_counts': vm_epoch_counts
    })
    output_df.to_csv(output_csv, index=False)
    print(f"Processing complete. Output saved to {output_csv}")

# Example usage
input_csv = "big_file_raw.csv"  # Replace with your input file
output_csv = "activity_counts_with_axes.csv"  # Replace with your desired output file
process_axivity_data(input_csv, output_csv)
