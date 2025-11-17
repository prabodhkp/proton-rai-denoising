# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:58:37 2025

@author: TRUE Lab
"""

# preprocessing_utils.py

import numpy as np
import h5py
import scipy.io
from scipy import signal

def zero_dead_channels(data):
    dead_channels = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113, 128, 129,
        144, 145, 160, 161, 176, 177, 192, 193, 208, 209, 224, 225,
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
        252, 253, 254, 255, 256
    ]
    dead_channels_idx = [ch - 1 for ch in dead_channels]
    data[dead_channels_idx, ...] = 0
    print(f"  Zeroed {len(dead_channels)} dead channels")
    return data

def load_test_data(file_path):
    print(f"Loading test data from: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            print("  Format: MATLAB v7.3")
            RF = np.array(f['RF'])
            if RF.shape[0] != 256:
                RF = np.transpose(RF, (2, 1, 0))
            Fs = float(np.array(f['Fs']).item()) if 'Fs' in f else None
            delay = float(np.array(f['delay']).item()) if 'delay' in f else 0.0
    except OSError:
        print("  Format: MATLAB v7 or earlier")
        data = scipy.io.loadmat(file_path)
        RF = data['RF']
        if RF.shape[0] != 256:
            RF = np.transpose(RF, (2, 1, 0))
        Fs = float(data['Fs'].item())
        delay = float(data['delay'].item()) if 'delay' in data else 0.0

    print(f"  RF shape: {RF.shape}")
    print(f"  Sampling frequency: {Fs/1e6:.2f} MHz")
    print(f"  Delay: {delay*1e6:.2f} µs")
    return RF, Fs, delay

def apply_delay_padding(RF, delay, Fs):
    if delay == 0:
        return RF
    num_pad_samples = int(np.round(delay * Fs))
    print(f"  Padding {num_pad_samples} samples at the beginning")
    RF_padded = np.pad(RF, ((0, 0), (num_pad_samples, 0), (0, 0)), mode='constant')
    return RF_padded

def bandpass_filter(data, fc1, fc2, Fs, axis=1):
    nyquist = Fs / 2
    low = fc1 / nyquist
    high = fc2 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=axis)

def downsample_data(data, Fs_original, Fs_target, axis=1):
    if Fs_original == Fs_target:
        return data
    decimation_factor = int(np.round(Fs_original / Fs_target))
    print(f"  Downsampling from {Fs_original/1e6:.2f} MHz to {Fs_target/1e6:.2f} MHz (factor: {decimation_factor})")
    return signal.decimate(data, decimation_factor, axis=axis, zero_phase=True)

def crop_or_pad(data, target_size, axis=1):
    current_size = data.shape[axis]
    if current_size == target_size:
        return data
    if current_size > target_size:
        print(f"  Cropping axis {axis} from {current_size} to {target_size}")
        if axis == 1:
            return data[:, :target_size, :]
        elif axis == 0:
            return data[:target_size, :, :]
    else:
        print(f"  Padding axis {axis} from {current_size} to {target_size}")
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (0, target_size - current_size)
        return np.pad(data, pad_width, mode='constant')
