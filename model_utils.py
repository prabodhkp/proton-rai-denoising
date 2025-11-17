# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:58:58 2025

@author: TRUE Lab
"""

# model_utils.py

import numpy as np
from tensorflow import keras

def load_model_and_normalization(model_path, norm_params_path):
    model = keras.models.load_model(model_path, compile=False)
    norm_params = np.load(norm_params_path, allow_pickle=True).item()
    return model, norm_params

def predict_in_chunks(RF, RFsum, model, norm_params, target_time_samples, n_avg, visualize_callback):
    num_frames = RF.shape[2]
    num_predictions = num_frames // n_avg
    print(f"  Total frames: {num_frames}")
    print(f"  Number of predictions: {num_predictions}")

    X_mean = norm_params['X_mean']
    X_std = norm_params['X_std']
    y_mean = norm_params['y_mean']
    y_std = norm_params['y_std']

    predictions = np.zeros((256, target_time_samples, num_predictions))
    correlations = []

    for i in range(num_predictions):
        start = i * n_avg
        end = start + n_avg
        input_avg = np.mean(RF[:, :, start:end], axis=2)
        X_input = input_avg.reshape(1, 256, target_time_samples, 1)
        X_normalized = (X_input - X_mean) / (X_std + 1e-8)
        y_pred_normalized = model.predict(X_normalized, verbose=0)
        y_pred = y_pred_normalized * y_std + y_mean
        prediction = y_pred[0, :, :, 0]

        predictions[:, :, i] = prediction
        corr = np.corrcoef(prediction.flatten(), RFsum.flatten())[0, 1]
        correlations.append(corr)

        print(f"  Prediction {i+1}/{num_predictions} (frames {start}-{end-1}): Corr = {corr:.4f}")
        visualize_callback(input_avg, RFsum, prediction, corr, i+1, start, end-1)

    return predictions, correlations
