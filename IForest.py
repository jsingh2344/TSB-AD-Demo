# -*- coding: utf-8 -*-
"""Example of using TimeSeriesOD for time series anomaly detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

import os
import sys

import numpy as np
from pyod.models.ts_od import TimeSeriesOD
from pyod.utils.data import generate_ts_data

if __name__ == "__main__":
    contamination = 0.1 # percentage of outliers

    # Generate synthetic time series with anomalies
    X_train, X_test, y_train, y_test = generate_ts_data(
        n_train=500, n_test=200, contamination=0.05, random_state=42)
    
    plt.figure(figsize=(12,3))
    plt.plot(X_train, label='Train data')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.legend()
    #plt.show()

    anomaly_idx = np.where(y_train == 1)[0]
    print(anomaly_idx)
    plt.plot(anomaly_idx,X_train[anomaly_idx], 'ro', markersize=6, label='Anomalies')
    #plt.show()
    

    # TimeSeriesOD with IForest (default)
    clf_name = 'TimeSeriesOD'
    clf = TimeSeriesOD(window_size=5, contamination=contamination)
    clf.fit(X_train)

    print("Detector: %s" % clf_name)
    print("Number of anomalies: %d" % clf.labels_.sum())
    print("Top 5 anomaly scores:", np.sort(clf.decision_scores_)[-5:])

    pred_anomaly_idx = np.where(clf.labels_ == 1)[0]


    plt.figure(figsize=(12,3))
    plt.plot(X_train, label='Train data')
    plt.plot(anomaly_idx, X_train[anomaly_idx], 'ro', markersize=10, label='True Anomalies')
    plt.plot(pred_anomaly_idx, X_train[pred_anomaly_idx], 'go', markersize=6, label='Predicted Anomalies')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.legend()
    plt.show()