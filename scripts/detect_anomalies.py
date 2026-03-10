import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import IsolationForest


def rolling_zscore(
    data: np.ndarray,
    window: int = 50,
    threshold: float = 3.0
) -> np.ndarray:

    from numpy.lib.stride_tricks import sliding_window_view

    n_rows, n_features = data.shape

    padded = np.pad(data, ((window - 1, 0), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, window_shape=window, axis=0)

    means = windows.mean(axis=2)
    stds = windows.std(axis=2)

    stds[stds == 0] = np.nan

    z = np.abs((data - means) / stds)
    z = np.nan_to_num(z, nan=0.0)

    return np.any(z > threshold, axis=1)


def cusum_detection(series: np.ndarray) -> np.ndarray:

    mean = series.mean()
    std = series.std()

    k = 0.5 * std
    h = 5.0 * std

    pos = 0.0
    neg = 0.0

    flags = np.zeros(len(series), dtype=np.bool_)

    for i, value in enumerate(series):

        diff = value - mean

        pos = max(0.0, pos + diff - k)
        neg = min(0.0, neg + diff + k)

        if pos > h or neg < -h:
            flags[i] = True
            pos = 0.0
            neg = 0.0

    return flags


def time_aware_isolation_forest(
    data: np.ndarray,
    lags: int = 3,
    contamination: float = 0.05
) -> np.ndarray:

    from numpy.lib.stride_tricks import sliding_window_view

    n_rows, n_features = data.shape
    window = lags + 1

    windows = sliding_window_view(data, window_shape=window, axis=0)

    X = np.ascontiguousarray(
        windows.reshape(n_rows - lags, window * n_features)
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    preds = model.fit_predict(X)

    anomalies = np.zeros(n_rows, dtype=np.bool_)
    anomalies[lags:] = preds == -1

    return anomalies


def detect_anomalies(
    df: pd.DataFrame,
    features: List[str]
) -> pd.DataFrame:

    data = df[features].to_numpy(dtype=np.float64)

    results = pd.DataFrame(index=df.index)

    results["zscore"] = rolling_zscore(data)

    results["cusum"] = cusum_detection(data[:, 0])

    results["iforest"] = time_aware_isolation_forest(data)

    return results

def main():

    normal = pd.read_csv("../data/prmon_normal.txt", sep="\t")
    cpu = pd.read_csv("../data/prmon_cpu_anomaly.txt", sep="\t")
    heavy = pd.read_csv("../data/prmon_heavy_anomaly.txt", sep="\t")

    normal["label"] = 0
    cpu["label"] = 1
    heavy["label"] = 1

    df = pd.concat([normal, cpu, heavy], ignore_index=True)

    df["cpu_total"] = df["utime"] + df["stime"]

    features = ["pss", "rss", "cpu_total"]

    results = detect_anomalies(df, features)

    df = pd.concat([df, results], axis=1)

    print("\nDetection Summary")
    print("-----------------")
    print("Z-score anomalies:", results["zscore"].sum())
    print("CUSUM anomalies:", results["cusum"].sum())
    print("IsolationForest anomalies:", results["iforest"].sum())

    import matplotlib.pyplot as plt

    plt.figure(figsize=(14,6))
    plt.plot(df["pss"], label="PSS Memory", linewidth=1)

    plt.scatter(
        df.index[df["zscore"]],
        df["pss"][df["zscore"]],
        color="red",
        label="Z-score",
        s=30
    )

    plt.scatter(
        df.index[df["iforest"]],
        df["pss"][df["iforest"]],
        color="green",
        label="IsolationForest",
        s=30
    )

    plt.scatter(
        df.index[df["cusum"]],
        df["pss"][df["cusum"]],
        color="purple",
        label="CUSUM",
        s=30
    )

    plt.legend()
    plt.title("Anomaly Detection on prmon Metrics")
    plt.xlabel("Time index")
    plt.ylabel("PSS Memory")

    plt.tight_layout()

    plt.savefig("../plots/anomaly_detection.png", dpi=300)

    print("\nPlot saved to ../plots/anomaly_detection.png")


if __name__ == "__main__":
    main()
