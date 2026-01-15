import numpy as np
import pandas as pd

def generate_workload(
    num_tasks=50,
    short_frac=0.5,
    mid_frac=0.3,
    long_frac=0.2,
    arrival_spread=200,
    seed=42
):
    np.random.seed(seed)

    n_short = int(num_tasks * short_frac)
    n_mid = int(num_tasks * mid_frac)
    n_long = num_tasks - n_short - n_mid

    bursts = np.concatenate([
        np.random.uniform(3, 8, n_short),     # short tasks
        np.random.uniform(15, 30, n_mid),     # medium tasks
        np.random.uniform(60, 120, n_long)    # long tasks
    ])

    arrivals = np.sort(np.random.uniform(0, arrival_spread, num_tasks))

    df = pd.DataFrame({
        "arrival_time_ms": arrivals,
        "burst_time_ms": bursts
    })

    return df

if __name__ == "__main__":
    df = generate_workload()
    df.to_csv("dataset/workload_example.csv", index=False)
    print("Synthetic workload generated.")
