import numpy as np
import pandas as pd

# -----------------------------
# Load synthetic dataset
# -----------------------------
data = pd.read_csv("../dataset/sample_synthetic_workload.csv")

arrivals = data["arrival_time_ms"].to_numpy(dtype=float)
bursts = data["burst_time_ms"].to_numpy(dtype=float)

assert len(arrivals) == len(bursts), "Dataset size mismatch!"

# -----------------------------
# SAVRR Scheduler
# -----------------------------
def run_savrr(arrivals, bursts):
    n = len(bursts)
    remaining = bursts.copy()
    completion = np.zeros(n)
    ready = []
    time = 0.0
    arrived = 0

    while True:
        while arrived < n and arrivals[arrived] <= time:
            ready.append(arrived)
            arrived += 1

        if not ready:
            if arrived < n:
                time = arrivals[arrived]
                continue
            break

        rem = [remaining[i] for i in ready]
        mean = np.mean(rem)
        sigma = np.std(rem)

        Q = max(1, int(np.ceil(mean - sigma / 2)))

        tid = ready.pop(0)
        if remaining[tid] > Q:
            time += Q
            remaining[tid] -= Q
            ready.append(tid)
        else:
            time += remaining[tid]
            completion[tid] = time
            remaining[tid] = 0

        if np.all(completion > 0):
            break

    return completion


# -----------------------------
# HRRN Scheduler (Non-preemptive)
# -----------------------------
def run_hrrn(arrivals, bursts):
    n = len(bursts)
    completion = np.zeros(n)
    time = 0.0
    ready = []
    arrived = 0

    while True:
        while arrived < n and arrivals[arrived] <= time:
            ready.append(arrived)
            arrived += 1

        if not ready:
            if arrived < n:
                time = arrivals[arrived]
                continue
            break

        ratios = [(time - arrivals[i] + bursts[i]) / bursts[i] for i in ready]
        tid = ready[np.argmax(ratios)]
        ready.remove(tid)

        time += bursts[tid]
        completion[tid] = time

        if np.all(completion > 0):
            break

    return completion


# -----------------------------
# HEFT Scheduler (Independent Tasks)
# -----------------------------
def run_heft(arrivals, bursts):
    """
    Simplified HEFT for independent tasks.
    Tasks are ordered by estimated execution time.
    """
    n = len(bursts)
    completion = np.zeros(n)
    time = 0.0

    # Average edge speed abstraction
    avg_speed = 2.3  # GHz (within 1.5–3.2 GHz as per paper)
    est_exec = bursts / avg_speed

    order = np.argsort(est_exec)

    for tid in order:
        if arrivals[tid] > time:
            time = arrivals[tid]
        time += bursts[tid]
        completion[tid] = time

    return completion


# -----------------------------
# Metrics
# -----------------------------
def summarize(name, completion):
    latency = completion - arrivals
    utilization = bursts.sum() / completion.max()

    print(f"\n{name}")
    print(f"Mean Latency: {latency.mean():.2f} ms")
    print(f"CPU Utilization: {utilization:.3f}")


# -----------------------------
# Run experiments
# -----------------------------
print("Running simulations...")

savrr = run_savrr(arrivals, bursts)
hrrn = run_hrrn(arrivals, bursts)
heft = run_heft(arrivals, bursts)

summarize("SAVRR", savrr)
summarize("HRRN", hrrn)
summarize("HEFT", heft)
