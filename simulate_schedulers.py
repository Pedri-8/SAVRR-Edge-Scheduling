import numpy as np

# ===============================
# Synthetic workload (50 tasks)
# ===============================

np.random.seed(42)

NUM_TASKS = 50

arrivals = np.sort(np.random.uniform(0, 40, NUM_TASKS))
bursts = np.random.uniform(5, 50, NUM_TASKS)

assert len(arrivals) == len(bursts)

# ===============================
# SAVRR Scheduler
# ===============================

def run_savrr(arrivals, bursts):
    remaining = bursts.copy()
    time = 0.0
    completion = np.zeros(len(bursts))
    ready = []
    arrived = 0

    while True:
        while arrived < len(bursts) and arrivals[arrived] <= time:
            ready.append(arrived)
            arrived += 1

        if not ready:
            if arrived < len(bursts):
                time = arrivals[arrived]
                continue
            break

        rem = [remaining[i] for i in ready]
        mean = np.mean(rem)
        sigma = np.std(rem)
        Q = max(1, int(np.ceil(mean - sigma / 2)))

        new_ready = []
        for i in ready:
            if remaining[i] > Q:
                remaining[i] -= Q
                time += Q
                new_ready.append(i)
            else:
                time += remaining[i]
                completion[i] = time
                remaining[i] = 0

        ready = new_ready

    return completion


# ===============================
# HRRN Scheduler (Non-preemptive)
# ===============================

def run_hrrn(arrivals, bursts):
    time = 0.0
    completion = np.zeros(len(bursts))
    remaining = list(range(len(bursts)))
    arrived = []

    while remaining:
        for i in remaining:
            if arrivals[i] <= time and i not in arrived:
                arrived.append(i)

        if not arrived:
            time = min(arrivals[i] for i in remaining)
            continue

        ratios = [(time - arrivals[i] + bursts[i]) / bursts[i] for i in arrived]
        idx = arrived[np.argmax(ratios)]

        time += bursts[idx]
        completion[idx] = time
        remaining.remove(idx)
        arrived.remove(idx)

    return completion


# ===============================
# HEFT (Simplified for Independent Tasks)
# ===============================

def run_heft(arrivals, bursts):
    avg_speed = 2.0  # abstract heterogeneous speed
    est_finish = bursts / avg_speed
    order = np.argsort(est_finish)

    time = 0.0
    completion = np.zeros(len(bursts))

    for i in order:
        if arrivals[i] > time:
            time = arrivals[i]
        time += bursts[i]
        completion[i] = time

    return completion


# ===============================
# Metrics
# ===============================

def summarize(name, completion):
    latency = completion - arrivals
    utilization = bursts.sum() / completion.max()
    print(f"\n{name}")
    print(f"Mean Latency: {latency.mean():.2f} ms")
    print(f"CPU Utilization: {utilization:.3f}")


# ===============================
# Run Experiments
# ===============================

print("Running simulations...")

summarize("SAVRR", run_savrr(arrivals, bursts))
summarize("HRRN", run_hrrn(arrivals, bursts))
summarize("HEFT", run_heft(arrivals, bursts))
