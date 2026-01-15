# SAVRR – Synthetic Workload Scheduling Experiments

This repository provides a reproducible implementation of the scheduling experiments
reported in the paper:

**“SAVRR: A Self-Adaptive, Parameter-Free Scheduling Algorithm for Latency Reduction in Dynamic Edge Computing Environments.”**

---

## Experiment Description

The experiments are conducted using synthetically generated task workloads to emulate
dynamic IoT–Edge computing environments.

Each task is characterized by:
- Arrival time (milliseconds)
- CPU burst time (milliseconds)

The workload is generated programmatically using a fixed random seed to ensure
reproducibility.

---

## Files

- `simulate_schedulers.py`  
  Implements SAVRR, HRRN, and HEFT scheduling algorithms and computes:
  - Mean task latency
  - CPU utilization

- `dataset/`  
  Contains a representative sample synthetic workload used for reference.

---

## How to Run

### Requirements
- Python 3.8+
- NumPy

Install dependency:
```bash
pip install numpy
