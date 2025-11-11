import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

labels = ["2-Ring Topo", "12-Ring Topo Mode1", "12-Ring Topo Mode2", "2-Ring Triv", "12-Ring Triv"]

df1 = pd.read_csv("SimulationData/2Ring_Topo_Eigen1_Conv19E-3.csv")
df2 = pd.read_csv("SimulationData/2Ring_Triv_Eigen1_Conv19E-4.csv")
df3 = pd.read_csv("SimulationData/12Ring_Topo_Eigen1_Conv125E-3.csv")
df4 = pd.read_csv("SimulationData/12Ring_Topo_Eigen2_Conv9E-3.csv")
df5 = pd.read_csv("SimulationData/12Ring_Triv_Eigen1_Conv144E-3.csv")

eigen_topo = [df1, df3, df4]
eigen_trivial = [df2, df5]
eigen_datasets = eigen_topo + eigen_trivial

# Mode amplitude datasets (for IPR)
dq1 = pd.read_csv("SimulationData/2Ring_Topo_Q1_Conv19E-3.csv")
dq2 = pd.read_csv("SimulationData/2Ring_Triv_Q1_Conv19E-4.csv")
dq3 = pd.read_csv("SimulationData/12Ring_Topo_Q1_Conv125E-3.csv")
dq4 = pd.read_csv("SimulationData/12Ring_Topo_Q2_Conv9E-3.csv")
dq5 = pd.read_csv("SimulationData/12Ring_Triv_Q1_Conv144E-3.csv")

amp_topo = [dq1, dq3, dq4]
amp_trivial = [dq2, dq5]
amp_datasets = amp_topo + amp_trivial


def compute_ipr(amplitudes):
    """Compute Inverse Participation Ratio for a normalized mode."""
    amp = amplitudes / np.linalg.norm(amplitudes)  # normalize
    return np.sum(amp**4)


# Compute mid-gap frequencies
for label, df, qf in zip(labels, eigen_datasets, amp_datasets):
    mode_cols = [c for c in df.columns if "Mode" in c]
    freqs = df.loc[0, mode_cols].values.astype(float)
    freqs = np.sort(freqs)
    
    # Take the two central modes and average them to define mid-gap
    mid_idx = len(freqs)//2
    mid_gap_freq = (freqs[mid_idx-1] + freqs[mid_idx])/2
    #print(f"{label}: mid-gap frequency = {mid_gap_freq:.6e}")

    # --- IPR for mid-gap mode ---
    q_cols = [c for c in qf.columns if "Q(" in c]
    mid_amplitudes = qf.loc[0, q_cols].values.astype(float)
    ipr_value = compute_ipr(mid_amplitudes)
    print(f"{label}: mid-gap frequency = {mid_gap_freq:.6e}, IPR = {ipr_value:.6f}")


