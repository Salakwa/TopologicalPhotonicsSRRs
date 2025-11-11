import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from BaseFormulas import *

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

disorder_strengths = np.linspace(0, 0.2, 10)


# Store results
ipr_results = {label: [] for label in labels}
s21_results = {label: [] for label in labels}  # placeholder for mid-gap transmission

# Loop through datasets
for label, df_amp, df_eig in zip(labels, amp_datasets, eigen_datasets):
    # Extract mode amplitudes
    mode_cols = [c for c in df_amp.columns if "Q(" in c]
    amplitudes = df_amp.loc[0, mode_cols].values.astype(float)
    
    # Extract eigenfrequencies
    eig_cols = [c for c in df_eig.columns if "Mode" in c]
    freqs = df_eig.loc[0, eig_cols].values.astype(float)
    freqs = np.sort(freqs)
    mid_idx = len(freqs)//2
    mid_gap_freq = (freqs[mid_idx-1] + freqs[mid_idx])/2
    
    # Loop over disorder strengths
    for delta in disorder_strengths:
        # Coupling disorder: perturb amplitudes
        perturbed = amplitudes * (1 + np.random.uniform(-delta, delta, size=amplitudes.shape))
        ipr = compute_ipr(perturbed)
        ipr_results[label].append(ipr)
        
        # S21 at mid-gap (approximate as squared amplitude at last site)
        s21 = perturbed[-1]**2 / np.sum(perturbed**2)
        s21_results[label].append(s21)

# --- Plot IPR vs disorder ---
plt.figure(figsize=(8,5))
for label in labels:
    plt.plot(disorder_strengths, ipr_results[label], marker='o', label=label)
plt.xlabel("Disorder Strength")
plt.ylabel("IPR (Mid-gap Mode)")
plt.title("IPR vs Disorder Strength")
plt.grid(True)
plt.legend(fontsize=8)
plt.show()

# --- Plot S21 vs disorder ---
plt.figure(figsize=(8,5))
for label in labels:
    plt.plot(disorder_strengths, s21_results[label], marker='o', label=label)
plt.xlabel("Disorder Strength")
plt.ylabel("S21 (Mid-gap)")
plt.title("Mid-gap Transmission vs Disorder Strength")
plt.grid(True)
plt.legend(fontsize=8)
plt.show()