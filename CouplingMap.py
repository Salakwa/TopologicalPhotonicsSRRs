import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from BaseFormulas import *

# Geometry → coupling map
# Task Description: Heatmap of retrieved J1, J2 (or J2/J1) vs gap/rotation/spacing

#First Get all the Files We Need all our dfs
df1 = pd.read_csv("SimulationData/2Ring_Topo_Eigen1_Conv19E-3.csv")
df2 = pd.read_csv("SimulationData/2Ring_Triv_Eigen1_Conv19E-4.csv")
df3 = pd.read_csv("SimulationData/12Ring_Topo_Eigen1_Conv125E-3.csv")
df4 = pd.read_csv("SimulationData/12Ring_Topo_Eigen2_Conv9E-3.csv")
df5 = pd.read_csv("SimulationData/12Ring_Triv_Eigen1_Conv144E-3.csv")

datasets = [df1, df2, df3, df4, df5]
labels = ["2-Ring Topo", "2-Ring Triv", "12-Ring Topo Mode1", "12-Ring Topo Mode2", "12-Ring Triv"]
colors = ["blue", "red", "green", "orange", "purple"]


# --- Compute J2/J1 from eigenfrequencies ---
def compute_J_ratio(df, J1=1.0):
    """Assume first two central modes define the gap; returns J2/J1."""
    mode_cols = [c for c in df.columns if "Mode" in c]
    row = df.iloc[0]
    freqs = np.array([row[c] for c in mode_cols], dtype=float)
    freqs = np.sort(freqs)
    # Take gap as difference between central modes
    mid_idx = len(freqs)//2
    gap = freqs[mid_idx] - freqs[mid_idx-1]
    # Solve J2 from bandgap formula: Δ = 2|J2 - J1|
    J2 = J1 + gap/2
    ratio = J2 / J1
    return ratio, gap

# --- Plot J2/J1 vs ring radius ---
plt.figure(figsize=(8,6))
for df, label in zip(datasets, labels):
    geom_col = [c for c in df.columns if "$r" in c or "mm" in c][0]
    geom = df.iloc[0][geom_col]
    ratio, gap = compute_J_ratio(df)
    plt.scatter(geom, ratio, s=100, label=f"{label} (gap={gap:.2e})")

plt.xlabel("Ring radius $r$ [mm]")
plt.ylabel("J2/J1")
plt.title("SSH Coupling Ratio vs Ring Radius")
plt.grid(True)
plt.legend(fontsize=6, ncol=2)
#plt.legend()
plt.show()

# --- Optional: Plot SSH Band Diagram ---
k_vals = np.linspace(-np.pi, np.pi, 400)
J1_example, J2_example = 1.0, 1.5
omega_upper = [eigenfrequencies_w(k, 1, J1_example, J2_example, True) for k in k_vals]
omega_lower = [eigenfrequencies_w(k, 1, J1_example, J2_example, False) for k in k_vals]

plt.figure(figsize=(8,5))
plt.plot(k_vals, omega_upper, label="Upper Band")
plt.plot(k_vals, omega_lower, label="Lower Band")
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("Wavevector k")
plt.ylabel("ω(k)")
plt.title(f"SSH Band Diagram (J2/J1 = {J2_example/J1_example:.2f})")
plt.grid(True)
#plt.legend()
plt.legend(fontsize=6, ncol=2)
plt.show()

# --- Domain-Wall Amplitude Heatmap ---
def plot_dw_heatmap(df_arr, labels_arr, title):
    for i, df in enumerate(df_arr):
        mode_cols = [col for col in df.columns if "Mode" in col]  # pick all Mode columns
        if len(mode_cols) == 0:
            print(f"No Mode columns found in dataset {labels_arr[i]}, skipping.")
            continue

        # Build 2D array: rows = modes, columns = ring positions
        q_matrix = []
        for mode in mode_cols:
            amplitudes = df[mode].values
            if amplitudes.size == 0:
                continue
            amplitudes = amplitudes / np.max(amplitudes)  # normalize
            q_matrix.append(amplitudes)
        q_matrix = np.array(q_matrix)

        if q_matrix.size == 0:
            print(f"No valid amplitudes for {labels_arr[i]}, skipping.")
            continue

        plt.figure(figsize=(8,6))
        plt.imshow(q_matrix, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(label='Normalized Amplitude')
        plt.xlabel("Ring Position")
        plt.ylabel("Mode Number")
        plt.title(f"{title} - {labels_arr[i]}")
        plt.show()

# Call function
plot_dw_heatmap(datasets, labels, "Domain-Wall State Comparison (Heatmap/Amplitude)")










# Group datasets by category
topo_dfs = [df1, df3, df4]
triv_dfs = [df2, df5]

def combine_and_plot_heatmap(df_list, title):
    combined_matrix = []

    for df in df_list:
        mode_cols = [c for c in df.columns if "Mode" in c]
        for mode in mode_cols:
            amplitudes = df[mode].values
            if amplitudes.size == 0:
                continue
            amplitudes = amplitudes / np.max(amplitudes)  # normalize
            combined_matrix.append(amplitudes)

    combined_matrix = np.array(combined_matrix)
    if combined_matrix.size == 0:
        print(f"No valid amplitudes for {title}")
        return

    plt.figure(figsize=(10,6))
    im = plt.imshow(combined_matrix, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(im, label='Normalized Amplitude')
    plt.xlabel("Ring Position")
    plt.ylabel("Mode Number (stacked)")
    plt.title(title)

    # Add grid lines
    plt.grid(which='both', color='white', linestyle='--', linewidth=0.5)
    plt.show()

# Plot Topological Heatmap
combine_and_plot_heatmap(topo_dfs, "Topological Resonators: Domain-Wall State Heatmap")

# Plot Trivial Heatmap
combine_and_plot_heatmap(triv_dfs, "Trivial Resonators: Domain-Wall State Heatmap")