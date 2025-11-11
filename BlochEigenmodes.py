import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Task Description: ω(k) with the gap vs J2/J1 (P); annotate the transition at J2​/J1​=1.

#Load in our Data File
df1 = pd.read_csv("SimulationData/2Ring_Topo_Eigen1_Conv19E-3.csv")
df2 = pd.read_csv("SimulationData/12Ring_Topo_Eigen1_Conv125E-3.csv")
df3 = pd.read_csv("SimulationData/12Ring_Topo_Eigen2_Conv9E-3.csv")

# These arry of ratios represent the value of J2/J1
ratios = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2]


# Plot eigenfrequencies vs ring radius
modes = [col for col in df1.columns if "Mode" in col]
for mode in modes:
    plt.plot(df1["$r [mm]"], df1[mode], marker='o', label=mode)

plt.xlabel("r [mm]")
plt.ylabel("Eigenfrequency (Hz)")
plt.title("2-Ring Topological Resonator: Eigenfrequencies vs Ring Radius")
plt.legend()
plt.grid(True)
plt.show()


# Q Value Plots are here, do it for both
def plot_q_vals(df, title):
    mode_cols = [col for col in df.columns if "Mode" in col]
    # Create a matrix: rows = modes, columns = ring positions
    q_matrix = []
    for mode in mode_cols:
        q_vals = df[mode] / df[mode].max()  # Normalize per mode
        q_matrix.append(q_vals.values)
    q_matrix = np.array(q_matrix)  # shape: (num_modes, num_rings)
    
    plt.figure(figsize=(10,6))
    plt.imshow(q_matrix, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label='q-value (Normalized Amplitude)')
    plt.xlabel('Ring Position')
    plt.ylabel('Mode Number')
    plt.title(title)
    plt.show()


# Plot q-values for df2 (Eigen1)
plot_q_vals(df2, "12-Ring Topological Resonator: q-values for Eigenmode 1")

# Plot q-values for df3 (Eigen2)
plot_q_vals(df3, "12-Ring Topological Resonator: q-values for Eigenmode 2")

