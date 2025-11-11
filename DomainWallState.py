import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Domain-Wall State in SSH Model (Python)
# Task Description: Simulates SSH model with a domain wall and visualizes mid-gap state localization.


# Import Files we want to use
df1 = pd.read_csv("SimulationData/2Ring_Topo_Q1_Conv19E-3.csv") 
df2 = pd.read_csv("SimulationData/2Ring_Triv_Q1_Conv19E-4.csv")
df3 = pd.read_csv("SimulationData/12Ring_Topo_Q1_Conv125E-3.csv")
df4 = pd.read_csv("SimulationData/12Ring_Topo_Q2_Conv9E-3.csv")
df5 = pd.read_csv("SimulationData/12Ring_Triv_Q1_Conv144E-3.csv")

# If we want to plot a singular one use this func below
def plot_dw_amp(df_array, plot_title):
    mode_cols = [col for col in df.columns if "Q(" in col]
    plt.figure(figsize=(10,6))

    for mode in mode_cols:
        # Take the single row of amplitudes for this mode
        amplitudes = df.loc[0, mode_cols].values
        # Normalize for q-value
        amplitudes = amplitudes / np.max(amplitudes)
        plt.plot(range(1, len(amplitudes)+1), amplitudes, marker='o', label=f"{mode}")
        
    plt.xlabel("Ring Position")
    plt.ylabel("Normalized Amplitude (q-value)")
    plt.title(plot_title)
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.show()

# === Function: Plot comparison for multiple datasets ===
def plot_dw_comparison_colored(df_arr, labels_arr, color_arr, title):
    # Safety check
    assert len(df_arr) == len(labels_arr) == len(color_arr), \
        "df_arr, labels_arr, and color_arr must have the same length"

    plt.figure(figsize=(10,6))

    for i, curr_df in enumerate(df_arr):
        curr_color = color_arr[i]
        curr_label = labels_arr[i]

        # Get all Q columns
        mode_cols = [col for col in curr_df.columns if "Q(" in col]
        amplitudes = curr_df.loc[0, mode_cols].values
        amplitudes = amplitudes / np.max(amplitudes)

        # Plot one line per dataset
        plt.plot(range(1, len(amplitudes) + 1), amplitudes,
                 color=curr_color, marker='o', label=curr_label)

    plt.xlabel("Mode Number")
    plt.ylabel("Normalized Amplitude (q-value)")
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.show()







#Plot the Two Graphs we need for comparison
#2-Ring Topological vs Trivial
plot_dw_comparison_colored(
    df_arr=[df1, df2],
    labels_arr=["2-Ring Topological", "2-Ring Trivial"],
    color_arr=["blue", "red"],
    title="2-Ring Resonator Comparison: Topological vs Trivial Domain-Wall States"
)

# 12-Ring Topological vs Trivial ---
plot_dw_comparison_colored(
    df_arr=[df3, df4, df5],
    labels_arr=[
        "12-Ring Topological (Mode 1)",
        "12-Ring Topological (Mode 2)",
        "12-Ring Trivial"
    ],
    color_arr=["green", "orange", "purple"],
    title="12-Ring Resonator Comparison: Topological vs Trivial Domain-Wall States"
)




# Plot domain-wall states for 2-ring and 12-ring topological systems
#plot_dw_amp(df1, "2-Ring Topological Resonator: Domain-Wall State (q-values)")
#plot_dw_amp(df2, "2-Ring Trivial Resonator: Domain-Wall State (q-values)")
#plot_dw_amp(df3, "12-Ring Topological Resonator: Domain-Wall State Eigenmode 1 (q-values)")
#plot_dw_amp(df4, "12-Ring Topological Resonator: Domain-Wall State Eigenmode 2 (q-values)")
#plot_dw_amp(df5, "12-Ring Trivial Resonator: Domain-Wall State (q-values)")
