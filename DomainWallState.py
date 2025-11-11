
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Domain-Wall State in SSH Model (Python)
# Task Description: Simulates SSH model with a domain wall and visualizes mid-gap state localization.


# Import Files we want to use
df1 = pd.read_csv("SimulationData/2Ring_Topo_Q1_Conv19E-3.csv") 
df2 = pd.read_csv("SimulationData/12Ring_Topo_Q1_Conv125E-3.csv")
df3 = pd.read_csv("SimulationData/12Ring_Topo_Q2_Conv9E-3.csv")

def plot_domain_wall(df, plot_title):
    mode_cols = [col for col in df.columns if "Mode" in col]
    
    # Build q-value matrix: rows = modes, columns = ring positions
    q_matrix = []
    for mode in mode_cols:
        q_vals = df[mode]# normalize per mode
        q_matrix.append(q_vals.values)
    
    q_matrix = np.array(q_matrix)
    
    # Plot as heatmap
    plt.figure(figsize=(10,6))
    plt.imshow(q_matrix, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label='q-value (Normalized Mode Amplitude)')
    plt.xlabel('Ring Position')
    plt.ylabel('Mode Number')
    plt.title(plot_title)
    plt.show()

# Plot domain-wall states for 2-ring and 12-ring topological systems
plot_domain_wall(df1, "2-Ring Topological Resonator: Domain-Wall State (q-values)")
plot_domain_wall(df2, "12-Ring Topological Resonator: Domain-Wall State Eigenmode 1 (q-values)")
plot_domain_wall(df3, "12-Ring Topological Resonator: Domain-Wall State Eigenmode 2 (q-values)")
    


# Parameters
#N = 40               # number of sites
#t1 = 1.0             # intra-cell hopping
#t2 = 0.5             # inter-cell hopping
#domain_wall_index = N // 2

# Construct SSH Hamiltonian with a domain wall
#H = np.zeros((N, N))

#for i in range(N - 1):
#  if i < domain_wall_index:
#        hopping = t1 if i % 2 == 0 else t2
#    else:
#       hopping = t2 if i % 2 == 0 else t1
#    H[i, i + 1] = H[i + 1, i] = hopping

# Compute eigenvalues and eigenvectors
#eigvals, eigvecs = np.linalg.eigh(H)

# Find mid-gap mode (closest to zero energy)
#mid_idx = np.argmin(np.abs(eigvals))
#mid_val = eigvals[mid_idx]
#mid_vec = np.abs(eigvecs[:, mid_idx])**2  # amplitude squared

# --- Plot Results ---

#plt.figure(figsize=(12, 5))

# Plot eigenvalue spectrum
#plt.subplot(1, 2, 1)
#plt.title("SSH Eigenvalue Spectrum")
#plt.scatter(range(N), eigvals, color='black', s=15)
#plt.axhline(0, color='red', linestyle='--', label='Zero Energy')
#plt.xlabel("Eigenstate Index")
#plt.ylabel("Energy")
#plt.legend()

# Plot mid-gap mode localization
#plt.subplot(1, 2, 2)
#plt.title(f"Domain Wall State (Energy = {mid_val:.3f})")
#plt.bar(range(N), mid_vec, color='blue')
#plt.axvline(domain_wall_index, color='red', linestyle='--', label='Domain Wall')
#plt.xlabel("Lattice Site")
#plt.ylabel("|ψ|² (Probability Amplitude)")
#plt.legend()

#plt.tight_layout()
#plt.show()


