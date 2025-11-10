import math
import numpy as np
import matplotlib.pyplot as plt


# Start w/ Our Saved Values Here
max_rings = 12


# These arry of ratios represent the value of J2/J1
ratios = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2]


# Create the Formula's for both our Upper & Lower band w(k)
# For the SSH (Su-Schrieffer-Heeger) model, the dispersion relation(or band structure) gives Bloch eigenfrequencies ω(k)
def eigenfrequencies_w(k: float, a: int, J1: int, J2: int, type: bool): 
    within = np.square(J1) + np.square(J2) + (2 * J1 * J2) * math.cos(k * a)
    sign = 1
    if type is False:
        sign = -1
    return sign * np.sqrt(within)


# The formula to calculate our Bandgap (The bandgap cloases at when J2/J1 = 1)
def bandgap_delta(j2: int, j1: int):
    return 2 * abs(j2 - j1)


# Create wavevector array
#k_vals = np.linspace(-np.pi/a, np.pi/a, 400)


# Plot the Diagrams Below Here
plt.axvline(0, color='k', linewidth=0.5)
plt.axhline(0, color='k', linewidth=0.5)
plt.title('Band Diagram ω(k) for SSH Model')
plt.xlabel('Wavevector k')
plt.ylabel('Eigenfrequency ω')
plt.legend()
plt.show()




