import math
import numpy as np 


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
