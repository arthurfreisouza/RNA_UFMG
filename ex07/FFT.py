import numpy as np
import matplotlib.pyplot as plt

# Generate time values from 0 to 1 second with a small time step
timestep=1000
begin_time = 0
end_time = 1

t = np.linspace(begin_time, end_time, timestep, endpoint=False)

# Create a complete sinusoidal cycle with a frequency of 1 Hertz
sinusoid = np.sin(2 * np.pi * 1 * t)

# Plot the sinusoidal cycle
plt.plot(t, sinusoid)
plt.title('Sinusoidal Cycle')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()