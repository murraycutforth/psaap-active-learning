import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the parameter space ---

# p: The Bernoulli probability of success.
# We'll create a linear space from 0 to 1.
p_values = np.linspace(0, 1, 201)  # 201 points for smooth gradient

# N: The number of samples.
# The request specifies a log scale for N.
# np.logspace is perfect for this, creating values evenly spaced on a log scale.
# We will generate 100 points from 10^1 (10) to 10^5 (100,000).
n_values = np.logspace(0, 2, 200, dtype=int)


# --- 2. Create a grid and calculate the SEM for each (p, N) pair ---

# np.meshgrid creates coordinate matrices from coordinate vectors.
# P and N will be 2D arrays.
P, N = np.meshgrid(p_values, n_values)

# Calculate the standard error on the mean using the formula:
# SEM = sqrt(p * (1 - p)) / sqrt(N)
# We handle the case where p=0 or p=1, where the numerator is 0.
# The calculation is done element-wise by numpy on the 2D arrays.
sem_values = np.sqrt(P * (1 - P)) / np.sqrt(N)

# --- 3. Plot the results as a heatmap ---

# Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(5, 4), dpi=200)


# Use pcolormesh to create the heatmap. It's well-suited for data on a
# non-uniform rectangular grid (like our log-scaled N-axis).
# `shading='auto'` helps avoid potential warnings with matplotlib versions.
heatmap = ax.pcolormesh(P, N, sem_values, shading='auto', cmap='viridis', alpha=1)
contour = ax.contour(P, N, sem_values, linewidths=1, levels=[0.1, 0.2, 0.3, 0.4, 0.5], colors='k')
manual_locations = [(0.5, 100), (0.5, 10), (0.5, 4), (0.5, 3) ]
ax.clabel(contour, contour.levels, inline=True, manual=manual_locations, colors='black')

# --- 4. Configure the plot aesthetics ---

# Set the y-axis to a logarithmic scale, as requested.
ax.set_yscale('log')

# Add a color bar to show the mapping of colors to SEM values
# The label describes what the color intensity represents.
cbar = fig.colorbar(heatmap, ax=ax)
cbar.set_label('Standard error on mean (SEM)', fontsize=12)

# Set labels for the axes and a title for the plot
ax.set_xlabel('Bernoulli parameter (p)', fontsize=12)
ax.set_ylabel('Number of samples (N)', fontsize=12)

# Ensure the layout is clean and labels do not overlap
plt.tight_layout()

# Display the plot
plt.show()