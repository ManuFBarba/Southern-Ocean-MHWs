# -*- coding: utf-8 -*-
"""

############################# Extended Data Table S3 ##########################

"""

## Load required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Loading the data from Excel files into pandas DataFrames
correlation_df = pd.read_excel(r'.\Extended_Data_Table_S3/Correlation_Data.xlsx', index_col=0)
p_value_df = pd.read_excel(r'.\Extended_Data_Table_S3/p_value_Data.xlsx', index_col=0)

# Dropping NaNs which are probably due to the upper triangle being empty
correlation_df_clean = correlation_df.dropna(how='all').dropna(axis=1, how='all')
p_value_df_clean = p_value_df.dropna(how='all').dropna(axis=1, how='all')

# Mask for the significant values (assuming significance is determined by p-value < 0.05)
mask_significant = p_value_df_clean < 0.05

# Define the number of variables
num_vars = len(correlation_df_clean.columns)

# Redefine the figure and axis with adjustments for the color bar and marker style
fig, ax = plt.subplots(figsize=(10, 8))

# Plotting grid lines only around cells with values
for i in range(num_vars):
    for j in range(num_vars):
        if not np.isnan(correlation_df_clean.iloc[i, j]):
            # Draw a box around non-NaN cells
            lines = [
                [(j - 0.5, i - 0.5), (j + 0.5, i - 0.5)],  # Top
                [(j + 0.5, i - 0.5), (j + 0.5, i + 0.5)],  # Right
                [(j + 0.5, i + 0.5), (j - 0.5, i + 0.5)],  # Bottom
                [(j - 0.5, i + 0.5), (j - 0.5, i - 0.5)],  # Left
            ]
            lc = LineCollection(lines, color='black', linewidths=0.5)
            ax.add_collection(lc)

# Plotting the correlation values with 'X' markers
for i in range(num_vars):
    for j in range(num_vars):
        if np.isnan(correlation_df_clean.iloc[i, j]) or i < j:
            continue
        if mask_significant.iloc[i, j]:
            size = (0.05 / p_value_df_clean.iloc[i, j])**0.5 * 50
        else:
            size = 0
        ax.scatter(j, i, s=size, c=[correlation_df_clean.iloc[i, j]], 
                   vmin=-1, vmax=1, cmap='RdBu_r', edgecolor='black')

# Customize the ticks and labels with 45-degree rotation for x-axis labels
ax.set_xticks(np.arange(0, num_vars, 1))
ax.set_yticks(np.arange(0, num_vars, 1))
ax.set_xticklabels(correlation_df_clean.columns, rotation=45, ha="right")
ax.set_yticklabels(correlation_df_clean.index)
plt.gca().invert_yaxis()

# Remove the frame of the plot
for spine in ax.spines.values():
    spine.set_visible(False)

# Add color bar to the right of the plot
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-1, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, orientation='vertical', shrink=0.9, pad=0.01)
cbar.set_label('Significant Spearman correlation coefficient (ρ)')
cbar.set_ticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

# Adjust layout to prevent overlap
plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.2)


# Save figure so far
outfile = r'.\Extended_Data_Table_S3.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')


