import pandas as pd
import matplotlib.pyplot as plt

# Define the data
data = {
    'Type': ['CoarseDropout', 'GaussianBlur', 'EnhanceSharpness', 'EnhanceContrast', 
             'EnhanceBrightness', 'EnhanceColor', 'Add', 'Invert', 'Multiply 1', 
             'Multiply 2', 'AddGaussianNoise', 'LinearContrast'],
    'TLESS_full_augmentation': [0.6750, 0.6750, 0.6750, 0.6750, 0.6750, 0.6750, 0.6750, 0.6750, 
                                0.6750, 0.6750, 0.6750, 0.6750],
    'TLESS_1000R_full_augmentation': [0.6200, 0.6200, 0.6200, 0.6200, 0.6200, 0.6200, 0.6200, 0.6200, 
                                      0.6200, 0.6200, 0.6200, 0.6200],
    'TLESS_1000R': [0.6260, 0.6290, 0.6690, 0.5990, 0.6230, 0.6600, 0.6510, 0.6400, 0.6580, 
                     0.6550, 0.6530, 0.6640],
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Set the 'Type' column as the index
df.set_index('Type', inplace=True)

# Plotting
plt.figure(figsize=(12, 6))

# Plot each dataset
ax = df.plot(kind='bar', ax=plt.gca())

# Adjust y-axis limits
min_value = df.min().min()
max_value = df.max().max()
ax.set_ylim(min_value - 0.01, max_value + 0.01)

plt.xlabel('Augmentation Type')
plt.ylabel('Mean Average Precision (mAP)')
plt.title('Comparison of Augmentation Types for TLESS Dataset')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Dataset')

plt.tight_layout()
plt.show()
