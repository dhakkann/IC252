import numpy as np
import matplotlib.pyplot as plt

def generate_samples(gamma, n):
    u_samples = np.random.uniform(0, 1, n)
    u_samples = np.minimum(u_samples, 1 - 1e-15)

    x_samples = -(1 / gamma) * np.log(1 - u_samples)
    return x_samples

gamma = 2
n = 100


X1 = generate_samples(gamma, n)
X2 = generate_samples(gamma, n)


population_mean = 1 / gamma

sample_mean_X1 = np.mean(X1)
sample_mean_X2 = np.mean(X2)

correlation_matrix = np.corrcoef(X1, X2)
correlation = correlation_matrix[0, 1] 

# Create a figure for visualizing independence
plt.figure(figsize=(15, 12))

# 1. Scatter plot of X1 vs X2
plt.subplot(2, 2, 1)
plt.scatter(X1, X2, alpha=0.5, s=5)
plt.title('Scatter Plot of X1 vs X2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True, linestyle='--', alpha=0.6)

# 2. Create 2D histogram
plt.subplot(2, 2, 2)
plt.hist2d(X1, X2, bins=30, cmap='viridis')
plt.colorbar(label='Count')
plt.title('2D Histogram (Joint Distribution)')
plt.xlabel('X1')
plt.ylabel('X2')

# 3. Marginal distribution of X1
plt.subplot(2, 2, 3)
plt.hist(X1, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Marginal Distribution of X1')
plt.xlabel('X1')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.6)

# 4. Marginal distribution of X2
plt.subplot(2, 2, 4)
plt.hist(X2, bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
plt.title('Marginal Distribution of X2')
plt.xlabel('X2')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# Create a second figure to compare joint distribution with product of marginals
plt.figure(figsize=(15, 6))

# Compute the marginal distributions manually using matplotlib's hist function
# For X1
hist_x1, bins_x1 = np.histogram(X1, bins=30, density=True)
bin_centers_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2

# For X2
hist_x2, bins_x2 = np.histogram(X2, bins=30, density=True)
bin_centers_x2 = (bins_x2[:-1] + bins_x2[1:]) / 2

# Define grid points for visualization
x1_grid = np.linspace(min(X1), max(X1), 100)
x2_grid = np.linspace(min(X2), max(X2), 100)

# For independent random variables, the product of marginals should match joint distribution
# We'll visualize this concept with a conditional distribution

# First plot: Actual joint distribution (2D histogram)
plt.subplot(1, 2, 1)
plt.hist2d(X1, X2, bins=30, cmap='viridis')
plt.colorbar(label='Count')
plt.title('Actual Joint Distribution')
plt.xlabel('X1')
plt.ylabel('X2')

# Second plot: For specific values of X1, plot the distribution of X2
plt.subplot(1, 2, 2)
# Divide X1 into quartiles
q1 = np.percentile(X1, 25)
q2 = np.percentile(X1, 50)
q3 = np.percentile(X1, 75)

# Get X2 values corresponding to different ranges of X1
x2_given_x1_q1 = X2[X1 <= q1]
x2_given_x1_q2 = X2[(X1 > q1) & (X1 <= q2)]
x2_given_x1_q3 = X2[(X1 > q2) & (X1 <= q3)]
x2_given_x1_q4 = X2[X1 > q3]

# Plot conditional distributions
plt.hist(x2_given_x1_q1, bins=20, alpha=0.5, density=True, label=f'X1 ≤ {q1:.2f}')
plt.hist(x2_given_x1_q2, bins=20, alpha=0.5, density=True, label=f'{q1:.2f} < X1 ≤ {q2:.2f}')
plt.hist(x2_given_x1_q3, bins=20, alpha=0.5, density=True, label=f'{q2:.2f} < X1 ≤ {q3:.2f}')
plt.hist(x2_given_x1_q4, bins=20, alpha=0.5, density=True, label=f'X1 > {q3:.2f}')
plt.title('Conditional Distributions of X2 Given X1')
plt.xlabel('X2')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# Add a text box with independence metrics to the first figure
plt.figure(1)  # Go back to the first figure
plt.figtext(0.5, 0.01, f"Independence Metrics:\nCorrelation: {correlation:.6f}", 
            ha='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.suptitle("Independence Check of X1 and X2", fontsize=16, y=0.98)
plt.show()


print(f"Population Mean (Theoretical): {population_mean:.4f}")
print(f"Sample Mean for X1: {sample_mean_X1:.4f}")
print(f"Sample Mean for X2: {sample_mean_X2:.4f}")

print(f"Correlation between X1 and X2: {correlation:.6f}")