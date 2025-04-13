import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def generate_samples(gamma, n):
    u_samples = np.random.uniform(0, 1, n)
    u_samples = np.minimum(u_samples, 1 - 1e-15)

    x_samples = -(1 / gamma) * np.log(1 - u_samples)
    return x_samples

gamma = 2
n = 100

X1 = generate_samples(gamma, n)
X2 = generate_samples(gamma, n)

def check_independence(X1, X2, bins=10):
    # Calculate bin edges for both variables
    X1_min, X1_max = X1.min(), X1.max()
    X2_min, X2_max = X2.min(), X2.max()
    X1_edges = np.linspace(X1_min, X1_max, bins + 1)
    X2_edges = np.linspace(X2_min, X2_max, bins + 1)
    
    # Calculate joint distribution P(X1, X2) using histogram2d
    joint_counts, _, _ = np.histogram2d(X1, X2, bins=[X1_edges, X2_edges])
    joint_dist = joint_counts / joint_counts.sum()
    
    # Calculate marginal distributions P(X1) and P(X2)
    marginal_X1, _ = np.histogram(X1, bins=X1_edges)
    marginal_X1 = marginal_X1 / marginal_X1.sum()
    
    marginal_X2, _ = np.histogram(X2, bins=X2_edges)
    marginal_X2 = marginal_X2 / marginal_X2.sum()
    
    # Calculate the product of marginal distributions (expected joint distribution if independent)
    expected_joint = np.outer(marginal_X1, marginal_X2)
    
    # Calculate the absolute difference between actual and expected joint distributions
    diff = np.abs(joint_dist - expected_joint)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Calculate mutual information (0 for independent variables)
    mi = calculate_mutual_information(joint_dist, marginal_X1, marginal_X2)
    
    print(f"Max difference between joint and product of marginals: {max_diff:.6f}")
    print(f"Mean difference between joint and product of marginals: {mean_diff:.6f}")
    print(f"Mutual Information: {mi:.6f} (closer to 0 means more independent)")
    
    return joint_dist, expected_joint, X1_edges, X2_edges

def calculate_mutual_information(joint_dist, marginal_X1, marginal_X2):
    """Calculate mutual information between X1 and X2"""
    mi = 0
    for i in range(len(marginal_X1)):
        for j in range(len(marginal_X2)):
            if joint_dist[i, j] > 0:
                mi += joint_dist[i, j] * np.log(joint_dist[i, j] / (marginal_X1[i] * marginal_X2[j]))
    return mi

def visualize_independence(X1, X2, joint_dist, expected_joint, X1_edges, X2_edges):
    """Visualize the independence of X1 and X2"""
    fig = plt.figure(figsize=(15, 10))
    
    # Scatter plot of X1 and X2
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(X1, X2, alpha=0.5)
    ax1.set_title('Scatter Plot of X1 vs X2')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    
    # Plot marginal distributions
    ax2 = fig.add_subplot(2, 3, 2)
    X1_centers = (X1_edges[:-1] + X1_edges[1:]) / 2
    X2_centers = (X2_edges[:-1] + X2_edges[1:]) / 2
    ax2.plot(X1_centers, np.histogram(X1, bins=X1_edges)[0] / n, label='P(X1)')
    ax2.plot(X2_centers, np.histogram(X2, bins=X2_edges)[0] / n, label='P(X2)')
    ax2.set_title('Marginal Distributions')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Probability')
    ax2.legend()
    
    # Heatmap of joint distribution
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(joint_dist, cmap='viridis', origin='lower')
    ax3.set_title('Actual Joint Distribution P(X1,X2)')
    ax3.set_xlabel('X2 bins')
    ax3.set_ylabel('X1 bins')
    plt.colorbar(im3, ax=ax3)
    
    # Heatmap of expected joint distribution (product of marginals)
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(expected_joint, cmap='viridis', origin='lower')
    ax4.set_title('Expected Joint Distribution P(X1)×P(X2)')
    ax4.set_xlabel('X2 bins')
    ax4.set_ylabel('X1 bins')
    plt.colorbar(im4, ax=ax4)
    
    # Heatmap of difference between actual and expected
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(np.abs(joint_dist - expected_joint), cmap='viridis', origin='lower')
    ax5.set_title('|P(X1,X2) - P(X1)×P(X2)|')
    ax5.set_xlabel('X2 bins')
    ax5.set_ylabel('X1 bins')
    plt.colorbar(im5, ax=ax5)
    
    # Theoretical explanation
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.text(0.1, 0.9, "Independence Test Results:", fontsize=12, fontweight='bold')
    ax6.text(0.1, 0.7, "For independent variables X1 and X2:", fontsize=10)
    ax6.text(0.1, 0.6, "P(X1,X2) = P(X1) × P(X2)", fontsize=10)
    ax6.text(0.1, 0.4, "Note: Small differences between actual\n"
                     "and expected joint distributions are due\n"
                     "to random sampling and finite sample size.", fontsize=10)
    
    plt.tight_layout()
    plt.show()

# Run the independence check
joint_dist, expected_joint, X1_edges, X2_edges = check_independence(X1, X2, bins=10)

# Visualize the results
visualize_independence(X1, X2, joint_dist, expected_joint, X1_edges, X2_edges)

# Theoretical explanation: X1 and X2 are independent because:
# 1. They are generated independently using separate calls to generate_samples()
# 2. This creates two independent exponential random variables
# 3. The joint distribution closely matches the product of marginals
# 4. The mutual information is close to zero