import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import random

class CustomDistribution:
    def __init__(self, lambda_val, mu_val):
        if lambda_val <= 0 or mu_val <= 0:
            raise ValueError("Parameters must be positive: lambda > 0, mu > 0")
        
        self.lambda_val = lambda_val
        self.mu_val = mu_val
    
    def pdf(self, x):
        x = np.asarray(x)  # Ensure x is a NumPy array for consistent handling
        result = np.zeros_like(x, dtype=float)
        mask = x > self.mu_val
        result[mask] = 2 * self.lambda_val * (x[mask] - self.mu_val) * np.exp(-self.lambda_val * (x[mask] - self.mu_val)**2)
        return result if result.size > 1 else result.item()  # Return scalar if input was scalar
    
    def cdf_analytical(self, x):
        x = np.asarray(x)  # Ensure x is a NumPy array for consistent handling
        result = np.zeros_like(x, dtype=float)
        mask = x > self.mu_val
        result[mask] = 1 - np.exp(-self.lambda_val * (x[mask] - self.mu_val)**2)
        return result if result.size > 1 else result.item()  # Return scalar if input was scalar
    
    def cdf_numerical(self, x):
        if x <= self.mu_val:
            return 0
        
        result, _ = integrate.quad(self.pdf, self.mu_val, x)
        return result
    
    def theoretical_mean(self):
        return self.mu_val + 1 / np.sqrt(2 * self.lambda_val)
    
    def inverse_cdf(self, p):

        if p <= 0:
            return self.mu_val
        if p >= 1:
            # Technically infinity, but we'll return a large value
            return self.mu_val + 100
        
        # F(x) = 1 - exp(-lambda * (x - mu)^2)
        # p = 1 - exp(-lambda * (x - mu)^2)
        # exp(-lambda * (x - mu)^2) = 1 - p
        # -lambda * (x - mu)^2 = ln(1 - p)
        # (x - mu)^2 = -ln(1 - p) / lambda
        # x - mu = sqrt(-ln(1 - p) / lambda)
        # x = mu + sqrt(-ln(1 - p) / lambda)
        
        return self.mu_val + np.sqrt(-np.log(1 - p) / self.lambda_val)
    
    def generate_samples(self, n_samples):
        
        uniform_samples = np.random.uniform(0, 1, n_samples)
        return np.array([self.inverse_cdf(p) for p in uniform_samples])
    
    def plot_pdf(self, x_start=None, x_end=10, step_size=0.03, lambdas=None, mus=None, figsize=(10, 6)):
        """
        Args:
            x_start (float, optional): Start of x range. Defaults to mu + 0.1.
            x_end (float): End of x range. Defaults to 10.
            step_size (float): Step size for x values. Defaults to 0.03.
            lambdas (list, optional): List of lambda values to plot. Defaults to [self.lambda_val].
            mus (list, optional): List of mu values to plot. Defaults to [self.mu_val].
            figsize (tuple): Figure size. Defaults to (10, 6).
        """
        if lambdas is None:
            lambdas = [self.lambda_val]
        if mus is None:
            mus = [self.mu_val]
        
        plt.figure(figsize=figsize)
        
        for lam in lambdas:
            for mu in mus:
                dist = CustomDistribution(lam, mu)
                x_start_actual = mu + 0.1 if x_start is None else max(mu + 0.1, x_start)
                x = np.arange(x_start_actual, x_end, step_size)
                plt.plot(x, dist.pdf(x), label=f'λ={lam}, μ={mu}')
        
        plt.title('Probability Density Function (PDF)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return plt.gcf()
    
    def plot_cdf(self, x_start=None, x_end=10, step_size=0.03, lambdas=None, mus=None, figsize=(10, 6)):
        """
        Args:
            x_start (float, optional): Start of x range. Defaults to mu + 0.1.
            x_end (float): End of x range. Defaults to 10.
            step_size (float): Step size for x values. Defaults to 0.03.
            lambdas (list, optional): List of lambda values to plot. Defaults to [self.lambda_val].
            mus (list, optional): List of mu values to plot. Defaults to [self.mu_val].
            figsize (tuple): Figure size. Defaults to (10, 6).
        """
        if lambdas is None:
            lambdas = [self.lambda_val]
        if mus is None:
            mus = [self.mu_val]
        
        plt.figure(figsize=figsize)
        
        for lam in lambdas:
            for mu in mus:
                dist = CustomDistribution(lam, mu)
                x_start_actual = mu + 0.1 if x_start is None else max(mu + 0.1, x_start)
                x = np.arange(x_start_actual, x_end, step_size)
                plt.plot(x, dist.cdf_analytical(x), label=f'λ={lam}, μ={mu}')
        
        plt.title('Cumulative Distribution Function (CDF)')
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return plt.gcf()
    
    def analyze_distribution(self, n_samples=5000):
        """
        Perform a comprehensive analysis of the distribution:
        1. Generate random samples
        2. Compare sample mean with theoretical mean
        3. Compare analytical and numerical CDF at the sample mean
        """
        # Generate samples
        samples = self.generate_samples(n_samples)
        
        # Calculate sample mean
        sample_mean = np.mean(samples)
        
        # Calculate theoretical mean
        theo_mean = self.theoretical_mean()
        
        # Calculate CDF at sample mean (both ways)
        cdf_analytical = self.cdf_analytical(sample_mean)
        cdf_numerical = self.cdf_numerical(sample_mean)
        
        # Print results
        print(f"Analysis for distribution with λ={self.lambda_val}, μ={self.mu_val}:")
        print(f"Sample mean (from {n_samples} samples): {sample_mean:.6f}")
        print(f"Theoretical (population) mean: {theo_mean:.6f}")
        print(f"Difference: {abs(sample_mean - theo_mean):.6f}")
        print(f"Percent error: {100 * abs(sample_mean - theo_mean) / theo_mean:.4f}%")
        print(f"CDF at sample mean:")
        print(f"  Analytical: {cdf_analytical:.6f}")
        print(f"  Numerical:  {cdf_numerical:.6f}")
        print(f"  Difference: {abs(cdf_analytical - cdf_numerical):.6f}")
        
        return {
            'samples': samples,
            'sample_mean': sample_mean,
            'theoretical_mean': theo_mean,
            'cdf_analytical': cdf_analytical,
            'cdf_numerical': cdf_numerical
        }

def main():
    """
    Main function to run the distribution analysis as required by the assignment.
    """
    # Part 1: Create plots for different parameter values
    # Define parameter values for plotting
    lambdas = [0.5, 1.0, 1.5, 2.0, 3.0]
    mus = [0.01, 0.25, 1.0, 2.0, 3.0]
    
    dist = CustomDistribution(lambda_val=1.0, mu_val=0.25)
    
    # Plot PDFs
    pdf_fig = dist.plot_pdf(x_end=10, lambdas=lambdas, mus=[0.25])
    pdf_fig.savefig('pdf_lambda_variation.png')
    
    pdf_fig = dist.plot_pdf(x_end=10, lambdas=[1.5], mus=mus)
    pdf_fig.savefig('pdf_mu_variation.png')
    
    # Plot CDFs
    cdf_fig = dist.plot_cdf(x_end=10, lambdas=lambdas, mus=[0.25])
    cdf_fig.savefig('cdf_lambda_variation.png')
    
    cdf_fig = dist.plot_cdf(x_end=10, lambdas=[1.5], mus=mus)
    cdf_fig.savefig('cdf_mu_variation.png')
    
    # Part 2: Generate samples and compare means and CDFs
    # Use the true parameter values: λ=1.5, μ=0.25
    true_dist = CustomDistribution(lambda_val=1.5, mu_val=0.25)
    
    # Perform the analysis
    results = true_dist.analyze_distribution(n_samples=5000)
    
    # Plot histogram of samples
    plt.figure(figsize=(10, 6))
    plt.hist(results['samples'], bins=50, density=True, alpha=0.7, label='Sample Histogram')
    
    # Plot the true PDF on top of the histogram
    x = np.linspace(min(results['samples']), max(results['samples']), 1000)
    plt.plot(x, true_dist.pdf(x), 'r-', label='True PDF')
    
    # Add vertical lines for sample and theoretical means
    plt.axvline(results['sample_mean'], color='b', linestyle='--', label=f'Sample Mean: {results["sample_mean"]:.4f}')
    plt.axvline(results['theoretical_mean'], color='g', linestyle='--', label=f'Theoretical Mean: {results["theoretical_mean"]:.4f}')
    
    plt.title('Histogram of 5000 Samples with λ=1.5, μ=0.25')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sample_histogram.png')
    
    plt.show()

main()