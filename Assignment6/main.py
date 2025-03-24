import numpy as np
from math import comb, exp, factorial

class ProbabilityDistribution:
    def __init__(self):
        pass
    
    def binomial_distribution(self, trials, success_prob, sample_count):
        
        pmf = [comb(trials, k) * (success_prob ** k) * ((1 - success_prob) ** (trials - k)) for k in range(trials + 1)]
        
        cdf = [sum(pmf[:k+1]) for k in range(trials + 1)]
        
        random_values = np.random.uniform(0, 1, sample_count)
        result_samples = []
        for value in random_values:
            k = 0
            while k < trials and cdf[k] < value:
                k += 1
            result_samples.append(k)
        
        average = np.mean(result_samples)
        return result_samples, average
    
    def poisson_distribution(self, lambda_param, sample_count):
        
        random_values = np.random.uniform(0, 1, sample_count)
        result_samples = []
        
        for value in random_values:
            k = 0
            cumulative_prob = exp(-lambda_param)  
            while value > cumulative_prob:  
                k += 1
                cumulative_prob += (exp(-lambda_param) * (lambda_param ** k) / factorial(k))
            result_samples.append(k)
        
        average = np.mean(result_samples)
        return result_samples, average
    
    def analyze_sample_sizes(self, lambda_param):
        sizes = [10, 100, 1000, 10000]
        results = {}

        for size in sizes:
            generatedSamples, avg = self.poisson_distribution(lambda_param, size) 
            results[size] = avg  
        
        return results


generator = ProbabilityDistribution()

binomial_results, binomial_average = generator.binomial_distribution(trials=15, success_prob=0.25, sample_count=1000)

poisson_results, poisson_average = generator.poisson_distribution(lambda_param=0.75, sample_count=1000)

print("Binomial Experimental Mean:", binomial_average)
print("Binomial THEORETICAL Mean:", 3.75)
print("\nPoisson Experimental Mean:", poisson_average)
print("Poisson THEORETICAL Mean:", 0.75)

sample_size_analysis = generator.analyze_sample_sizes(lambda_param=0.75)
print("\nPoisson Mean Analysis for Different Sample Sizes:")
for size, avg in sample_size_analysis.items():
    print(f"Samples: {size}, Experimental Mean: {avg}")
