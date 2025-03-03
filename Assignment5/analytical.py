from math import factorial

class AnalyticalProbability:
    def __init__(self, n):
        self.n = n
        
    def calculate_derangement(self, n):

        if n == 0:
            return 1
        if n == 1:
            return 0
            
        # Using formula: !n = n! * (1/0! - 1/1! + 1/2! - ... + (-1)^n/n!)
        result = factorial(n)
        sum_term = 0
        
        for i in range(n + 1):
            # Calculate (-1)^i / i!
            sum_term += (-1)**i / factorial(i)
            
        return result * sum_term
    
    def nCr(self, n, r):
        return factorial(n) / (factorial(r) * factorial(n - r))
    
    def calculate_probability(self, j):
        total_probability = 0
        
        # Calculate probability for exactly k matches, where k >= j
        for k in range(j, self.n + 1):

            ways_to_choose = self.nCr(self.n, k)

            derangements = self.calculate_derangement(self.n - k)

            total_probability += ways_to_choose * derangements / factorial(self.n)
            
        return total_probability


n = int(input("Enter the number of employees (n): "))
j = int(input("Enter the minimum number of self-matches to check for (j): "))

calculator = AnalyticalProbability(n)
probability = calculator.calculate_probability(j)

print(f"\nAnalytical calculation:")
print(f"Probability that at least {j} out of {n} employees")
print(f"receive their own gift: {probability:.6f}")





