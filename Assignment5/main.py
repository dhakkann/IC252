import random

class GiftSimulation:
    def __init__(self, n):
        self.n = n  # number of employees
        
    def run_single_trial(self):
        # Generate random gift distribution
        gifts = list(range(self.n))
        random.shuffle(gifts)
        
        # Count how many employees got their own gift
        self_matches = sum(1 for i in range(self.n) if gifts[i] == i)
        return self_matches
    
    def calculate_probability(self, j, trials=10000):
        successful_trials = 0
        
        for _ in range(trials):
            matches = self.run_single_trial()
            if matches >= j:
                successful_trials += 1
                
        return successful_trials / trials



n = int(input("n: "))
j = int(input("j: "))


simulation = GiftSimulation(n)
probability = simulation.calculate_probability(j)

print(f"\nAfter 10,000 trials:")
print(f"Probability that at least {j} out of {n} employees")
print(f"receive their own gift: {probability}")



