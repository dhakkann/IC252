import numpy as np
import matplotlib.pyplot as plt

class UrnSimulation:
    def __init__(self, white_balls=4, black_balls=6, draws=5, trials=1000):
        self.white_balls = white_balls
        self.black_balls = black_balls
        self.draws = draws
        self.trials = trials
        self.results = []
        
    def simulate(self):
        # 1 represents white balls, 0 represents black balls
        urn = np.array([1] * self.white_balls + [0] * self.black_balls)
        
        # Perform trials
        for _ in range(self.trials):
            drawn_balls = np.random.choice(urn, size=self.draws, replace=False)

            white_count = np.sum(drawn_balls)
            self.results.append(white_count)
            
    def calculate_distribution(self):
        unique_values, counts = np.unique(self.results, return_counts=True)
        probabilities = counts / self.trials
        return unique_values, probabilities
    
    def plot_distribution(self):
        values, probs = self.calculate_distribution()
        
        plt.figure(figsize=(10, 6))
        plt.bar(values, probs)
        plt.title('Probability Distribution of White Balls Drawn')
        plt.xlabel('Number of White Balls')
        plt.ylabel('Probability')
        plt.grid(True, alpha=0.3)
        plt.show()

sim = UrnSimulation()
sim.simulate()
sim.plot_distribution()

values, probs = sim.calculate_distribution()
print("\nProbability Distribution:")
for i in range(len(values)):
    print(f"P(X = {values[i]}) = {probs[i]:.3f}")





