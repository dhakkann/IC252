import random

num_couples = int(input("Enter number of couples "))
num_simulations = int(input("Enter number of simulations "))

all_people = list(i for i in range(2 * num_couples))
couples = [(2 * i, 2 * i + 1) for i in range(num_couples)]

results = []

for sim in range(num_simulations):
    random.shuffle(all_people)

    position = [0] * (2 * num_couples)
    for i in range(len(all_people)):
        position[all_people[i]] = i

    adjacent_couples = 0
    for couple in couples:
        person1, person2 = couple
        if abs(position[person1] - position[person2]) == 1 or abs(position[person1] - position[person2]) == (2 * num_couples - 1):
            adjacent_couples += 1

    results.append(adjacent_couples)

empirical_mean = sum(results) / num_simulations
empirical_var = sum((x - empirical_mean) ** 2 for x in results) / (num_simulations - 1)

p = 2 / (2 * num_couples - 1)
theoretical_mean = num_couples * p

theoretical_variance = num_couples * p * (1 - p) + num_couples * (num_couples - 1) * (4 / ((2 * num_couples - 1) * (2 * num_couples - 3)) - p**2)

print(f"Simulated E(T): {empirical_mean:.10f}")
print(f"Sample Var(T): {empirical_var:.10f}")
print(f"Theoretical E(T): {theoretical_mean:.10f}")
print(f"Theoretical Var(T): {theoretical_variance:.10f}")
