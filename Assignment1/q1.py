import random

num_simulations = 10000

same_number = 0
sum_greater_than_9 = 0

for i in range(num_simulations):
    die1 = random.randint(1, 6)
    die2 = random.randint(1, 6)
    
    if die1 == die2:
        same_number += 1
    
    if die1 + die2 > 9:
        sum_greater_than_9 += 1

prob_same_number = same_number / num_simulations
prob_sum_greater_9 = sum_greater_than_9 / num_simulations

print("Probability of both dice showing the same number:", prob_same_number)
print("Probability of sum being greater than 9:", prob_sum_greater_9)