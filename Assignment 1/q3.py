num_simulations = 1000000
import random

p1WinCount = 0
p2WinCount = 0
drawCount = 0

for i in range(num_simulations):
    p1Sum = random.randint(1, 6) + random.randint(1, 6)
    p2Sum = random.randint(1, 6) + random.randint(1, 6) + random.randint(1, 6)

    if p1Sum == p2Sum:
        drawCount += 1
    elif p1Sum > p2Sum:
        p1WinCount += 1
    else:
        p2WinCount += 1

p1WinProb = p1WinCount / num_simulations
p2WinProb = p2WinCount / num_simulations
drawProb = drawCount / num_simulations

print("Probability of Player 1 winning:", p1WinProb)
print("Probability of Player 2 winning:", p2WinProb)
print("Probability of a draw:", drawProb)
print("Sum of probabilities:", p1WinProb + p2WinProb + drawProb)