import random

def simulate_card_experiment(num_trials):
    cards = [
        ('red', 'red'),    # Card 1
        ('black', 'black'), # Card 2
        ('red', 'black')   # Card 3
    ]
    
    red_up_count = 0
    black_other_side = 0
    
    for _ in range(num_trials):
        # Randomly select a card
        card = random.choice(cards)
        
        # Randomly decide which side is up (0 or 1)
        up_side = random.randint(0, 1)
        
        # If the up-facing side is red
        if card[up_side] == 'red':
            red_up_count += 1
            # Check if other side is black
            if card[1 - up_side] == 'black':
                black_other_side += 1
    
    probability = black_other_side / red_up_count if red_up_count > 0 else 0
    return probability

num_trials = 1000000
result = simulate_card_experiment(num_trials)
print(f"After {num_trials:,} trials, the probability that the other side")
print(f"is black, given that the upturned side is red: {result}")