import random

def simulate_card_experiment(num_trials):
    cards = [
        ('red', 'red'),    
        ('black', 'black'), 
        ('red', 'black')   
    ]
    
    red_up_count = 0
    black_other_side = 0
    
    for i in range(num_trials):
        # select card
        card = random.choice(cards)
        
        # decide which side is up (0 or 1)
        up_side = random.randint(0, 1)
        
        # If up-facing side is red
        if card[up_side] == 'red':
            red_up_count += 1
            # Check if other side is black
            if card[1 - up_side] == 'black':
                black_other_side += 1
    
    probability = black_other_side / red_up_count if red_up_count > 0 else 0
    return probability

num_trials = 1000000
result = simulate_card_experiment(num_trials)
print("the probability that the other side")
print("is black, given that the upturned side is red:", result)