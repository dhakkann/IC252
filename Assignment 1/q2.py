import random

def simulate_ball_selection(num_simulations=100000):
    total_balls = 100
    blue_balls = 50
    red_balls = 50
    
    success_count = 0
    
    for i in range(num_simulations):
        # Create list of balls; 1 represents blue, 0 represents red
        balls = [1] * blue_balls + [0] * red_balls
        
        selected_balls = random.sample(balls, 5)
        
        blue_count = sum(selected_balls)
        
        if blue_count == 3:
            success_count += 1
    
    probability = success_count / num_simulations
    return probability

probability = simulate_ball_selection()
print("Simulated probability:", probability)


# As number of trials are increased, 
# the delta between the different simulated branches decreases 
# i.e. the simulated probability converges
# to the actual probability.

'''
choosing 5 balls from 100, the total number of ways to do this is given by: 
100! / (5! * 95!)

choosing 3 blue balls from 50, the total number of ways to do this is given by: 
50! / (3! * 47!)

choosing 2 red balls from 50, the total number of ways to do this is given by: 
50! / (2! * 48!)

the probability of selecting exactly 3 blue balls and 2 red balls 
when drawing 5 balls at random from a bag containing 100 balls (50 red and 50 blue) is given by:
(50! / (3! * 47!)) * (50! / (2! * 48!)) / (100! / (5! * 95!)) = 0.318910757055087
'''

print(((50*49*48)/(3*2*1)) * ((50*49)/(2*1)) / ((100*99*98*97*96)/(5*4*3*2*1)))