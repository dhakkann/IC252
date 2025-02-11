import random

def simulate_birthday_collection():
    """Simulates collecting unique birthdays until all 365 days are found."""
    unique_days = set()
    attempts = 0
    
    while len(unique_days) != 365:
        new_birthday = random.randrange(1, 366)
        unique_days.add(new_birthday)
        attempts += 1
    
    return attempts

def calculate_average_attempts():
    """Runs multiple simulations and returns the average attempts needed."""
    total_simulations = 1000
    results = [simulate_birthday_collection() for _ in range(total_simulations)]
    return round(sum(results) / total_simulations)

print(calculate_average_attempts())