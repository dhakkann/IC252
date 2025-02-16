import random
from math import perm,comb,factorial




def check_triple_occurrence(birthday_list):    
    for birthday in birthday_list:
        occurrences = sum(1 for day in birthday_list if day == birthday)
        if occurrences >= 3:
            return True
    return False

class BirthdaySimulation:
    DAYS_IN_YEAR = 365
    SIMULATION_COUNT = 100000
    
    def __init__(self, required_matches, group_size):
        self.matches_needed = required_matches
        self.people_count = group_size
        self.run_monte_carlo()
        self.calculate_theoretical()
    
    def run_monte_carlo(self):
        successful_matches = 0
        possible_days = list(range(1, self.DAYS_IN_YEAR + 1))

        for _ in range(self.SIMULATION_COUNT):
            random_birthdays = random.choices(possible_days, k=self.people_count)
            if check_triple_occurrence(random_birthdays):
                successful_matches += 1

        probability = successful_matches / self.SIMULATION_COUNT
        print(f"Monte Carlo Probability ({self.SIMULATION_COUNT} trials) "
              f"for {self.matches_needed}+ shared birthdays: {probability}")

    def calculate_theoretical(self):
        analyticalProb = perm(365, 25) / pow(365, 25)
        for cases in range(1, 25 // 2 + 1):
            pairs = 1
            for n_mult in range(cases):
                pairs *= comb(25 - 2 * n_mult, 2) * (365 - n_mult)
            remaining_unique = perm(365 - cases, 25 - 2 * cases)
            analyticalProb += (pairs * remaining_unique) / (pow(365, 25) * factorial(cases))

        analyticalProb = 1 - analyticalProb
        
        print(f"Theoretical Probability for {self.matches_needed}+ "
              f"shared birthdays: {analyticalProb:.5f}")

simulation = BirthdaySimulation(3, 25)







