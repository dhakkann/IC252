import random
import math




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
        total_prob = 0
        for i in range(12):
            term = math.perm(self.DAYS_IN_YEAR, self.people_count-i-1)
            term /= (self.DAYS_IN_YEAR ** self.people_count * math.factorial(i+1))
            
            for j in range(self.people_count-2*i, self.people_count+1, 2):
                term *= math.comb(j, 2)
            total_prob += term

        final_prob = 1 - total_prob - (math.perm(self.DAYS_IN_YEAR, self.people_count)
                                     / (self.DAYS_IN_YEAR ** self.people_count))
        
        print(f"Theoretical Probability for {self.matches_needed}+ "
              f"shared birthdays: {final_prob:.5f}")

simulation = BirthdaySimulation(3, 25)







