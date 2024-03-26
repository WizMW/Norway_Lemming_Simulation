import random


class Lemming:
    def __init__(self):
        self.age = 0
        self.toughness = 5
        self.last_eaten = 0

    def check_dead(self, death_probability):
        # Calculate the probability of death for a given day
        if death_probability is None:
            death_probability = self.age/(365*2)  # Adjust the constant as needed
        return random.random() < death_probability

    def check_reproduce(self):
        givebirth_porp = 2.5/(365*2)
        if random.random() < givebirth_porp:
            return 7
        else:
            return 0

    def check_food(self, N, food):
        if N <= food:
            return True
        else:
            prop_food = food/N
            if random.random() < prop_food:
                return True
            else:
                return False

    def live_a_day(self, N, death_probability=None, food=75):
        # Check if the lemming survives the day


        if death_probability is not None:
            if death_probability > self.age/(365*2):
                alive = not self.check_dead(death_probability)
            else: 
                 alive = not self.check_dead(self.age/(365*2))   
        else:
            alive = not self.check_dead(death_probability)


        reproduce = self.check_reproduce()
        if not self.check_food(N, food):
            self.last_eaten += 1
        else:
            self.last_eaten = 0
        alive = (alive and (self.last_eaten < self.toughness))
        if alive:
            return alive, reproduce
        else:
            reproduce = 0
            return alive, reproduce
