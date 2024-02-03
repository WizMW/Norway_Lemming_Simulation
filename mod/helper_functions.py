import math
from scipy.special import binom

def return_b(): #born per day
    litter_prod = 2.5/365 #per day
    ind_per_litter = 7
    return litter_prod*ind_per_litter/2 

#Death per day as a lifespan result
def return_d():
    lifespan = 1/(2*365)
    return lifespan

#Death as a result of starvation 
def return_f(N):
    F = 75 #Lemmings that can eat every day
    D = 5 #Days after they starve miserably
    if N > F:
        sum = 0
        for i in range(D-1):
            sum += binom(D-1,i)*((F/N)**i)/(D+i) 
        return (((N - F)/N)**D)*sum

    else:   
        return 0    

