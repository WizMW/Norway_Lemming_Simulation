import mod.helper_functions as mod
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mod.lemming import Lemming

def population_balance(t,N):

    dNdt = N * (mod.return_b() - mod.return_d() - mod.return_f(N))

    return dNdt


N_INIT = 500
SIM_TIME = 3 * 365

# Parameter definitions
save_interval = 200                   # Save interval [s]

t_save = np.arange(0,SIM_TIME,save_interval)   # Start at t_0 go with save_interval steps until t_end is reached

sol = solve_ivp(fun=population_balance, t_span=(0,SIM_TIME), y0=np.array([N_INIT,]), 
                     method='RK45')

# Plot the solution
fig, ax = plt.subplots(figsize=(17, 10))
ax.plot(sol.t, sol.y[0], label='Population')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population')
ax.legend()

# Set font properties
font_properties = {'family': 'Arial', 'size': 11}
plt.rc('font', **font_properties)

# Save the plot with specified resolution and folder
plt.savefig('./exp/part1.png', dpi=300)

L = [Lemming() for _ in range(N_INIT)]
lemming_population = np.empty(1)
for day in range(SIM_TIME):
    current_population = len(L)
    new_lemmings = []
    for lemming in L:
        alive, reproduce = lemming.live_a_day(current_population)
        if alive:
            new_lemmings.append(lemming)
        for _ in range(reproduce):
            new_lemmings.append(Lemming())
    L = new_lemmings
    lemming_population = np.append(lemming_population,len(L))

fig, ax = plt.subplots(figsize=(17, 10))
ax.plot(sol.t,sol.y[0], label='Part 1')
ax.plot(lemming_population, label='Part 2')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population')
ax.legend()

# Setze die Schriftartenigenschaften
font_properties = {'family': 'Arial', 'size': 11}
plt.rc('font', **font_properties)

# Speichere das Diagramm mit angegebener Auflösung und im Unterverzeichnis ./exp/
plt.savefig('./exp/part2.png', dpi=300)

