# %% Part 1
import mod.helper_functions as mod
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mod.lemming import Lemming


def population_balance(t, N):

    dNdt = N * (mod.return_b() - mod.return_d() - mod.return_f(N))

    return dNdt


N_INIT = 500
SIM_TIME = 3 * 365


sol = solve_ivp(fun=population_balance, t_span=(0, SIM_TIME), y0=np.array([N_INIT]),
                method='RK45')

# Plot the solution
fig, ax = plt.subplots(figsize=(17/2.54, 10/2.54))
ax.plot(sol.t, sol.y[0], label='Population')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population')
ax.legend()

# Set font properties
font_properties = {'family': 'Arial', 'size': 11}
plt.rc('font', **font_properties)

# Save the plot with specified resolution and folder
plt.savefig('./exp/part1.png', dpi=300)


# %% Part 2

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
    lemming_population = np.append(lemming_population, len(L))

fig, ax = plt.subplots(figsize=(17/2.54, 10/2.54))
ax.plot(sol.t, sol.y[0], label='Part 1')
ax.plot(lemming_population, label='Part 2')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population')
ax.legend()

font_properties = {'family': 'Arial', 'size': 11}
plt.rc('font', **font_properties)

plt.savefig('./exp/part2.png', dpi=300)


# %% Part 3
# mod.fit_f_sd_T()
# mod.fit_d_sd()


def population_balance(t, N):

    snow_depth, temp = mod.get_weather("01-01", t)

    food = mod.get_f(snow_depth, temp)
    death_rate = mod.get_d(snow_depth)

    if death_rate < mod.return_d():
        death_rate = mod.return_d()

    dNdt = N * (mod.return_b() - death_rate - mod.return_f(N, food))

    return dNdt


sol = solve_ivp(fun=population_balance, t_span=(0, SIM_TIME), y0=np.array([N_INIT], ),
                method='RK45')


L = [Lemming() for _ in range(N_INIT)]
lemming_population = np.empty(1)

for day in range(SIM_TIME):

    snow_depth, temp = mod.get_weather("01-01", day)
    current_population = len(L)
    new_lemmings = []
    death_prop_model = mod.get_d(snow_depth)
    food_model = mod.get_f(snow_depth, temp)

    for lemming in L:

        alive, reproduce = lemming.live_a_day(
            current_population, death_prop_model, food_model)
        if alive:
            new_lemmings.append(lemming)
        for _ in range(reproduce):
            new_lemmings.append(Lemming())
    L = new_lemmings
    lemming_population = np.append(lemming_population, len(L))

fig, ax = plt.subplots(figsize=(17/2.54, 10/2.54))
fig, ax = plt.subplots(figsize=(17/2.54, 10/2.54))
ax.plot(sol.t, sol.y[0], label='Part 1 with data model')
ax.plot(lemming_population, label='Part 2 with data model')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population')
ax.legend()

font_properties = {'family': 'Arial', 'size': 11}
plt.rc('font', **font_properties)

plt.savefig('./exp/part3.png', dpi=300)


# %%
