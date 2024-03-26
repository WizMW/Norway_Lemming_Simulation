import datetime
from scipy.special import binom
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from scipy.interpolate import LinearNDInterpolator




def return_b():
    """
    Calculate the number of births per day.
    Returns:
        float: The number of births per day.
    """
    litter_prod = 2.5/365  # per day
    ind_per_litter = 7
    return litter_prod*ind_per_litter/2


def return_d():
    """
    Calculate the number of death per day.
    Returns:
        float: The number of death per day.
    """
    return 1/(2*365)

# Death as a result of starvation


def return_f(N, F = 75):
    D = 5  # Days after they starve miserably
    if N > F:
        sum = 0
        for i in range(D-1):
            sum += binom(D-1, i)*((F/N)**i)/(D+i)
        return (((N - F)/N)**D)*sum

    else:
        return 0


def get_weather(start_date, t):

    data = np.load('data/weather_data.npy', allow_pickle=True).item()
    month = data['month']
    day = data['day']
    snow = data['snow']
    temp = data['temp']

    month_convert = np.array(month, int)
    snow_convert = np.array(snow, float)
    temp_convert = np.array(temp, float)

    date_format = "%m-%d"
    date = datetime.strptime(start_date, date_format)
    temp = date + timedelta(days=t)
    new_date = temp.strftime(date_format)

    if new_date == "02-29":
        new_date = "02-28"

    month_index = new_date.split("-",1)

    i = 0
    while month[i] != month_index[0]:
        i += 1

    index = i + int(month_index[1]) -1
    snow_depth, temp = np.array([snow_convert[index], temp_convert[index]])
    return snow_depth, temp


def function(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))




def fit_d_sd():


    data = np.load('data/d_sd.npy', allow_pickle=True).item()
    death_rate = data['d']
    snow_depth = data['snow_depth']

    x_data = snow_depth
    y_data = death_rate

    params, covariance = curve_fit(function, x_data, y_data, maxfev=10000)

    y_fit = function(x_data, *params)
    
    plt.figure()
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, y_fit, color='red', label='Fit')
    plt.xlabel('snow depth (m)')
    plt.ylabel('death rate')
    plt.legend()
    plt.title('Curve Fitting')
    plt.savefig('exp/d_sd_fit.png', dpi=1200)
    np.save('data/fit_d_sd_parameters', params, allow_pickle=True)


def get_d(snow_depth, data=None):




    if data is None:
        data = np.load('data/fit_d_sd_parameters.npy', allow_pickle=True)
    
    r = function(snow_depth, *data)
    return r



def load_data():
    data = np.load('data/f_sd_T.npy', allow_pickle=True).item()
        
    x = data['snow_depth']
    y = data['temperature']
    z = data['food']
    
    return x, y ,z


def fit_f_sd_T():

    X, Y, Z = load_data()
    
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    Z = Z.reshape(-1, 1)

    matrix = np.concatenate((X, Y), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(matrix, Z, test_size=0.2, random_state=42)

    mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam',alpha=0.001, max_iter=2000, random_state=1)
    y_temp = y_train.ravel()
    mlp_regressor.fit(X_train, y_temp)
    print('It took', mlp_regressor.n_iter_, 'iterations to fit the MLP model!')

    z_pred = mlp_regressor.predict(X_test)
   
    squared_error = mean_squared_error(y_test, z_pred)
    absolute_error = mean_absolute_error(y_test, z_pred)

    score = mlp_regressor.score(X_train, y_train)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='blue', label='Train Data')
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='red', label='Test Data')

    ax.set_xlabel('Snow Depth', fontsize=9)
    ax.set_ylabel('Temperature', fontsize=9)
    ax.set_zlabel('Food for N lemmings', fontsize=9)
    ax.set_title(f"Squared Error: {squared_error:.2f} / Absolute Error: {absolute_error:.2f} / Score: {score:.2f}", fontsize=8)
    ax.grid(True)
    ax.legend()
    plt.savefig('exp/f_sd_T_fit.png', dpi=1200)
    np.save('data/model_f', mlp_regressor, allow_pickle=True)




def get_f(snow_depth, temp, data=None):


    if snow_depth > 1 or snow_depth < 0.002002002002002002:
        print("snow_depth is outside the calibrated range of values")
        
    if temp > 30 or temp < -29.81981981981982:
        print("temperature is outside the calibrated range of values")
    
    if data is None:
        food_model = np.load('data/model_f.npy', allow_pickle=True).item()
        food = food_model.predict(np.array([[snow_depth, temp]]))
        return food
    else:
        food_model = data
        food = food_model.predict(np.array([[snow_depth, temp]]))
        return food