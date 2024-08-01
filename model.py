import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import newton

# Constants
g = 9.80665  # acceleration due to gravity, m/s^2
ho = 0.1  # initial height of water, meters
hf = 0.02  # final height of water, meters
d_pipe = 0.00794  # diameter of the pipe, meters
area_tank = 0.32 * 0.26  # cross-sectional area of the tank, m^2
area_pipe = math.pi * ((d_pipe / 2) ** 2)  # cross-sectional area of the pipe, m^2
sin_theta = 1 / 150  # sin(theta)
k_pipe = 0.5  # constant (pipe entrace pressure loss value)
    # for a sharp entrance, supposed to be 0.5. 
    # 0.3 is giving better results for the 20cm and 60cm measurements...
    # would just need to justify somehow, say we assumed a non perfectly flush entrance

L_outlet = 0.02 # length of one side of outlet pipe, meters
d_outlet = 0.01125 # diameter of t-joint outlet, meters
area_outlet = math.pi * ((d_outlet / 2) ** 2)
k_joint = 1.0 # minor loss constant for branching flow t-joint

# https://wiki.anton-paar.com/en/water/ @ 20 degrees celcius
    # using normal temperature pressure (NTP)
mu = 1.0016 * 1e-3  # dynamic viscosity of water, mPa * s -> Pa*s
pho = 0.9982 * 1e3  # density of water, kg/m^3, g/cm^3 -> kg/m^3

# https://enggcyclopedia.com/2011/09/absolute-roughness/
epsilon = 0.0015 * 1e-3  # surface roughness, mm -> m

def reynolds_number(v):
    return (v * d_pipe * pho) / mu

def velocity(f, h, L_pipe):
    num = (2 * g) * (h - (L_pipe * sin_theta))
    denom = (1 - ((area_pipe / area_tank) ** 2) + (f * (L_pipe / d_pipe)) + k_pipe)
    return np.sqrt(num / denom)

# friction factor function (Colebrook equation)
# turbulant
def colebrook(f, Re):
    # move everything over to the left side s.t. it = 0
    # when f is chosen s.t. the colebrooke eq = 0, we have found the correct f
    return 1 / np.sqrt(f) + 2.0 * np.log10(epsilon / (3.7 * d_pipe) + 2.51 / (Re * np.sqrt(f)))

# laminar
def laminar_f(f, Re):
    # when this = 0
    return 64/Re - f

# Define the integral function
def integrand(h, f, L_pipe):
    v = velocity(f, h, L_pipe)
    return 1 / v

# T-Joint Helper Functions
def velocity_t_joint(f_pipe, h, L_pipe, f_joint):
    num = (2 * g) * (h - (L_pipe * sin_theta))
    denom = (1 - ((area_pipe / area_tank) ** 2) + (f_pipe * (L_pipe / d_pipe)) + k_pipe) * (1 + (f_joint * (L_outlet / d_outlet)) + k_joint)
    return np.sqrt(num / denom)

# Iterative solution to find the friction factor
def solve_friction_factor(L_pipe, f_pipe=None, t_joint=False):
    # Initial guess for f
    f_guess = 0.01

    def func(f):
        h_mid = (ho + hf) / 2
        if t_joint:
            v = velocity_t_joint(f_pipe, h_mid, L_pipe, f)
        else:
            v = velocity(f, h_mid, L_pipe) 
        Re = reynolds_number(v)
        if Re <= 2300: return laminar_f(f, Re)
        if Re >= 4000: return colebrook(f, Re)
        else: return colebrook(f, Re) # transient (not accurate)
    
    # Use Newton-Raphson method to find the friction factor
        # ** since we're not supplying the deriv of the colebrook eq, scikit uses secant method
    # given a guess for f, iteratively go through to solve for f
        # once the colebrook eq = 0, we've found f
    f_solution = newton(func=func, x0=f_guess, maxiter=50) # fprime=func_deriv
        # secant method based on 2 initial guesses
        # since we're just passing one guess in, it auto picks a second one
            # guess 2: p1 = x0 * (1 + 1e-4)
    return f_solution

def integrand_t_joint(h, f_pipe, L_pipe, f_joint):
    v = velocity_t_joint(f_pipe, h, L_pipe, f_joint)
    return 1 / v

def calculate_drain_time(L_pipe):

    f_solution = solve_friction_factor(L_pipe)
    integral_value, _ = quad(integrand, ho, hf, args=(f_solution, L_pipe))
    
    # solve for t
    t_calculated = (-area_tank / area_pipe) * integral_value
    
    return t_calculated

def calculate_drain_time_with_tjoint(L_pipe):
    # find friction factor of the pipe first
    f_pipe = solve_friction_factor(L_pipe)
    f_joint = solve_friction_factor(L_pipe, f_pipe=f_pipe, t_joint=True)
    integral_value, _ = quad(integrand_t_joint, ho, hf, args=(f_pipe, L_pipe, f_joint))
    
    # solve for t
    t_calculated = (-area_tank / (2 * area_outlet) * integral_value)
    return t_calculated
    
experimental_results = {0.2: 199, 0.3: 213, 0.4: 266, 0.6: 288}

def calculate_error(pipe_length, result):
    experimental_result = experimental_results[pipe_length]
    error = abs((result-experimental_result)/experimental_result) * 100
    return error

if __name__ == "__main__":
    # Print the results
    print("RESULTS\n______________")
    print('{:<20s} {:<30s} {:<20s}'.format("Pipe length (L)", "Time to drain (seconds)", "Error (%)"))
    drain_time_20 = calculate_drain_time(0.2)
    print('{:<20} {:<30} {:<20}'.format("0.2", drain_time_20, calculate_error(0.2, drain_time_20)) )
    drain_time_30 = calculate_drain_time(0.3)
    print('{:<20} {:<30} {:<20}'.format("0.3", drain_time_30, calculate_error(0.3, drain_time_30)) )
    drain_time_40 = calculate_drain_time(0.4)
    print('{:<20} {:<30} {:<20}'.format("0.4", drain_time_40, calculate_error(0.4, drain_time_40)) )
    drain_time_60 = calculate_drain_time(0.6)
    print('{:<20} {:<30} {:<20}'.format("0.6", drain_time_60, calculate_error(0.6, drain_time_60)) )
    print("\n T-JOINT RESULTS\n_______")
    print('{:<20s} {:<30s} {:<20s}'.format("Pipe length (L)", "Time to drain (seconds)", "Error (%)"))
    print('{:<20} {:<30} {:<20}'.format("0.2", calculate_drain_time_with_tjoint(0.2), "N/A") )
    print('{:<20} {:<30} {:<20}'.format("0.3", calculate_drain_time_with_tjoint(0.3), "N/A") )
    print('{:<20} {:<30} {:<20}'.format("0.4", calculate_drain_time_with_tjoint(0.4), "N/A") )
    print('{:<20} {:<30} {:<20}'.format("0.6", calculate_drain_time_with_tjoint(0.6), "N/A") )