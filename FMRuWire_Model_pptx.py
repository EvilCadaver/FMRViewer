import csv
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import time
import warnings
from scipy.integrate import quad, IntegrationWarning

import matplotlib.pyplot as plt

H0 = float(5) #kOe
H_K = float(0.5) #kOe
M_S = float(0.65) #kGauss
PHI = float(45) #deg
ALPHA = 1e-3 #Gilbert damping
GYROMAG_FACTOR = float(2.0) 
FREQ = float(36) #GHz
GAMMA = 1.399611 #GHz/kOe
omg = (FREQ/(GAMMA*GYROMAG_FACTOR)) #omega/gamma 

def correct_theta(vals):
    vals = [x for x in vals if x>0]
    return min(vals) if vals else None

def find_thetas(H = H0, Hk = H_K, Ms = M_S, phi = PHI):
    phi = np.radians(phi)
    A = 4*np.pi*Ms-Hk*(np.sin(phi)**2-np.cos(phi)**2)
    B = Hk*np.sin(2*phi)
    ## (A**2+B**2)*x**4 + 2*B*H*x**3 + (H**2 - A**2 - B**2)*x**2 - B*H*x + B**2/4 = 0, x = sin(theta)
    a = (A**2+B**2)
    b = 2*B*H
    c = (H**2 - B**2 - A**2)
    d = - B*H
    e = B**2/4
    
    coeffs = [e, d, c, b, a]
    p = np.polynomial.Polynomial(coeffs)
    roots = p.roots()
    roots = [x for x in roots if -1 < x < 1]
    # print('Roots: ', roots)
    
    theta1 = [np.asin(x) for x in roots]
    theta2 = [np.pi - np.asin(x) for x in roots]
    thetas = theta1 + theta2
    
    eps = 1e-7
    true_thetas = [theta for theta in thetas if abs(H - (Hk/2*np.sin(2*(phi-theta))/np.sin(theta) - 4*np.pi*Ms*np.cos(theta)))<eps]

    # print('Useable thetas:', np.rad2deg(true_thetas))
    # print('Equation check:', [(A*np.sin(theta)*np.cos(theta)-B/2+B*np.sin(theta)**2+H*np.sin(theta)) for theta in true_thetas])
    # print('H with replacements check:', [B/np.sin(theta)/2-B*np.sin(theta)-A*np.cos(theta) for theta in true_thetas])
    # print('H original check:', [Hk/2*np.sin(2*(phi-theta))/np.sin(theta) - 4*np.pi*Ms*np.cos(theta) for theta in true_thetas])
    return true_thetas

def mu_eff(eta = 0, H = H0, Hk = H_K, Ms = M_S, phi = PHI, omg = omg, alpha = ALPHA):
    
    theta = correct_theta(find_thetas(H= H, Hk= Hk, Ms= Ms, phi= phi))
    Heff = H*np.cos(theta) - 2*np.pi*Ms*np.sin(theta)**2 + Hk*np.cos(phi-theta)**2
    A = 4*np.pi*Ms*np.cos(theta)*np.sin(eta)*np.cos(eta)
    B = Heff + 4*np.pi*Ms*np.sin(eta)**2
    C = Heff - Hk*np.sin(phi-theta)**2+4*np.pi*Ms*np.cos(theta)**2*np.cos(eta)**2

    a = A**2 - (1 + alpha**2)*omg**2 + B*C + 4*np.pi*Ms*(np.sin(eta)**2*np.cos(theta)**2*B + np.cos(eta)**2*C)
    b = alpha*omg*(B + C + 4*np.pi*Ms*(np.sin(eta)**2*np.cos(theta)**2 + np.cos(eta)**2))
    b = -b #Kittel's variant
    c = B*C - A**2 - (1 + alpha**2)*omg**2
    d = alpha*omg*(B + C)
    d = -d #Kittel's variant

    return (a+1j*b)/(c+1j*d)
    # return [(a*c+b*d)/(c**2+d**2), 1j*(b*c - a*d)/(c**2 + d**2)]

def mu_eff_integrated(H = H0, Hk = H_K, Ms = M_S, phi = PHI, omg = omg, alpha = ALPHA):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", IntegrationWarning)
        start = time.perf_counter()
        mu_Re, err_mu_Re = quad(lambda x: np.real(mu_eff(x, H, Hk, Ms, phi, omg, alpha)), 0, np.pi/2, epsabs=1e-8, epsrel=1e-8, limit=50)
        mu_Re = mu_Re/np.pi
        mu_Im, err_mu_Im = quad(lambda x: np.imag(mu_eff(x, H, Hk, Ms, phi, omg, alpha)), 0, np.pi/2, epsabs=1e-8, epsrel=1e-8, limit=50)
        mu_Im = mu_Im/np.pi
        elapsed = time.perf_counter() - start
        if w:
            print("Warning:", w[0].message)
            print(f"With inputs: H={H}, Hk={Hk}, Ms={Ms}, phi={phi}, omg={omg}, alpha={alpha}")
    return mu_Re + 1j*mu_Im, err_mu_Re + 1j*err_mu_Im, elapsed

# parameter_sets = [
#     {"H_K": 0.5, "M_S": 0.65, "phi": 15, "alpha": 1e-3, "g": 2.0, "f": 36},
#     {"H_K": 0.5, "M_S": 0.65, "phi": 30, "alpha": 1e-3, "g": 2.0, "f": 36},
#     {"H_K": 0.5, "M_S": 0.65, "phi": 0, "alpha": 5e-3, "g": 2.0, "f": 36},
# ]

def frange(start, stop, step, decimals=10):
    return np.round(np.arange(start, stop + step / 2, step), decimals)

param_ranges = {
    "alpha": [1e-3, 5e-3],
    "H_K": [0.5],
    "M_S": [0.65],
    "phi": frange(5, 90, 5),
    "g": [2.0],
    "f": [36],
}

parameter_sets = [
    dict(zip(param_ranges.keys(), values))
    for values in product(*param_ranges.values())
]

step = 5 #Oe       
H_oe = np.arange(step, 20000 + step, step)
H_koe = H_oe / 1000

def calculate_block(params):
    omg_i = params["f"] / (GAMMA * params["g"])

    mu_values = np.array([
        mu_eff_integrated(
            H=H,
            Hk=params["H_K"],
            Ms=params["M_S"],
            phi=params["phi"],
            omg=omg_i,
            alpha=params["alpha"],
        )[0]
        for H in H_koe
    ])

    mu_Re = np.real(mu_values)
    mu_Im = np.imag(mu_values)

    dP_dH = np.gradient(
        np.sqrt(np.sqrt(mu_Re**2 + mu_Im**2) + mu_Im),
        H_oe
    )

    return mu_Re, mu_Im, dP_dH

result_blocks = []

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        result_blocks = list(executor.map(calculate_block, parameter_sets))

    with open("./Output/mu_eff_sweep.csv", "w", newline="") as f:
        writer = csv.writer(f)

        # First row
        writer.writerow(
            ["H"] + ["mu_Re", "mu_Im", "dP/dH"] * len(parameter_sets)
        )

        # Parameter rows
        for param_name in ["H_K", "M_S", "phi", "alpha", "g", "f"]:
            row = [param_name]

            for params in parameter_sets:
                row.extend([params[param_name]] * 3)

            writer.writerow(row)

        # Data rows
        for i, H in enumerate(H_oe):
            row = [H]

            for mu_Re, mu_Im, dP_dH in result_blocks:
                row.extend([
                    mu_Re[i],
                    mu_Im[i],
                    dP_dH[i],
                ])

            writer.writerow(row)


# mu_values = np.array([mu_eff_integrated(H)[0] for H in H_koe])

# mu_Re = np.real(mu_values)
# mu_Im = np.imag(mu_values)
# dP_dH = np.gradient(np.sqrt(np.sqrt(mu_Re**2+mu_Im**2) + mu_Im), H_oe)

# fig, ax = plt.subplots()

# ax.plot(H_oe, mu_Re, label="Re(mu)")
# ax.plot(H_oe, mu_Im, label="Im(mu)")
# ax.plot(H_oe, dP_dH, label="dP/dH")

# ax.set_xlabel("H (Oe)")
# ax.set_ylabel("mu")
# ax.set_title("Effective permeability vs field")
# ax.legend()
# ax.grid(True)

# plt.show()
