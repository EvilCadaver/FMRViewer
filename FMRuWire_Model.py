import numpy as np
import time
import warnings
from scipy.integrate import quad, IntegrationWarning


Gamma = 1.399611 #GHz/kOe

Hk = float(4.0) #kOe
Ms = float(6) #kGauss
phi = float(30) #deg
Ho = float(10) #kOe
gyromagnetic_factor = float(2)
freq = float(36) #GHz
omg = (2*np.pi*freq/Gamma/gyromagnetic_factor) # omega/gamma 
alpha = 1e-3 #Gilbert damping

def correct_theta(vals):
    vals = [x for x in vals if x>0]
    return min(vals) if vals else None

def find_thetas(H = Ho, Hk = Hk, Ms = Ms, phi = phi):
    phi = np.radians(phi)
    A = 4*np.pi*Ms-Hk*(np.sin(phi)**2-np.cos(phi)**2)
    B = Hk*np.sin(2*phi)
    # (A**2+B**2)*x**4 + 2*B*H*x**3 + (H**2 - A**2 - B**2)*x**2 - B*H*x + B**2/4 = 0, x = sin(theta)
    a = (A**2+B**2)
    b = 2*B*H
    c = (H**2 - B**2 - A**2)
    d = - B*H
    e = B**2/4
    
    coeffs = [e, d, c, b, a]
    p = np.polynomial.Polynomial(coeffs)
    roots = p.roots()

    print('Roots: ', roots)
    theta1 = [np.asin(x) for x in roots]
    theta2 = [np.pi - np.asin(x) for x in roots]
    thetas = theta1 + theta2
    eps = 1e-7
    
    true_thetas = [theta for theta in thetas if abs(H - (Hk/2*np.sin(2*(phi-theta))/np.sin(theta) - 4*np.pi*Ms*np.cos(theta)))<eps]

    print('Useable thetas:', np.rad2deg(true_thetas))
    print('Equation check:', [(A*np.sin(theta)*np.cos(theta)-B/2+B*np.sin(theta)**2+H*np.sin(theta)) for theta in true_thetas])
    print('H with replacements check:', [B/np.sin(theta)/2-B*np.sin(theta)-A*np.cos(theta) for theta in true_thetas])
    print('H original check:', [Hk/2*np.sin(2*(phi-theta))/np.sin(theta) - 4*np.pi*Ms*np.cos(theta) for theta in true_thetas])
    return true_thetas

# mu = (a + j*b)/(c + j*d) = (a*c + b*d)/(c**2 + d**2) + j*(b*c - a*d)/(c**2 + d**2)
# mu_Re = (a*c + b*d)/(c**2 + d**2)
def mu_Re(x = 0, H = Ho, Hk = Hk, Ms = Ms, phi = phi, omg = omg, alpha = alpha, theta = 0):
    
    Heff = H*np.cos(theta) - 2*np.pi*Ms*np.sin(theta)**2 + Hk*np.cos(phi-theta)**2
    A = 4*np.pi*Ms*np.cos(theta)*np.sin(x)*np.cos(x)
    B = Heff + 4*np.pi*Ms*np.sin(x)**2
    C = Heff - Hk*np.sin(phi-theta)**2+4*np.pi*Ms*np.cos(theta)**2*np.cos(x)**2

    a = A**2 - (1 + alpha**2)*omg**2 + B*C + 4*np.pi*Ms*(np.sin(x)**2*np.cos(theta)**2*B + np.cos(x)**2*C)
    b = alpha*omg*(B + C + 4*np.pi*Ms*(np.sin(x)**2*np.cos(theta)**2 + np.cos(x)**2))
    b = -b #Kittel's variant
    c = B*C - A**2 - (1 + alpha**2)*omg**2
    d = alpha*omg*(B + C)

    return (a*c+b*d)/(c**2+d**2)

# mu_Im = (b*c - a*d)/(c**2 + d**2)
def mu_Im(x = 0, H = Ho, Hk = Hk, Ms = Ms, phi = phi, omg = omg, alpha = alpha, theta = 0):
    
    Heff = H*np.cos(theta) - 2*np.pi*Ms*np.sin(theta)**2 + Hk*np.cos(phi-theta)**2
    A = 4*np.pi*Ms*np.cos(theta)*np.sin(x)*np.cos(x)
    B = Heff + 4*np.pi*Ms*np.sin(x)**2
    C = Heff - Hk*np.sin(phi-theta)**2+4*np.pi*Ms*np.cos(theta)**2*np.cos(x)**2

    a = A**2 - (1 + alpha**2)*omg**2 + B*C + 4*np.pi*Ms*(np.sin(x)**2*np.cos(theta)**2*B + np.cos(x)**2*C)
    b = alpha*omg*(B + C + 4*np.pi*Ms*(np.sin(x)**2*np.cos(theta)**2 + np.cos(x)**2))
    b = -b #Kittel's variant
    c = B*C - A**2 - (1 + alpha**2)*omg**2
    d = alpha*omg*(B + C)

    return (b*c - a*d)/(c**2 + d**2)


print("Checking for H:")
for theta in (roots := find_thetas()):
    print(f"Theta = {np.rad2deg(theta):.4f}")

print("Chosen theta =", f"{np.rad2deg(theta := correct_theta(roots)):.5f}")

print("Re(mu(eta=0)) =", mu_Re(theta= theta))
print("Im(mu(eta=0)) =", mu_Im(theta= theta))

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", IntegrationWarning)
    start = time.perf_counter()
    result, err = quad(mu_Re, 0, 2*np.pi, epsabs=1e-12, epsrel=1e-12, limit=200, args=(Ho, Hk, Ms, phi, omg, alpha, theta))
    result = result/2/np.pi
    elapsed = time.perf_counter() - start
    print("Re(mu(eta_avg, 2*pi)) =", f"{result:.10f}", "Execution time =", f"{elapsed:.5f} (s)")
    if w:
        print("Warning:", w[0].message)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", IntegrationWarning)
    start = time.perf_counter()
    result, err = quad(mu_Im, 0, 2*np.pi, epsabs=1e-12, epsrel=1e-12, limit=200, args=(Ho, Hk, Ms, phi, omg, alpha, theta))
    result = result/2/np.pi
    elapsed = time.perf_counter() - start
    print("Im(mu(eta_avg, 2*pi)) =", f"{result:.10f}", "Execution time =", f"{elapsed:.5f} (s)")
    if w:
        print("Warning:", w[0].message)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", IntegrationWarning)
    start = time.perf_counter()
    result, err = quad(mu_Re, 0, np.pi, epsabs=1e-12, epsrel=1e-12, limit=200, args=(Ho, Hk, Ms, phi, omg, alpha, theta))
    result = result/np.pi
    elapsed = time.perf_counter() - start
    print("Re(mu(eta_avg, pi)) =", f"{result:.10f}", "Execution time =", f"{elapsed:.5f} (s)")
    if w:
        print("Warning:", w[0].message)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", IntegrationWarning)
    start = time.perf_counter()
    result, err = quad(mu_Im, 0, np.pi, epsabs=1e-12, epsrel=1e-12, limit=200, args=(Ho, Hk, Ms, phi, omg, alpha, theta))
    result = result/np.pi
    elapsed = time.perf_counter() - start
    print("Im(mu(eta_avg, pi)) =", f"{result:.10f}", "Execution time =", f"{elapsed:.5f} (s)")
    if w:
        print("Warning:", w[0].message)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", IntegrationWarning)
    start = time.perf_counter()
    result, err = quad(mu_Re, 0, np.pi/2, epsabs=1e-12, epsrel=1e-12, limit=200, args=(Ho, Hk, Ms, phi, omg, alpha, theta))
    result = result/np.pi*2
    elapsed = time.perf_counter() - start
    print("Re(mu(eta_avg, pi/2)) =", f"{result:.10f}", "Execution time =", f"{elapsed:.5f} (s)")
    if w:
        print("Warning:", w[0].message)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", IntegrationWarning)
    start = time.perf_counter()
    result, err = quad(mu_Im, 0, np.pi/2, epsabs=1e-12, epsrel=1e-12, limit=200, args=(Ho, Hk, Ms, phi, omg, alpha, theta))
    result = result/np.pi*2
    elapsed = time.perf_counter() - start
    print("Im(mu(eta_avg, pi/2)) =", f"{result:.10f}", "Execution time =", f"{elapsed:.5f} (s)")
    if w:
        print("Warning:", w[0].message)
