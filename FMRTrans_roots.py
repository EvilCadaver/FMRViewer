import numpy as np

def correct_sintheta(vals):
    vals = [x for x in vals if x>0]
    return min(vals) if vals else None

Hk = 4.0 #kOe
Ms = 6 #kGauss
phi = 90 #deg
Ho = 1 #kOe

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

print("Checking for H:")
for theta in (roots := find_thetas()):
    print(f"Theta = {np.rad2deg(theta):.4f}")

print("Chosen theta =", f"{np.rad2deg(correct_sintheta(roots)):.5f}")