import numpy as np

def correct_sintheta(vals, eps=1e-5):
    normalised = [0.0 if abs(x) <= eps else x for x in vals]
    candidates = [x for x in normalised if x>=0]
    return min(candidates) if candidates else None

Hk = 4.0 #kOe
Ms = 6 #kGauss
phi = 20 #deg
Ho = 10 #kOe

def find_thetas(H = Ho, Hk = Hk, Ms = Ms, phi = phi):
    phi = np.radians(phi)
    A = 4*np.pi*Ms-Hk*(2*np.sin(phi)**2-1)
    B = Hk*np.sin(2*phi)/2
    # (A**2+B**2)*x**4 + 2*B*H*x**3 + (H**2 - 4*B**2 - A**2)*x**2 - 4*B*H*x + 4*B**2 = 0, x = sin(theta)
    a = (A**2+B**2)
    b = 2*B*H
    c = (H**2 - 4*B**2 - A**2)
    d = - 4*B*H
    e = 4*B**2
    
    coeffs = [e, d, c, b, a]
    p = np.polynomial.Polynomial(coeffs)
    
    print('Roots: ', roots:=p.roots())
    print('H check: ', [(H*x + 2*np.pi*Ms*np.sin(2*np.asin(x))-Hk*np.sin(2*(phi-np.asin(x)))) for x in roots])
    return roots

print("Checking for H:")
for sintheta in (roots := find_thetas()):
    if sintheta**2<=1:
        theta = np.asin(sintheta)
        print(f"Theta = {np.rad2deg(theta):.4f}")

print("Chosen theta =", f"{np.rad2deg(theta := np.asin(correct_sintheta(roots))):.5f}")