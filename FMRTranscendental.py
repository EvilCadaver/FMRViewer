import numpy as np

def correct_sintheta(vals, eps=1e-5):
    normalised = [0.0 if abs(x) <= eps else x for x in vals]
    candidates = [x for x in normalised if x>=0]
    return min(candidates) if candidates else None

Hk = 5.0 #kOe
Ms = 8 #kGauss
phi = 15 #deg

phi = np.radians(phi)
A = Hk*np.sin(2*phi)
B = 4*np.pi*Ms + Hk*np.cos(2*phi)
C = Hk/2*np.sin(2*phi)

H = 50 #kOe

# (A^2+B^2)*x^4 + 2*A*H*x^3 - (2*A*C+B^2)*x^2 - 2*H*C*x + C^2 = 0

a = (A**2+B**2)
b = 2*A*H
c = -(2*A*C+B**2)
d = -2*H*C
e = C**2

coeffs = [a, b, c, d, e]
print("Roots for sin(theta):", roots := np.roots(coeffs))

print("Checking for H:")
for sintheta in roots:
    if sintheta**2<=1:
        theta = np.asin(sintheta)
        print(f"Theta = {np.rad2deg(theta):.2f}")
        H_ = Hk/2*np.sin(2*(phi-theta))/np.sin(theta) - 4*np.pi*Ms*np.cos(theta)
        print(f"H = {H:.3f} kOe")

print("Chosen theta =", f"{np.rad2deg(theta := np.asin(correct_sintheta(roots))):.3f}")