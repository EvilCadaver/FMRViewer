import numpy as np

Hk = 1.0 #kOe
Ms = 8 #kOe
phi = 10 #deg
phi = np.radians(phi)

A = Hk*np.sin(2*phi)
B = 4*np.pi*Ms + Hk*np.cos(2*phi)
C = Hk/2*np.sin(2*phi)

H = 20 #kOe

# (A^2+B^2)*x^4 + 2*A*H*x^3 - (2*A*C+B^2)*x^2 - 2*H*C*x + C^2 = 0

a = (A**2+B**2)
b = 2*A*H
c = -(2*A*C+B**2)
d = -2*H*C
e = C**2

coeffs = [a, b, c, d, e]
roots = np.roots(coeffs)

print(roots)