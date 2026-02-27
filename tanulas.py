import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 5)
print(x)  # [0, 2.5, 5, 7.5, 10]

# Minden elemre egyszerre, for ciklus nélkül
print(x * 2)        # [0, 5, 10, 15, 20]
print(x + 10)       # [10, 12.5, 15, 17.5, 20]
print(x ** 2)       # [0, 6.25, 25, 56.25, 100]
print(np.sin(x))    # szinusz minden elemre
print(np.sqrt(x))   # négyzetgyök minden elemre
print(np.exp(x))    # e^x minden elemre

x = np.linspace(0, 10, 100) 
y1 = x**2
y2 = np.sin(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(x, y1, color="blue", label="x négyzet")
ax1.set_title("Parabola")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend()
ax1.grid(True)

ax2.plot(x, y2, color="red", label="szinusz")
ax2.set_title("Szinusz")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()