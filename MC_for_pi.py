# Use Monte Carlo methods to estimate the value of pi. Suppose we inscribe a 
# a circle with radius 1 m inside a square of width 1 m. The area of the square is 
# 1 m^2 and the circle area is pi. By randomly populating data within the square, 
# we can find how many points lie within the circle. The probability of picking a 
# point from a uniform distribution that lies within the circle is equal to the 
# ratios of the circle area with the square area. 

import numpy as np 
import matplotlib.pyplot as plt 
import random 

from library import set_plot_style
# Set plot style
set_plot_style()

def pi_estimate(N, store_coords=False):
    # N is the number of trials, store_coords is a boolean to store x and y coordinates
    # of the points that lie within the circle.
    # The area of the circle is pi*r^2 and the area of the square is 1.
    # The ratio of the area of the circle to the area of the square is pi/4.
    # The probability of picking a point from a uniform distribution that lies within the circle
    # is equal to the ratio of the area of the circle to the area of the square.
    # The area of the circle is pi*r^2 and the area of the square is 1.
    count = 0
    if store_coords:
        x_coords = []
        y_coords = []
        for i in range(N):
            x = random.uniform(-0.5, 0.5) 
            y = random.uniform(-0.5, 0.5)
            x_coords.append(x)
            y_coords.append(y) 
            if x**2 + y**2 <= 0.25:
                count += 1  
        return 4*count/N, x_coords, y_coords
    else:    
        for i in range(N):
            x = random.uniform(-0.5, 0.5) 
            y = random.uniform(-0.5, 0.5) 
            if x**2 + y**2 <= 0.25:
                count += 1  
        return 4*count/N

estimates = []
num_trials = np.arange(100, 10000, 10)
for N in num_trials:
    estimates.append(pi_estimate(N))
estimates = np.array(estimates)
fig,ax = plt.subplots()
ax.scatter(num_trials, estimates, s = 1, c='b')
ax.set_xlabel("Number of Trials", fontsize=12)
ax.set_ylabel(r"Estimate of $\pi$", fontsize=12)
ax.axhline(y=np.pi, color='r', ls='--', label='True Value of Pi')
ax.legend()
plt.tight_layout()
plt.savefig('pi_estimate.png', dpi=300)
plt.show()

# Calculate the percent error and plot the trend line 
error = np.abs(estimates - np.pi)/np.pi

# Do a moving average to smooth the error
window_size = 20
smoothed_error = np.convolve(error, np.ones(window_size)/window_size, mode='valid')
smoothed_trials = num_trials[:len(smoothed_error)]

fig,ax = plt.subplots()
ax.plot(num_trials, error*100, c='lightblue', lw=1, label='Raw Error')
ax.plot(smoothed_trials, smoothed_error*100, c='blue', lw=2, label='Smoothed Error')
ax.set_xlabel("Number of Trials", fontsize=12)
ax.set_ylabel("Percent Error", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('pi_percent_error.png', dpi=300)
plt.show()

# Create a square plot with an inscribed circle. Create a random seed for reproducibility
# np.random.seed(0)
# N = 1000
# estimates, x_coords, y_coords = pi_estimate(N, store_coords=True) 
# fig,ax = plt.subplots()
# ax.set_aspect('equal')
# ax.set_xlim(-0.5, 0.5)
# ax.set_ylim(-0.5, 0.5)
# ax.plot([0, 0], [-0.5, 0.5], c='k', lw=1)
# ax.plot([-0.5, 0.5], [0, 0], c='k', lw=1)   
# x = np.linspace(-0.5, 0.5, 100)
# y = np.sqrt(0.25 - x**2)
# ax.plot(x, y, c='b', lw=1)
# ax.plot(x, -y, c='b', lw=1)
# ax.scatter(x_coords, y_coords, s=1.5, c='r')
# plt.tight_layout()
# plt.savefig('pi_square.png')
# plt.show()