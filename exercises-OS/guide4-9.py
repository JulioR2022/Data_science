from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
# This program calculates the tail area of a normal distribution given its mean and standard deviation.
# It uses the z-score to find the percentile of a given value.
# The program also plots the normal distribution curve with the tail area shaded.

def z_score(x,mean, std):
    return (x - mean) / std

def lower_tail_area(z):
    return norm.cdf(z)

def nomal_curve_with_tail(z, media= 0, std= 1, lower_tail = True):
    x = np.linspace(media - 4 * std, media + 4 * std, 1000)
    y = norm.pdf(x, media, std)

    plt.plot(x, y, label='Normal Distribution')
    plt.title('Normal Distribution with Tail Area')
    plt.xlabel('Observations')
    plt.ylabel('Density')

    x_fill = 0
    if lower_tail:
        x_fill = np.linspace(media - 4*std, media + z * std, 1000)
    
    else:
        x_fill = np.linspace(media + z * std, media + 4*std, 1000)

    y_fill = norm.pdf(x_fill, media, std)
    plt.fill_between(x_fill, y_fill, alpha=0.5, label='Tail Area',color='green')

    plt.axvline(x=media + z * std, color='red', linestyle='--', label='Z-Score')
    plt.legend()  
    plt.grid(True)
    plt.show()


message = 'Enter the mean and the standard deviation of the normal distribution: '
mean, std = map(float,input(message).split())

while True:
    entry = input('Enter a value to calculate the percentile (or "exit" to quit): ')
    lower_tail = input('Do you want to calculate the lower or upper tail area? (L/U): ')
    if lower_tail.lower() == 'l':
        lower_tail = True
    else:
        lower_tail = False
    if entry.lower() == 'exit':
        print('Exiting the program.')
        break

    entry = float(entry)
    z = z_score(entry, mean, std)
    p = lower_tail_area(z)
    print(f'The lower tail area for {entry} is: {p:.4f}')
    print(f'The upper tail area for {entry} is: {(1 - p):.4f}')
    nomal_curve_with_tail(z, mean, std, lower_tail)
