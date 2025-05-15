from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


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
mean, std = map(int,input(message).split())

while True:
    entry = input('Enter a percentile (or "exit" to quit): ')
    if entry.lower() == 'exit':
        print('Exiting the program.')
        break

    entry = float(entry)
    observation = norm.ppf(entry, loc=mean, scale=std)
    print(f'The observation corresponding to the percentile {entry} is: {observation:.4f}')
    z = z_score(observation, mean, std)
    p = lower_tail_area(z)
    print(f'The lower tail area for {entry} is: {p:.4f}')
    nomal_curve_with_tail(z, mean, std)
