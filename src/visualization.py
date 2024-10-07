# src/visualization.py
import matplotlib.pyplot as plt

def plot_metrics(metric_data, title):
    plt.plot(metric_data)
    plt.title(title)
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.show()
