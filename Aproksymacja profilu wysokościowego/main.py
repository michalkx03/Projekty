import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from methods import *
def load_data():
    data_everest = pd.read_csv("2018_paths/MountEverest.csv")
    x_everest = data_everest['Dystans (m)']
    y_everest = data_everest['Wysokość (m)']

    data_canyon = pd.read_csv("2018_paths/WielkiKanionKolorado.csv")
    x_canyon = data_canyon['Dystans (m)']
    y_canyon = data_canyon['Wysokość (m)']

    return x_everest,y_everest,x_canyon,y_canyon

def plot_lagrange_interpolation(num_points,data,x,y,chebyshev):
    b = len(x)
    if(chebyshev):
        indices = chebyshev_nodes(x,num_points)
    else:
        indices = normal_nodes(0,b-1,num_points)
    distance_nodes = [x[i] for i in indices]
    height_nodes = [y[i] for i in indices]
    distance_new = np.linspace(x.min(), x.max(), 500)
    height_new = lagrange_interpolation(distance_nodes, height_nodes, distance_new)

    plt.plot(x, y, '-', label='Dane oryginalne')
    plt.plot(distance_nodes, height_nodes, 's', label='Punkty węzłowe')
    if(chebyshev):
        plt.plot(distance_new, height_new, '-', label=f'Interpolacja Lagrange\'a ({num_points} węzłów Czebyszewa)')
        plt.title(f'Interpolacja Lagrange\'a - {data} ({num_points} węzłów Czebyszewa)')
    else:
        plt.plot(distance_new, height_new, '-', label=f'Interpolacja Lagrange\'a ({num_points} punktów)')
        plt.title(f'Interpolacja Lagrange\'a - {data} ({num_points} punktów)')
    plt.xlabel('Dystans (m)')
    plt.ylabel('Wysokość (m)')
    plt.legend()
    plt.grid(True)
    filename = f"{data}_{num_points}_{'chebyshev' if chebyshev else 'normal'}_lagrage.png"
    filepath = f"Wykresy/{filename}"
    plt.savefig(filepath)
    plt.close()

def plot_cubic_spline_interpolation(num_points,data,x,y,chebyshev):
    b = len(x)
    if(chebyshev):
        indices = chebyshev_nodes(x,num_points)
    else:
        indices = normal_nodes(0,b-1,num_points)
    distance_nodes = [x[i] for i in indices]
    height_nodes = [y[i] for i in indices]
    distance_new = np.linspace(x.min(), x.max(), 500)
    height_new = cubic_spline_interpolation(distance_nodes, height_nodes, distance_new)
    plt.plot(x, y, '-', label='Dane oryginalne')
    plt.plot(distance_nodes, height_nodes, 's', label='Punkty węzłowe')
    if(chebyshev):
        plt.plot(distance_new, height_new, '-', label=f'Interpolacja funkcjami sklejanymi ({num_points} węzłów Czebyszewa)')
        plt.title(f'Interpolacja Funkcjami Sklejanymi - {data} ({num_points} węzłów Czebyszewa)')
    else:
        plt.plot(distance_new, height_new, '-', label=f'Interpolacja funkcjami sklejanymi ({num_points} punktów)')
        plt.title(f'Interpolacja Funkcjami Sklejanymi - {data} ({num_points} punktów)')
    plt.xlabel('Dystans (m)')
    plt.ylabel('Wysokość (m)')
    plt.legend()
    plt.grid(True)
    filename = f"{data}_{num_points}_{'chebyshev' if chebyshev else 'normal'}_cubic.png"
    filepath = f"Wykresy/{filename}"
    plt.savefig(filepath)
    plt.close()

x_everest,y_everest,x_canyon,y_canyon = load_data()

plot_lagrange_interpolation(15,"Mount Everest",x_everest,y_everest,True)
plot_lagrange_interpolation(15,"Mount Everest",x_everest,y_everest,False)
plot_lagrange_interpolation(30,"Mount Everest",x_everest,y_everest,True)
plot_lagrange_interpolation(30,"Mount Everest",x_everest,y_everest,False)
plot_lagrange_interpolation(45,"Mount Everest",x_everest,y_everest,True)
plot_lagrange_interpolation(45,"Mount Everest",x_everest,y_everest,False)
plot_lagrange_interpolation(60,"Mount Everest",x_everest,y_everest,True)
plot_lagrange_interpolation(60,"Mount Everest",x_everest,y_everest,False)

plot_lagrange_interpolation(15,"Grand Canyon",x_canyon,y_canyon,True)
plot_lagrange_interpolation(15,"Grand Canyon",x_canyon,y_canyon,False)
plot_lagrange_interpolation(30,"Grand Canyon",x_canyon,y_canyon,True)
plot_lagrange_interpolation(30,"Grand Canyon",x_canyon,y_canyon,False)
plot_lagrange_interpolation(45,"Grand Canyon",x_canyon,y_canyon,True)
plot_lagrange_interpolation(45,"Grand Canyon",x_canyon,y_canyon,False)
plot_lagrange_interpolation(60,"Grand Canyon",x_canyon,y_canyon,True)
plot_lagrange_interpolation(60,"Grand Canyon",x_canyon,y_canyon,False)

plot_cubic_spline_interpolation(15,"Mount Everest",x_everest,y_everest,True)
plot_cubic_spline_interpolation(15,"Mount Everest",x_everest,y_everest,False)
plot_cubic_spline_interpolation(30,"Mount Everest",x_everest,y_everest,True)
plot_cubic_spline_interpolation(30,"Mount Everest",x_everest,y_everest,False)
plot_cubic_spline_interpolation(45,"Mount Everest",x_everest,y_everest,True)
plot_cubic_spline_interpolation(45,"Mount Everest",x_everest,y_everest,False)
plot_cubic_spline_interpolation(60,"Mount Everest",x_everest,y_everest,True)
plot_cubic_spline_interpolation(60,"Mount Everest",x_everest,y_everest,False)

plot_cubic_spline_interpolation(15,"Grand Canyon",x_canyon,y_canyon,True)
plot_cubic_spline_interpolation(15,"Grand Canyon",x_canyon,y_canyon,False)
plot_cubic_spline_interpolation(30,"Grand Canyon",x_canyon,y_canyon,True)
plot_cubic_spline_interpolation(30,"Grand Canyon",x_canyon,y_canyon,False)
plot_cubic_spline_interpolation(45,"Grand Canyon",x_canyon,y_canyon,True)
plot_cubic_spline_interpolation(45,"Grand Canyon",x_canyon,y_canyon,False)
plot_cubic_spline_interpolation(60,"Grand Canyon",x_canyon,y_canyon,True)
plot_cubic_spline_interpolation(60,"Grand Canyon",x_canyon,y_canyon,False)
