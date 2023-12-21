from numpy import array, zeros, linspace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Orbits import Arenstorf
from Cauchy_Problem import Cauchy_Problem, Cauchy_problem_RK4_emb
from Temporal_Schemes import  ERK2

#----------------------------------------------------------
def Stepsize_Variation():
    # Initial Conditions
    U0 = array([0.994, 0, 0, -2.00158510637908252240537862224])

    # Parameters definition
    T = 17.0652165601579625588917206249
    N = 100000
    dt = T/N
    t = linspace(0, T, N)
    x = array(zeros(N))
    y = array(zeros(N))
    x[0] = U0[0]
    y[0] = U0[1]

    sol, h = Cauchy_problem_RK4_emb(t, ERK2, Arenstorf, U0)
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #create_animation_stepsize(h,t,60,4,'Sepsize_Variation.gif',N)
    anim_save_name1 = 'Sepsize_Variation.gif'
    anim_duration = 4
    anim_fps = 60

    # Functions to create the animation:
    # - Initialization function: called to create the empty base frame
    def init():
        line.set_data([], [])
        return line,
    # - Update function for each frame of the animation
    def update(frame):
        x1 = t[:frame]  # Obtener datos hasta el cuadro actual
        y1 = h[:frame]
        line.set_data(x1, y1)
        return line,

    # Create the figure using plt.figure
    plt.figure(figsize=(8, 8))
    plt.tick_params(labelsize=13)

    # Create an empty line that will be updated in the animation
    line, = plt.plot([], [], '-', color='red')

    # Define the parameters of the animation
    anim_frames = linspace(0, N, int(anim_fps*anim_duration+1), dtype=int)
    anim_interval = (anim_duration / int(anim_fps*anim_duration+1)) * 1000

    # Define the limits of the graph axes
    rangex = (max(t[:]) - min(t[:])) * 0.1
    rangey = (max(h[:]) - min(h[:])) * 0.1
    xlim = [min(t[:]) - rangex, max(t[:]) + rangex]
    ylim = [min(h[:]) - rangey, max(h[:]) + rangey]

    # Define properties of the graph
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('Time', fontdict={'fontsize': 14, 'fontweight': 'normal', 'color': 'k'})
    plt.ylabel('Step Size', fontdict={'fontsize': 14, 'fontweight': 'normal', 'color': 'k'})
    plt.title('Adaptive Step Size over Time', fontdict={'fontsize': 18, 'fontweight': 'normal', 'color': 'k'})
    # Create and save the animation
    ani = FuncAnimation(plt.gcf(), update, frames=anim_frames, init_func=init, blit=True, interval=anim_interval)
    ani.save(anim_save_name1, writer='Pillow', fps=anim_fps)

    plt.show()


