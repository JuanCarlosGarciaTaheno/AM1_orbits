from numpy import array, zeros, linspace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Orbits import Arenstorf
from Cauchy_Problem import Cauchy_problem_RK4_emb
from Temporal_Schemes import ERK2


def create_animation_Arenstorf():
   #Initial Conditions
   U0 = array([0.994, 0, 0, -2.00158510637908252240537862224])
   #U0 = array([1.2,0,0,-1.049357510])
    #Parameters definition
   T = 17.0652165601579625588917206249
   N = 100000
   dt = T/N
   t = linspace(0,T,N)
   x = array(zeros(N))
   y = array(zeros(N))
   x[0] = U0[0]
   y[0] = U0[1]


   sol,h = Cauchy_problem_RK4_emb(t, ERK2, Arenstorf, U0)
   anim_fps = 60
   anim_duration = 3
   anim_save_name = 'Arenstorf_Animation.gif'
   # Funciones para crear la animacion:
   # - Funcion de inicializacion: se llama para crear la trama base vacia
   def init():
      line.set_data([], [])
      return line,
   # - Funcion de actualizacion para cada cuadro de la animacion
   def update(frame):
      x1 = sol[:frame, 0]  # Obtener datos hasta el cuadro actual
      y1 = sol[:frame, 1]
      line.set_data(x1, y1)
      return line,
 
   # Crear la figura usando plt.figure
   plt.figure(figsize=(8, 8))
   plt.tick_params(labelsize=13)
 
   # Crear una linea vacia que se actualizara en la animacion
   line, =  plt.plot([], [], '-',color='red')
 
   # Definir los parametros de la animaci�n
   anim_frames = linspace(0, N, int(anim_fps*anim_duration+1), dtype=int)
   anim_interval = (anim_duration / int(anim_fps*anim_duration+1)) * 1000
 
   # Definir los limites de los ejes de la grafica
   rangex = (max(sol[:,0]) - min(sol[:,0])) * 0.1
   rangey = (max(sol[:,1]) - min(sol[:,1])) * 0.1
   xlim = [min(sol[:,0]) - rangex, max(sol[:,0]) + rangex]
   ylim = [min(sol[:,1]) - rangey, max(sol[:,1]) + rangey]
 
   # Definicion de propiedades de la grafica
   plt.xlim(xlim)
   plt.ylim(ylim)
   plt.xlabel('x', fontdict = {'fontsize':14, 'fontweight':'normal', 'color':'k'})
   plt.ylabel('dx/dt', fontdict = {'fontsize':14, 'fontweight':'normal', 'color':'k'})
   plt.title('Arenstorf Orbit', fontdict = {'fontsize':18, 'fontweight':'normal', 'color':'k'})
   plt.plot(0,0,marker='o', color='b')
   plt.plot(1,0,marker='o', color='k')
 
   # Creacion y guardado de la animacion
   ani = FuncAnimation(plt.gcf(), update, frames=anim_frames, init_func=init, blit=True, interval=anim_interval)
   ani.save(anim_save_name, writer='Pillow', fps=anim_fps)
 
   plt.show()
 
