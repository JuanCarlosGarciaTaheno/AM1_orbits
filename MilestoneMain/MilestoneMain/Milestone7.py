print( "" )
print( "   ==========================================================================" )
print( "   ==             MASTER UNIVERSITARIO EN SISTEMAS ESPACIALES              ==" )
print( "   ==                    AMPLIACION DE MATEMATICAS 1                       ==" )
print( "   ==                       Rodrigo Torrico Gijon                          ==" )
print( "   ==                  Juan Carlos Garcia Taheno-Hijes                     ==" )
print( "   ==                       Alejandro Engel Kurson                         ==" )
print( "   ==                    Inigo Javier Palacios Martinez                    ==" )
print( "   ==========================================================================" )
print( "" )
print( "   ==========================================================================" )
print( "   =========================    ARENSTORF ORBIT   ===========================" )
print( "   ==========================================================================" )
print( "" )
print( "   Seleccione que desea ejecutar:" )
print( "" )
print( "     1 - GUI")
print( "     2 - CPU Times functions")
print( "     3 - Stepsize Variation")
print( "     4 - Arenstorf Animation")
print( "" )
tarea = int(input("   Introduzca el numero deseado: "))
print( "" )

from CPU_Times import CPU_Times
from Stepsize_Variation import Stepsize_Variation
from Arenstorf_Animation import create_animation_Arenstorf

from Cauchy_Problem import Cauchy_Problem, Cauchy_problem_RK4_emb
from Temporal_Schemes import RK4, GBS_solution_NL, Leap_Frog, ERK, ERK2, Euler, Crank_Nicolson

from Orbits import Arenstorf


if tarea == 1:
    from GUI import PlotApp
    from tkinter import Tk
    root = Tk()
    app = PlotApp(root)
    root.mainloop()
    
elif tarea == 2:
    
    CPU_Times()
    
elif tarea == 3:
    
    Stepsize_Variation()
    
elif tarea == 4:
    
    create_animation_Arenstorf()
    
else:
    
    exit