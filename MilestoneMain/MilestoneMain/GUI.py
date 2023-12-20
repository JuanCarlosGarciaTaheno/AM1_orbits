from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numpy import array, linspace
import tkinter as tk
from tkinter import ttk, Entry, StringVar
import matplotlib.pyplot as plt
from ModulosM7.Temporal_schemes import Euler, Inverse_Euler, Crank_Nicolson, RK4, adaptive_RK_emb,Leapfrog,GBS,leap_frog
from ModulosM7.Cauchy_problem import Cauchy_problem, Cauchy_problem_LP, Cauchy_Problem_GBS, Cauchy_problem_RK4_emb
from ModulosM7.ODE_Arenstorf import Arenstorf

class PlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Orbit of Arenstorf for different methods')

        self.function_label1 = tk.Label(root, text='Select the first numerical method:')
        self.function_label1.grid(row=0, column=0, padx=5, pady=5)

        self.function_combobox1 = ttk.Combobox(root, values=['Runge-Kutta 4', 'Euler', 'Crank Nicolson', 'Leap Frog', 'GBS', 'RK4 Multi-Step'])
        self.function_combobox1.grid(row=1, column=0, padx=5, pady=5)

        self.function_label2 = tk.Label(root, text='Select the second numerical method:')
        self.function_label2.grid(row=2, column=0, padx=5, pady=5)

        self.function_combobox2 = ttk.Combobox(root, values=['Runge-Kutta 4', 'Euler', 'Crank Nicolson', 'Leap Frog', 'GBS', 'RK4 Multi-Step'])
        self.function_combobox2.grid(row=3, column=0, padx=5, pady=5)

        self.nl_label = tk.Label(root, text='Number of levels (NL) for GBS:')
        self.nl_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        self.nl_entry_var1 = StringVar()
        self.nl_entry1 = Entry(root, textvariable=self.nl_entry_var1, state='disabled')
        self.nl_entry1.grid(row=1, column=1, padx=5, pady=5)

        self.nl_entry_var2 = StringVar()
        self.nl_entry2 = Entry(root, textvariable=self.nl_entry_var2, state='disabled')
        self.nl_entry2.grid(row=3, column=1, padx=5, pady=5)

        self.plot_button = tk.Button(root, text='Plot', command=self.plot_function,height=6, width=10)
        self.plot_button.grid(row=4, column=0, columnspan=2, pady=20)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.grid(row=0, column=2, rowspan=5, padx=10)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack()

        # Asocia el metodo de actualizacion del Entry con la seleccion del metodo
        self.function_combobox1.bind("<<ComboboxSelected>>", lambda _: self.update_nl_entry_state(1))
        self.function_combobox2.bind("<<ComboboxSelected>>", lambda _: self.update_nl_entry_state(2))

    def plot_function(self):
        function1 = self.function_combobox1.get()
        function2 = self.function_combobox2.get()

        #U0 = array([0.994, 0, 0, -2.0015851063798025664053786222])
        U0 = array([1.2, 0, 0, -1.049357510])
        time_domain = linspace(0, 17.06521656015796, 100000)

        # Limpiar los ejes antes de cada nueva plot
        self.ax1.clear()
        self.ax2.clear()

        for i, function in enumerate([function1, function2]):
            if function == "GBS":
                try:
                    NL = int(self.nl_entry_var1.get()) if i == 0 else int(self.nl_entry_var2.get())
                except ValueError:
                    print("Invalid value for NL. Using default value.")
                    NL = 1

                U = Cauchy_Problem_GBS(time_domain, self.get_method(function), Arenstorf, U0, NL)
            elif function == "RK4 Multi-Step":
                U,h = Cauchy_problem_RK4_emb( Arenstorf, time_domain, U0, self.get_method(function))
            elif function == "Leap Frog":
                U = Cauchy_problem_LP(Arenstorf, time_domain, U0, self.get_method(function))
            else:
                U = Cauchy_problem(Arenstorf,time_domain, U0, self.get_method(function))

            ax = (self.ax1, self.ax2)[i]
            ax.plot(U[:, 0], U[:, 1],'r', label=function)
            ax.set_title(f'Orbit of Arenstorf with {function}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.plot(0,0,'bo', label = 'Earth')
            ax.plot(1,0,'ko', label = 'Moon')
            ax.axis('equal')
            ax.legend()

        # Actualizar el lienzo de la figura
        self.canvas.draw()

    def get_method(self, method_name):
        if method_name == "Runge-Kutta 4":
            return RK4
        elif method_name == "Euler":
            return Euler
        elif method_name == "Crank Nicolson":
            return Crank_Nicolson
        elif method_name == "Leap Frog":
            return Leapfrog
        elif method_name == "GBS":
            return GBS
        elif method_name == "RK4 Multi-Step":
            return adaptive_RK_emb
        else:
            return None

    def update_nl_entry_state(self, entry_number):
        selected_function = self.function_combobox1.get() if entry_number == 1 else self.function_combobox2.get()
        entry_var = self.nl_entry_var1 if entry_number == 1 else self.nl_entry_var2
        nl_entry = self.nl_entry1 if entry_number == 1 else self.nl_entry2

        if selected_function == "GBS":
            nl_entry.config(state='normal')
        else:
            nl_entry.config(state='disabled')
            entry_var.set("")  # Limpiar el contenido si no es GBS


root = tk.Tk()
app = PlotApp(root)
root.mainloop()