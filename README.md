# AM1_orbits
Juan Carlos García-Taheño Hijes

Este proyecto se ha desarrollado en seis milestones, cada uno abordando aspectos específicos del estudio de órbitas y problemas astrofísicos utilizando métodos numéricos. 
Antes de empezar a abordarlos es conveniente presentar la estructura que se va encontrar en este repositorio:
  Se encontraran carpetas con milestones individuales; todas ellas estan expuestas públicamente para mostrar el desarrollo de este repositorio y no influirán en el código principal.
  Carpeta "MilestoneMain": en este carpeta se encuentran los archivos principales para desarrollar todos los milestone, en ella aparece "MilestoneMain.py" archivo que controlará que ejercicio se resolverá introduciendo en pantalla el número deseado del problema. El milestone7 posee una particularidad, ya que se ha realizado en grupo, por lo que los módulos asociados a este serán indendientes al resto.


A continuación, se presenta un resumen de los hitos alcanzados:

Milestone 1: Prototipos para Integrar Órbitas sin Funciones
En este hito, se implementaron scripts para integrar órbitas de Kepler utilizando diferentes métodos numéricos, como Euler, Crank-Nicolson y Runge-Kutta de cuarto orden. Además, se exploraron variaciones en el paso del tiempo y se discutieron los resultados obtenidos.

Milestone 2: Prototipos para Integrar Órbitas con Funciones
En este milestone, se llevaron a cabo funciones más avanzadas para la integración de órbitas, incluyendo métodos como Euler, Crank-Nicolson y RK4. Se implementó una función para expresar la fuerza en el movimiento de Kepler y se realizaron integraciones con diferentes esquemas, analizando los resultados obtenidos.

Milestone 3: Estimación del Error de Soluciones Numéricas
Este hito se centró en la evaluación de errores en las soluciones numéricas mediante la extrapolación de Richardson. Se calcularon errores para diferentes esquemas temporales y se evaluó la tasa de convergencia con respecto al paso del tiempo.

Milestone 4: Problemas Lineales. Regiones de Estabilidad Absoluta
Se integró el oscilador lineal con varios métodos y se exploraron las regiones de estabilidad absoluta de estos métodos. Los resultados numéricos se explicaron en función de estas regiones de estabilidad.

Milestone 5: Problema N-Cuerpos
En este hito, se desarrolló una función para integrar el problema N-cuerpos y se simuló un ejemplo para analizar los resultados obtenidos.

Milestone 6: Puntos de Lagrange y su Estabilidad
El último milestone abordó la implementación de un método de Runge-Kutta de alto orden y la simulación del problema del cuerpo restringido circular tres-cuerpos. Se determinaron y analizaron los puntos de Lagrange, así como la estabilidad de estos puntos y las órbitas alrededor de ellos.

Este informe proporciona una visión general de los logros alcanzados en cada milestone, destacando la progresión y el enfoque en el estudio de la dinámica celeste mediante métodos numéricos.

Milestone 7: Órbita Periódica de Arenstorf
Descripción
Este milestone se enfoca en la integración de la órbita periódica de Arenstorf, un problema celestial complejo que implica la interacción entre dos cuerpos celestiales y un tercer cuerpo más pequeño en órbita alrededor de uno de ellos. El objetivo principal es comparar los resultados obtenidos mediante el método de Gauss-Bulirsch-Stoer (GBS), un Runge-Kutta (RK) embebido y evaluar los tiempos de computación asociados.

Tareas
Implementación de Órbita Periódica de Arenstorf:

Desarrollar una función para integrar la órbita periódica de Arenstorf utilizando métodos numéricos.
Comparación entre GBS y RK Embebido:

Utilizar el método de Gauss-Bulirsch-Stoer y un Runge-Kutta embebido para integrar la órbita de Arenstorf.
Comparar los resultados obtenidos por ambos métodos en términos de precisión y eficiencia.
Evaluación de Tiempos de Computación:

Medir y comparar los tiempos de computación asociados con la integración de la órbita de Arenstorf utilizando GBS y RK embebido.
Análisis de Resultados:

Analizar y discutir los resultados obtenidos, destacando las diferencias en la precisión y eficiencia de los métodos utilizados.

<p align="center">
  <img src=".\Organizacion_de_Modulos.png" alt="Organizacion del Milestone7">
</p>
