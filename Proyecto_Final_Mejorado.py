import random
import math
import matplotlib.pyplot as plt
import numpy as np
#Funcion para generar la semilla que servira para la generacion de la poblacion inicial
def establecer_semilla(seed):
    random.seed(seed)
    np.random.seed(seed)

#funcion para delimitar los valores de las variables dentro de un rango especifico, con el fin que no exceda las restricciones de caja
def clip(x, min_val, max_val):
    return max(min_val, min(max_val, x))

#funcion la cual evalua la aptitud de un individuo basado en las variables x1, x2, x3 y x4
def EvaluarAptitud(x):
    x1, x2, x3, x4 = x
    return 1.10471 * (x1 ** 2) * x2 + 0.04811 * x3 * x4 * (14.0 + x2)

#calculo del valor de teta prima del problema
def Teta_prima(x1, x2):
    return 6000 / (math.sqrt(2) * (x1 * x2))
#calculo del valor de M del problema
def M(x2):
    return 6000 * (14 + (x2 / 2))
#calculo del valor de R del problema
def R(x2, x1, x3):
    return math.sqrt((x2 ** 2) / 4 + ((x1 + x3) / 2) ** 2)

#calculo del valor de J del problema
def J(x1, x2, x3):
    return 2 * ((math.sqrt(2) * x1 * x2) * ((x2 ** 2) / 12 + ((x1 + x3) / 2) ** 2))

#calculo del valor de teta biprima del problema
def Teta_biprima(M_, R_, J_):
    return (M_ * R_) / J_

#calculo del valor de teta total del problema
def Teta_total(teta_p, teta_bp, R_, x2):
    return math.sqrt((teta_p ** 2) + 2 * teta_p * teta_bp * (x2 / (2 * R_)) + (teta_bp ** 2))

#calculo del valor de sigma del problema
def sigma(x4, x3):
    return 504000 / (x4 * (x3 ** 2))

#calculo del valor de delta del problema
def delta(x3, x4):
    return 2.1952 / ((x3 ** 3) * x4)

#calculo del valor de Pc del problema
def P(x3, x4):
    return 64746.022 * (1 - 0.0282346 * x3) * x3 * (x4 ** 3)

#funcion para validar que el individuo cumple con las restricciones del problema
def verificar_restricciones(x):
    x1, x2, x3, x4 = x
    try:
        apt = EvaluarAptitud(x)
        if apt < 0 :
            return False  
        if not (0.1 <= x1 <= 2):
            return False
        if not (0.1 <= x2 <= 10):
            return False 
        if not (0.1 <= x3 <= 10):
            return False
        if not (0.1 <= x4 <= 2):
            return False
             
        #calculo de los parametros con las funciones definidas
        tp = Teta_prima(x1, x2)
        m = M(x2)
        r = R(x2, x1, x3)
        j = J(x1, x2, x3)
        tb = Teta_biprima(m, r, j)
        t = Teta_total(tp, tb, r, x2)
        s = sigma(x4, x3)
        d = delta(x3, x4)
        p = P(x3, x4)
        #validacion de las restricciones
        if t <= 13000:
            r1 = 1
        if s <= 30000:
            r2 = 1
        if x1 <= x4:
            r3 = 1 
        if (0.1047 * (x1**2) + 0.04811 * x3 * x4 * (14 + x2)) <= 5:
            r4 = 1
        if x1 >= 0.125:
            r5 = 1 
        if d <= 0.25:
            r6 = 1
        if p >= 6000:
            r7 = 1
        # Se suman los resultados de las restricciones, el total debe ser 7 para cumplir todas las restricciones
        resultados = [r1, r2, r3, r4, r5, r6, r7]
        return sum(resultados) == 7  
    except:
        return False

#generacion de todos los vectores necesarios para el algoritmo GWO
def vector_a(max_iteraciones, iteracion_actual):
    return 2 - ((2 * iteracion_actual) / max_iteraciones)

def generar_vector():
    return [round(random.uniform(0, 1), 4) for _ in range(4)]

def vector_A(a):
    r1 = generar_vector()
    return [(2 * a * r1[i]) - a for i in range(4)]

def vector_C():
    r2 = generar_vector()
    return [2 * r2[i] for i in range(4)]

def calcular_D(C, Xa, x):
    return [abs(C[i] * Xa[i] - x[i]) for i in range(4)]

def calcular_X(Xa, A, D_):
    return [Xa[i] - A[i] * D_[i] for i in range(4)]

#actualizacion de la posicion del individuo basado en los valores de los lobos alfa, beta y delta
def actualiza_paso(X1, X2, X3):
    return [(X1[i] + X2[i] + X3[i]) / 3 for i in range(4)]

#generacion de la poblacion inicial, asegurando que cada individuo cumple con las restricciones del problema
def generar_poblacion(tam):
    poblacion = []
    while len(poblacion) < tam:
        ind = [random.uniform(0.1, 15) for _ in range(4)]
        if verificar_restricciones(ind):
            poblacion.append(ind)
    return poblacion

# Definicion de la jerarquia de los lobos, obteniendo los mejores individuos de la poblacion
def Definir_Jerarquia(poblacion):
    evaluadas = [(ind, EvaluarAptitud(ind)) for ind in poblacion]
    ordenadas = sorted(evaluadas, key=lambda x: x[1])
    return ordenadas[0][0], ordenadas[1][0], ordenadas[2][0], ordenadas[:5]

#flujo completo del algoritmo GWO
def GWO(tam_poblacion, iteraciones):
    poblacion = generar_poblacion(tam_poblacion)
    historial_aptitud_alfa = []

    mejor_global = None
    mejor_aptitud_global = float('inf')
    #repeticion del proceso de optimizacion por el numero de iteraciones
    for i in range(iteraciones):
        a = vector_a(iteraciones, i)
        alfa, beta, delta, _ = Definir_Jerarquia(poblacion)
        lideres = [alfa, beta, delta]
        aptitud_alfa = EvaluarAptitud(alfa)
        historial_aptitud_alfa.append(aptitud_alfa)

        if aptitud_alfa < mejor_aptitud_global:
            mejor_global = alfa.copy()
            mejor_aptitud_global = aptitud_alfa

        nueva_poblacion = []
        #actualizacon de cada individuo de la poblacion
        for j in range(tam_poblacion):
            actual = poblacion[j]
            #actualizacion de los individuos que no son lideres
            if actual not in lideres:
                A1, A2, A3 = vector_A(a), vector_A(a), vector_A(a)
                C1, C2, C3 = vector_C(), vector_C(), vector_C()

                D1 = calcular_D(C1, alfa, actual)
                D2 = calcular_D(C2, beta, actual)
                D3 = calcular_D(C3, delta, actual)

                X1 = calcular_X(alfa, A1, D1)
                X2 = calcular_X(beta, A2, D2)
                X3 = calcular_X(delta, A3, D3)

                nuevo = actualiza_paso(X1, X2, X3)
                # Perturbación aleatoria para evitar convergencia prematura
                perturbacion = [(random.uniform(-0.01, 0.01)) for _ in range(4)]
                nuevo = [nuevo[k] + perturbacion[k] for k in range(4)]
                # Asegurar que el nuevo individuo cumple con las restricciones de caja
                nuevo = [
                    clip(nuevo[0], 0.1, 2),
                    clip(nuevo[1], 0.1, 10),
                    clip(nuevo[2], 0.1, 10),
                    clip(nuevo[3], 0.1, 2),
                ]
                # Verificar restricciones antes de añadir a la nueva población
                if verificar_restricciones(nuevo):
                    nueva_poblacion.append(nuevo)
                else:
                    # Si no cumple restricciones, mantener el individuo actual
                    nueva_poblacion.append(actual)  
            else:
                nueva_poblacion.append(actual)  

        poblacion = nueva_poblacion

        print(f"Iteración {i+1}: Alfa = {alfa}, Aptitud = {aptitud_alfa:.4f}")

    # Reporta mejor solución global
    print("Mejor solución global:", mejor_global, "con aptitud:", mejor_aptitud_global)

    plt.figure()
    plt.plot(range(len(historial_aptitud_alfa)), historial_aptitud_alfa, marker='o', linestyle='-')
    plt.title("Convergencia de la aptitud del lobo alfa")
    plt.xlabel("Iteración")
    plt.ylabel("Aptitud de alfa")
    plt.grid(True)
    plt.show()


def main():
    #Configuracion de la semilla y parametros del algoritmo GWO
    seed = 3
    establecer_semilla(seed)
    poblacion = 50
    max_iteraciones = 500
    GWO(poblacion, max_iteraciones)

if __name__ == "__main__":
    main()
