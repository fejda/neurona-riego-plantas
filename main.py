"""Neurona artificial para el riego de plantas.

Una sola neurona con dos entradas (humedad y temperatura), dos pesos,
un sesgo y activación sigmoide. Salida binaria: 1 = regar, 0 = no regar.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Datos de entrenamiento
# ---------------------------------------------------------------------------
X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

# ---------------------------------------------------------------------------
# Normalización: se divide cada columna por su valor máximo esperado.
# Esta misma escala se usa para los datos nuevos.
# ---------------------------------------------------------------------------
escala = np.array([100, 50])
X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)


def sigmoide(z):
    """Convierte z en una probabilidad entre 0 y 1."""
    return 1 / (1 + np.exp(-z))


def entrenar(tasa_aprendizaje, epocas, mostrar=False, semilla=42):
    """Entrena la neurona y devuelve (pesos, sesgo, error_final).

    Siempre se parte de los mismos valores iniciales (misma semilla) para
    que los experimentos solo difieran en el parámetro que se modifica.
    """
    rng = np.random.default_rng(semilla)
    pesos = rng.normal(0, 0.1, size=(2, 1))  # peso humedad y peso temperatura
    sesgo = 0.0
    n = len(X_normalizado)
    error_final = None

    for epoca in range(1, epocas + 1):
        # Suma ponderada y activación
        z = X_normalizado @ pesos + sesgo
        probabilidad = sigmoide(z)

        # Error y error cuadrático medio
        error = probabilidad - y
        error_final = float(np.mean(error ** 2))

        # Gradientes (derivada de MSE a través de la sigmoide)
        gradiente_z = error * probabilidad * (1 - probabilidad) * 2 / n
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = float(np.sum(gradiente_z))

        # Actualización
        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if mostrar and (epoca % max(1, epocas // 10) == 0 or epoca == 1):
            print(f"Época {epoca:>6} | Error cuadrático medio: {error_final:.6f}")

    # Error final con los parámetros definitivos
    probabilidad = sigmoide(X_normalizado @ pesos + sesgo)
    error_final = float(np.mean((probabilidad - y) ** 2))
    return pesos, sesgo, error_final


def predecir(humedad_temperatura, pesos, sesgo):
    """Probabilidad de regar. Normaliza con la MISMA escala del entrenamiento."""
    datos = np.atleast_2d(np.array(humedad_temperatura, dtype=float)) / escala
    return sigmoide(datos @ pesos + sesgo)


def a_respuesta(probabilidad, umbral=0.5):
    return (probabilidad >= umbral).astype(int)


# ---------------------------------------------------------------------------
# Configuración inicial (prueba base)
# ---------------------------------------------------------------------------
tasa_aprendizaje = 0.5
epocas = 10000

print("\n=== Entrenamiento base ===")
pesos, sesgo, error_final = entrenar(tasa_aprendizaje, epocas, mostrar=True)
probabilidades = predecir(X, pesos, sesgo)
respuestas = a_respuesta(probabilidades)

print(f"\nPeso de la humedad:     {pesos[0, 0]:.4f}")
print(f"Peso de la temperatura: {pesos[1, 0]:.4f}")
print(f"Sesgo:                  {sesgo:.4f}")
print(f"Error final (MSE):      {error_final:.6f}")

print("\nProbabilidad por caso:")
print("Caso | Humedad | Temp | Probabilidad | Respuesta | Esperado")
for i in range(len(X)):
    print(f"{i + 1:>4} | {X[i, 0]:>6.0f}% | {X[i, 1]:>3.0f}°C | "
          f"{probabilidades[i, 0]:>12.4f} | {respuestas[i, 0]:>9} | {int(y[i, 0]):>8}")

if pesos[0, 0] < 0:
    print("\nEl peso de la humedad es negativo: más humedad -> menos probabilidad de regar.")
if pesos[1, 0] > 0:
    print("El peso de la temperatura es positivo: más temperatura -> más probabilidad de regar.")

# ---------------------------------------------------------------------------
# Pruebas con condiciones nuevas
# ---------------------------------------------------------------------------
nuevos = np.array([[75, 30], [45, 34], [25, 22], [50, 25], [30, 40]], dtype=float)
prob_nuevos = predecir(nuevos, pesos, sesgo)
resp_nuevos = a_respuesta(prob_nuevos)

print("\n=== Predicciones con condiciones nuevas ===")
print("Humedad | Temperatura | Probabilidad | Decisión")
for i in range(len(nuevos)):
    decision = "Regar" if resp_nuevos[i, 0] == 1 else "No regar"
    print(f"{nuevos[i, 0]:>6.0f}% | {nuevos[i, 1]:>9.0f}°C | "
          f"{prob_nuevos[i, 0]:>12.4f} | {decision} ({resp_nuevos[i, 0]})")

# ---------------------------------------------------------------------------
# Experimentos con épocas y tasas de aprendizaje
# ---------------------------------------------------------------------------
experimentos = [
    ("Prueba base", 10000, 0.5),
    ("Pocas épocas", 100, 0.5),
    ("Cantidad intermedia", 1000, 0.5),
    ("Más épocas", 20000, 0.5),
    ("Tasa pequeña", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0),
]

print("\n=== Experimentos ===")
print(f"{'Experimento':<20} {'Épocas':>7} {'Tasa':>5} {'Error final':>12} "
      f"{'Aciertos':>9} {'P(45%,34°C)':>12}")
for nombre, ep, tasa in experimentos:
    p, s, err = entrenar(tasa, ep)
    aciertos = int(np.sum(a_respuesta(predecir(X, p, s)) == y))
    prob_45_34 = predecir([45, 34], p, s)[0, 0]
    print(f"{nombre:<20} {ep:>7} {tasa:>5} {err:>12.6f} {aciertos:>7}/10 {prob_45_34:>12.4f}")

# ---------------------------------------------------------------------------
# Prueba adicional con umbrales (sin volver a entrenar)
# ---------------------------------------------------------------------------
print("\n=== Umbrales 0.4 / 0.5 / 0.6 (sin reentrenar, modelo base) ===")
print("Datos de entrenamiento:")
print("Caso | Probabilidad | U=0.4 | U=0.5 | U=0.6 | Esperado")
for i in range(len(X)):
    pr = probabilidades[i, 0]
    print(f"{i + 1:>4} | {pr:>12.4f} | {int(pr >= 0.4):>5} | {int(pr >= 0.5):>5} | "
          f"{int(pr >= 0.6):>5} | {int(y[i, 0]):>8}")

print("\nCondiciones nuevas:")
print("Humedad | Temp | Probabilidad | U=0.4 | U=0.5 | U=0.6")
for i in range(len(nuevos)):
    pr = prob_nuevos[i, 0]
    print(f"{nuevos[i, 0]:>6.0f}% | {nuevos[i, 1]:>3.0f}°C | {pr:>12.4f} | "
          f"{int(pr >= 0.4):>5} | {int(pr >= 0.5):>5} | {int(pr >= 0.6):>5}")
