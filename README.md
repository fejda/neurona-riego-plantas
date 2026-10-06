# Neurona artificial para el riego de plantas

Una sola neurona artificial (dos entradas, dos pesos, un sesgo y activación sigmoide) que decide si una planta necesita riego a partir de la **humedad del suelo** (0–100 %) y la **temperatura ambiental** (0–50 °C). Salida: `1` = regar, `0` = no regar. Los datos son didácticos y no representan una recomendación agronómica real.

## Ejecución

```bash
uv sync
uv run main.py
```

El script entrena la neurona, imprime pesos, sesgo, error y probabilidades, prueba cinco condiciones nuevas, ejecuta los ocho experimentos de épocas/tasa de aprendizaje y compara umbrales (0.4, 0.5, 0.6).

## Cómo funciona

1. **Normalización:** `X_normalizado = X / [100, 50]`. La misma `escala` se usa para los datos nuevos.
2. **Suma ponderada:** `z = X_normalizado @ pesos + sesgo`.
3. **Activación:** `sigmoide(z) = 1 / (1 + e^-z)`.
4. **Error:** error cuadrático medio (MSE); los gradientes pasan por la derivada de la sigmoide.
5. **Actualización:** `pesos -= tasa * gradiente_pesos`, `sesgo -= tasa * gradiente_sesgo`.
6. **Decisión:** `respuesta = (probabilidad >= 0.5).astype(int)`.

Los pesos se inicializan siempre con la misma semilla (42), así los experimentos solo cambian en el parámetro indicado.

## Resultado de la prueba base (10000 épocas, tasa 0.5)

| Parámetro | Valor |
|---|---|
| Peso humedad | -19.5201 |
| Peso temperatura | 2.1946 |
| Sesgo | 7.4780 |
| Error final (MSE) | 0.019539 |
| Aciertos | 10/10 |

| Caso | Humedad | Temp. | Probabilidad | Respuesta | Esperado |
|---|---|---|---|---|---|
| 1 | 80 % | 18 °C | 0.0006 | 0 | 0 |
| 2 | 70 % | 22 °C | 0.0054 | 0 | 0 |
| 3 | 65 % | 28 °C | 0.0183 | 0 | 0 |
| 4 | 55 % | 25 °C | 0.1033 | 0 | 0 |
| 5 | 50 % | 32 °C | 0.2937 | 0 | 0 |
| 6 | 40 % | 30 °C | 0.7284 | 1 | 1 |
| 7 | 35 % | 25 °C | 0.8511 | 1 | 1 |
| 8 | 30 % | 32 °C | 0.9538 | 1 | 1 |
| 9 | 20 % | 35 °C | 0.9940 | 1 | 1 |
| 10 | 10 % | 38 °C | 0.9992 | 1 | 1 |

**Signo de los pesos:** el peso de la humedad es negativo (a más humedad, menor probabilidad de regar). El peso de la temperatura es positivo (a más temperatura, mayor probabilidad de regar).

## Predicciones con condiciones nuevas

| Humedad | Temperatura | Probabilidad | Decisión |
|---|---|---|---|
| 75 % | 30 °C | 0.0029 | No regar (0) |
| 45 % | 34 °C | 0.5464 | Regar (1) |
| 25 % | 22 °C | 0.9724 | Regar (1) |
| 50 % | 25 °C | 0.2342 | No regar (0) |
| 30 % | 40 °C | 0.9670 | Regar (1) |

## Experimentos

| Experimento | Épocas | Tasa | Error final | Aciertos | P(45 %, 34 °C) | Aprendizaje |
|---|---|---|---|---|---|---|
| Prueba base | 10000 | 0.5 | 0.019539 | 10/10 | 0.5464 | Adecuado |
| Pocas épocas | 100 | 0.5 | 0.169361 | 10/10 | 0.5138 | Insuficiente |
| Cantidad intermedia | 1000 | 0.5 | 0.066970 | 10/10 | 0.5739 | Aceptable, aún mejorando |
| Más épocas | 20000 | 0.5 | 0.011669 | 10/10 | 0.5246 | Mejora pequeña |
| Tasa pequeña | 10000 | 0.01 | 0.131320 | 10/10 | 0.5318 | Lento |
| Tasa moderada | 10000 | 0.1 | 0.049284 | 10/10 | 0.5804 | Algo lento |
| Tasa alta | 10000 | 1.0 | 0.011669 | 10/10 | 0.5246 | Rápido y estable |
| Tasa muy alta | 10000 | 2.0 | 0.006556 | 10/10 | 0.5070 | Muy rápido, estable en estos datos |

Nota: con 100 épocas la neurona clasifica bien los diez casos, pero casi todas las probabilidades están cerca de 0.5 (entre 0.31 y 0.68), así que está poco segura. Con tasa 1.0 y 10000 épocas el error coincide con el de tasa 0.5 y 20000 épocas porque el avance total depende aproximadamente de tasa × épocas.

## Prueba con umbrales (sin reentrenar)

Con el modelo base, los diez datos de entrenamiento dan la misma respuesta con 0.4, 0.5 y 0.6 (la probabilidad más cercana, 0.2937 y 0.7284, no cruza ningún umbral). Entre las condiciones nuevas solo cambia **45 % / 34 °C** (probabilidad 0.5464): con 0.4 y 0.5 es *Regar* y con 0.6 es *No regar*. Cambiar el umbral solo modifica cómo se interpreta la probabilidad; los pesos y el sesgo ya fueron calculados durante el entrenamiento y el umbral no interviene en él.

## Análisis

1. **¿Por qué normalizar?** La humedad llega hasta 100 y la temperatura hasta ~50. Sin normalizar, la columna de mayor escala domina el gradiente, `z` toma valores grandes, la sigmoide se satura y el entrenamiento se vuelve inestable o lento. Normalizar deja ambas entradas en un rango parecido (0–1) y hace comparables sus pesos.
2. **Uso de `X_normalizado` y de `X`:** `X_normalizado` se usa en la suma ponderada (`z = X_normalizado @ pesos + sesgo`) y en el gradiente de los pesos (`X_normalizado.T @ gradiente_z`). `X` se conserva para mostrar los valores originales (%, °C) y facilitar la interpretación de los resultados.
3. **Solo 100 épocas:** el error final fue 0.169 y las probabilidades quedaron cerca de 0.5. Clasificó 10/10, pero con muy poca confianza: el aprendizaje fue insuficiente.
4. **¿Más épocas siempre mejoran?** No de forma importante. De 100 a 1000 el error cae de 0.169 a 0.067, y de 1000 a 10000 a 0.020, pero de 10000 a 20000 solo baja de 0.020 a 0.012. Los aciertos ya eran 10/10 desde 100 épocas. Hay rendimientos decrecientes.
5. **Tasa demasiado pequeña (0.01):** el aprendizaje es lento; tras 10000 épocas el error sigue en 0.131, muy por encima del de la prueba base.
6. **Tasa alta o muy alta (1.0 y 2.0):** en estos datos no hubo inestabilidad: el error bajó más rápido (0.0117 y 0.0066). Esto ocurre porque los datos son pocos, están normalizados y el gradiente se promedia. Con tasas todavía mayores o con otros datos podría oscilar o diverger, y una tasa alta puede volver las probabilidades demasiado extremas (sobreconfianza).
7. **Signo del peso de la humedad (negativo):** más humedad empuja `z` hacia abajo y reduce la probabilidad de regar; menos humedad la aumenta.
8. **Signo del peso de la temperatura (positivo):** más temperatura empuja `z` hacia arriba y aumenta la probabilidad de regar.
9. **¿Por qué un umbral?** La sigmoide devuelve un valor continuo entre 0 y 1, pero la acción es binaria (regar o no). El umbral (0.5 por defecto) convierte la probabilidad en una decisión; puede ajustarse según el costo de equivocarse.
10. **Limitaciones:** solo usa dos variables (ignora especie, tipo de suelo, lluvia, hora, sol, viento); solo tiene 10 datos didácticos y no hay conjunto de prueba; es un modelo lineal (frontera de decisión recta) y no captura relaciones más complejas; las etiquetas son inventadas y no provienen de criterio agronómico; y no está validado con datos reales.

## Archivos

- `main.py`: desarrollo de la neurona, experimentos y umbrales.
- `pyproject.toml` y `uv.lock`: dependencias gestionadas con uv.
- `README.md`: este documento.
