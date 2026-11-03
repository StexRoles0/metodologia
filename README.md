# Prueba de concepto: DMD y DMDc en Puente Aranda (Bogotá)

Steven Navarro Parrales y Santiago Andrés Guarín Rojas. Universidad Distrital Francisco José de Caldas.

## Datos
Velocidades Bitcarrier, noviembre de 2022 (Secretaría Distrital de Movilidad, datos abiertos).
Se descarga el CSV completo desde el portal de datos abiertos de la SDM (dataset "Velocidades Bitcarrier Noviembre 2022").
El script usa 21 tramos alrededor de la intersección de Puente Aranda (lista `TRAMOS` en el código).

## Uso
```
pip install -r requirements.txt
python poc_dmdc.py ruta/Velocidades_Bitcarrier_Noviembre_2022.csv
```
Salida en `out/`: figuras (PNG) y `metricas.json`.

## Qué hace
1. Preprocesamiento: recorte a [1, 90] km/h, interpolación de huecos de hasta 1 h, mediana de la franja horaria en huecos largos, media móvil de 30 min.
2. Matriz X: 21 tramos × 2.880 instantes de 15 min, centrada con la media de entrenamiento (1–21 nov).
3. SVD, DMD y DMDc (Proctor, Brunton y Kutz, 2016) implementadas con NumPy; rangos elegidos con validación (19–21 nov).
4. Evaluación en 22–30 nov: RMSE y MAE por horizonte (15 min a 4 h) contra persistencia, perfil histórico y DMD sin control.
5. Contrafactual: domingo 27 nov simulado con la entrada "no laborable" en 0.
