# BLT-PINNs

Documentación inicial basada en los archivos del repositorio, revisados el 22 de octubre de 2026. Versión científica inspeccionada: [`c32ab1b8ae115602b0b79e735030a21e8bf24273`](https://github.com/Ruben474/BLT-PINNs/tree/c32ab1b8ae115602b0b79e735030a21e8bf24273), commit «Se creó con Colab» del 16 de diciembre de 2025.

**Estado:** código y salidas guardadas inspeccionados; entrenamiento completo no reejecutado durante esta documentación. «Verificado» significa comprobado en el archivo, no validación independiente de sus conclusiones.

## Objetivo científico y alcance

El cuaderno principal explora redes neuronales informadas por la física (PINNs) para un problema estacionario de convección–difusión con perfil de velocidad tipo Couette. Incluye regularización mediante una referencia de diferencias finitas (FD), entrenamiento progresivo en el parámetro de difusión, refinamiento adaptativo por residuo (RAR) y variantes con aproximaciones asintóticas compuestas.

Aunque el archivo se llama `652025NavierStokes.ipynb`, el problema de la primera celda de código es una **EDP escalar de temperatura**, no el sistema completo de Navier–Stokes. El repositorio también contiene un experimento independiente de inferencia de estructuras de ARN; su relación con el estudio PINN no está documentada.

## Ecuación y datos de entrada verificados

En la celda de índice 1 del cuaderno principal:

$$yT_x=\varepsilon(T_{xx}+T_{yy}),\qquad (x,y)\in(0,1)^2.$$

Las condiciones son entrada caliente $T(0,y)=1$ y salida y paredes frías $T(1,y)=T(x,0)=T(x,1)=0$. En las dos esquinas de entrada hay incompatibilidad entre los datos de los lados; el solver FD da prioridad a la entrada. Esto importa al interpretar los errores de frontera.

| Parámetro | Valor de la celda 1 |
| --- | --- |
| Secuencia de difusión | `CURRICULUM_EPS = [1.0, 0.1, 0.06, 0.04]` |
| Malla FD | `121 × 121`, diferencias centrales, `scipy.sparse.linalg.spsolve` |
| Red | `[2, 64, 64, 64, 1]`, activación `Tanh` |
| Semilla | `1234` para Python, NumPy y PyTorch |
| Entrenamiento | Adam, tasa `1e-3`, 6000 épocas por difusión |
| Muestreo | 2000 puntos de residuo y 1000 de frontera por lote |
| RAR | 500 puntos añadidos desde 5000 candidatos cada 500 épocas |
| Pérdida | residuo + frontera + ajuste a FD interpolada bilinealmente |
| Peso FD | `0.5 * (1 - 0.5 * ep / EPOCHS_PER_EPS)` |

No requiere datos experimentales externos para esta ruta: genera malla, puntos y referencia FD. La misma red continúa entre etapas de difusión. Otras celdas redefinen modelos y parámetros, e incluyen perfiles $y^m$; no deben interpretarse como una configuración única.

## Estructura de archivos

| Archivo versionado | Contenido |
| --- | --- |
| [652025NavierStokes.ipynb](652025NavierStokes.ipynb) | 41 celdas; variantes Couette, FD, PINNs, aproximaciones compuestas, métricas y figuras guardadas |
| [ribonanzanet-3d-inference.ipynb](ribonanzanet-3d-inference.ipynb) | 16 celdas; inferencia de coordenadas de ARN con pesos externos de RibonanzaNet |
| `README.md` | Esta guía |

En la versión inspeccionada no hay un archivo de dependencias fijadas, conjunto de CSV de resultados ni pesos entrenados independientes. Las imágenes y registros embebidos en los cuadernos no sustituyen esos artefactos.

## Ejecución en Google Colab

[Abrir cuaderno Couette](https://colab.research.google.com/github/Ruben474/BLT-PINNs/blob/main/652025NavierStokes.ipynb).

1. Guarda una copia y selecciona un entorno Python con GPU si está disponible. La celda 1 admite también CPU.
2. Añade al inicio esta instalación orientativa, inferida de los imports; las versiones históricas no están fijadas:

   ```python
   %pip install torch numpy scipy matplotlib pandas mpmath
   ```

3. En un entorno limpio, ejecuta **solo la primera celda de código**, cuyo encabezado es `PINN Couette híbrido: PDE + BC + referencia FD (sin solución compuesta en la pérdida)`. Es el índice 1 contando todas las celdas desde cero. No es el contador de ejecución de Jupyter.
4. Conserva la secuencia completa de difusión para comparar con su salida histórica. Reducir épocas sirve para comprobar funcionamiento, pero cambia el experimento.
5. Descarga `/content/outputs/compare_2D_eps_*.png` y `compare_1D_eps_*.png`, y guarda los registros impresos. La variable `results` contiene tuplas `(eps, métricas, campo_PINN, campo_FD)`; esta celda no exporta esos campos a CSV ni guarda pesos.

**No ejecutar todas las celdas indiscriminadamente:** hay variantes que redefinen funciones, rutas y modelos. La celda 19 requiere CSV `T_FD_eps_{tag}.csv`, `T_PINN_eps_{tag}_couette_pinn.csv`, `T_PINN_eps_{tag}_couette_hybrid.csv` y `T_Composite_eps_{tag}.csv` en `/content/outputs`; no los genera la ruta mínima anterior. Hay que seleccionar y ejecutar las variantes productoras correspondientes antes de esa validación.

### Cuaderno independiente de ARN

[Abrir en Colab](https://colab.research.google.com/github/Ruben474/BLT-PINNs/blob/main/ribonanzanet-3d-inference.ipynb). **No funciona con el repositorio por sí solo:** sus rutas apuntan a Kaggle y usa `.cuda()` explícitamente.

| Entrada externa | Uso |
| --- | --- |
| `/kaggle/input/stanford-ribonanza-2-rna-folding-in-3-d/test_sequences.csv` | columnas `sequence` y `target_id`; secuencias tokenizadas como A/C/G/U |
| `/kaggle/input/ribonanzanet2d-final/Network.py` y sus dependencias | clase `RibonanzaNet` |
| `/kaggle/input/ribonanzanet2d-final/configs/pairwise.yaml` | configuración de arquitectura |
| `/kaggle/input/ribonanzanet-3d-finetune/RibonanzaNet-3D.pt` | pesos cargados |

Para trasladarlo a Colab se deben obtener esos recursos, corregir las rutas y disponer de GPU; además importa PyYAML y Plotly. Su disponibilidad y versiones siguen pendientes. Produce `submission.csv` con `ID`, `resname`, `resid` y cinco ternas `x_i,y_i,z_i`. La visualización usa `preds[7]`, por lo que requiere al menos ocho secuencias. No hay evaluación contra estructuras verdaderas ni métrica de precisión implementada. El intento de variar predicciones mediante `model.dropout.train()` no acredita incertidumbre: ese módulo tiene probabilidad cero y no se usa en el `forward` mostrado.

## Métricas de error

Para $e=T_{PINN}-T_{FD}$ en $M$ nodos, la celda 1 calcula:

$$\mathrm{L2}_{codigo}=\sqrt{M^{-1}\sum_i e_i^2},\quad L_\infty=\max_i|e_i|,\quad MAE=M^{-1}\sum_i|e_i|.$$

`L2` es RMSE discreto, no una norma integral sin normalización ni un error relativo. Las variantes `_core` excluyen una fila de nodos en cada borde (`[1:-1,1:-1]`). `P99` es el percentil 99 del error absoluto en toda la malla. La referencia es numérica y participa en el entrenamiento: estos errores miden acuerdo con FD, no error certificado frente a una solución exacta independiente.

## Resultados reproducibles: evidencia y protocolo

**Salidas históricas verificadas en la celda 1, no reproducidas de nuevo:**

| Difusión | RMSE (`L2`) | RMSE interior (`L2_core`) | P99 |
| --- | --- | --- | --- |
| 1.0 | 5.788e-02 | 5.155e-02 | 2.704e-01 |
| 0.1 | 2.109e-02 | 1.559e-02 | 6.698e-02 |
| 0.06 | 1.873e-02 | 1.414e-02 | 4.564e-02 |
| 0.04 | 1.560e-02 | 1.175e-02 | 4.026e-02 |

La celda 18 contiene promedios escritos manualmente para una gráfica; no constituye un cálculo reproducible de esos promedios. Las salidas guardadas pueden proceder de ejecuciones anteriores al último cambio de fuente.

Para una reproducción auditable: usar el commit indicado, ejecutar la ruta mínima en entorno limpio, registrar versiones (`%pip freeze`), dispositivo y duración, y exportar campos, métricas y pesos por difusión. Comparar con FD refinada en varias mallas; después repetir con varias semillas. Conservar configuración, malla y definición de núcleo para cada variante.

### Pendiente por completar

- Fijar versiones de dependencias y comprobar una ejecución limpia completa de cada experimento seleccionado.
- Separar variantes en cuadernos independientes y resolver las distintas fórmulas de solución compuesta antes de atribuirles exactitud.
- Publicar CSV, pesos y metadatos junto con estudio de convergencia de la referencia FD.
- Comparar PINN estándar e híbrida con condiciones, presupuesto y evaluación comunes, incluyendo coste del solver FD.
- Confirmar significado oficial de BLT, vínculo con un manuscrito y licencia; no se asigna una licencia por inferencia.
