# Estimación de parámetros cinéticos de *E. coli* productora de citramalato mediante redes generadoras y estrategias evolutivas

Código del Trabajo Fin de Grado. Se entrena una red neuronal generadora de parámetros cinéticos para el modelo de *E. coli* de Millard et al., ampliado con la reacción de síntesis de citramalato (`CITRA_SYN`). La red se entrena mediante estrategias evolutivas naturales (NES), inspiradas en el método RENAISSANCE (Choudhury et al., 2024).

Cada modelo generado se simula hasta el estado estacionario y se evalúa:

- **Estabilidad:** mediante el autovalor dominante del Jacobiano, λ_slow. Los modelos inestables reciben una penalización de −1000 y los que no se pueden simular, de −100.
- **Ajuste a los datos experimentales:** mediante una función de fitness que compara el rendimiento de citramalato (Y = 0.58 mol/mol) y la tasa de crecimiento (μ = 0.25 h⁻¹) con los valores experimentales.

## Contenido

| Fichero | Descripción |
|---|---|
| `train_nes_generator.py` | Entrena el generador con una configuración dada y guarda métricas, resumen, parámetros del mejor individuo y gráficas. |
| `run_nes_experiments.py` | Ejecuta el barrido secuencial de casos e hiperparámetros, con ajuste fino, y genera un resumen de todas las ejecuciones. |
| `E_coli_Millard2016.xml` | Modelo cinético de Millard et al. en formato SBML. |
| `requirements.txt` | Dependencias de Python. |

## Instalación

Se ha utilizado Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

## Uso

### Un entrenamiento

El script se configura mediante variables de entorno:

| Variable | Descripción | Por defecto |
|---|---|---|
| `NES_PARAMETER_MODE` | Parámetros estimados: A (citramalato), B (+ crecimiento), C (todos los Vmax), D (Vmax y Km), E (todos). | `B` |
| `NES_POPULATION_SIZE` | Tamaño de la población *n*. | `20` |
| `NES_SIGMA` | Ruido de exploración σ. | `0.1` |
| `NES_LEARNING_RATE` | Tasa de aprendizaje inicial η₀. | `0.01` |
| `NES_LEARNING_RATE_DECAY` | Factor de reducción de η por generación. | `0.95` |
| `NES_GENERATIONS` | Número de generaciones. | `25` |
| `NES_HIDDEN_LAYERS` | Capas ocultas del generador. | `256,512,1024` |
| `NES_NOISE_DIM` | Dimensión del vector de ruido *z*. | `16` |
| `NES_FEED` | Aporte de glucosa del modelo original (mM/s). | `0.23` |
| `NES_SEED` | Semilla de los generadores de números aleatorios. | `42` |
| `NES_OUTPUT_DIR` | Carpeta de resultados. Si no se indica, las gráficas se muestran en pantalla y no se guarda nada. | — |
| `NES_RUN_NAME` | Prefijo de los ficheros de resultados. | `single_run` |

Ejemplo (configuración final del trabajo):

```bash
# Linux / macOS
NES_PARAMETER_MODE=E NES_POPULATION_SIZE=200 NES_SIGMA=0.032 NES_LEARNING_RATE=0.008 \
NES_OUTPUT_DIR=resultados NES_RUN_NAME=final python train_nes_generator.py
```

```powershell
# Windows (PowerShell)
$env:NES_PARAMETER_MODE="E"; $env:NES_POPULATION_SIZE="200"; $env:NES_SIGMA="0.032"
$env:NES_LEARNING_RATE="0.008"; $env:NES_OUTPUT_DIR="resultados"; $env:NES_RUN_NAME="final"
python train_nes_generator.py
```

Los ficheros de resultados son los siguientes:

- `<run>_metrics.csv`: métricas por generación (fitness, λ_slow, rendimiento y crecimiento).
- `<run>_summary.json`: resumen de la ejecución, incluida la referencia del modelo original.
- `<run>_best_params.csv`: parámetros del mejor individuo frente a los originales.
- `<run>_best_params_by_generation.csv`: mejor individuo de cada generación.
- Gráficas de evolución en formato PNG.

### Barrido de hiperparámetros

```bash
python run_nes_experiments.py
```

El barrido sigue estos pasos:

1. Prueba los casos A–E y, después, σ, η₀, la arquitectura y el tamaño de población, cada uno partiendo del mejor valor del anterior.
2. Hace un ajuste fino de σ, η₀ y *n* (×0.8, ×1, ×1.2).
3. Si el mejor valor de un grupo queda en un extremo del intervalo, prueba un valor más allá.
4. Con la semilla fija, no vuelve a ejecutar las configuraciones repetidas.

Los resultados se guardan en `resultados_nes_semilla/`. Los valores probados se pueden cambiar al principio del script.

## Tiempo de cálculo

El tiempo de un entrenamiento es aproximadamente *t* ≈ *G* · *n* · 1.5 s, donde *G* es el número de generaciones y *n* el tamaño de la población. Por ejemplo, la configuración por defecto (*G* = 25, *n* = 20) tarda unos 12.5 minutos en CPU (Intel Core Ultra 7 255H).

## Referencias

- Millard, P., Smallbone, K. y Mendes, P. (2017). Metabolic regulation is sufficient for global and robust coordination of glucose uptake, catabolism, energy production and growth in *Escherichia coli*. *PLoS Computational Biology*, 13(2), e1005396.
- Choudhury, S. et al. (2024). Generative machine learning produces kinetic models that accurately characterize intracellular metabolic states. *Nature Catalysis*, 7, 1086–1098.
- Webb, J. P. et al. (2018). Efficient bio-production of citramalate using an engineered *Escherichia coli* strain. *Microbiology*, 164, 133–141.
