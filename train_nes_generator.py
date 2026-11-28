import os

# Necesario para que las operaciones de cuBLAS sean deterministas en GPU.
# Debe definirse antes de importar torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
from scipy.stats import rankdata
import torch
import torch.nn as nn
import numpy as np
import tellurium as te
import matplotlib.pyplot as plt
import csv
import json
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# CONFIGURACIÓN DE EXPERIMENTOS
# ===========================

FEED = float(os.getenv("NES_FEED", "0.23"))

# A: solo citramalato, B: citramalato + parámetros GROWTH,
# C: todos los Vmax, D: todos los Vmax y Km, E: todos los parámetros.
PARAMETER_MODE = os.getenv("NES_PARAMETER_MODE", "B")

NOISE_DIM = int(os.getenv("NES_NOISE_DIM", "16"))
POPULATION_SIZE = int(os.getenv("NES_POPULATION_SIZE", "20"))
SIGMA = float(os.getenv("NES_SIGMA", "0.1"))
LEARNING_RATE = float(os.getenv("NES_LEARNING_RATE", "0.01"))
LEARNING_RATE_DECAY = float(os.getenv("NES_LEARNING_RATE_DECAY", "0.95"))
GENERATIONS = int(os.getenv("NES_GENERATIONS", "25"))

# Pequeña: 16 -> 64 -> parámetros; media: 16 -> 128 -> 64 -> parámetros;
# grande (por defecto): 16 -> 256 -> 512 -> 1024 -> parámetros.
GENERATOR_HIDDEN_LAYERS = tuple(
    int(layer) for layer in os.getenv("NES_HIDDEN_LAYERS", "256,512,1024").split(",")
    if layer.strip()
)
OUTPUT_DIR = os.getenv("NES_OUTPUT_DIR")
RUN_NAME = os.getenv("NES_RUN_NAME", "single_run")

# Semilla fija: con la misma configuración se obtiene exactamente el mismo
# resultado, de modo que las diferencias entre hiperparámetros no se deben al azar.
SEED = int(os.getenv("NES_SEED", "42"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # también inicializa la semilla de todas las GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# ===========================
# 1. MODELO
# ===========================

antimony_str = te.sbmlToAntimony('E_coli_Millard2016.xml')

citramalate_block = """
CITRA_SYN: ACCOA + PYR -> ; Vmax_cim * ACCOA / (ACCOA + Km_accoa);

Vmax_cim = 4.06;
Km_accoa = 0.495;
"""


idx = antimony_str.rfind('end')

antimony_model = (
    antimony_str[:idx] +
    citramalate_block +
    '\nend' +
    antimony_str[idx+3:]
)

# cargar modelo
rr_tmp = te.loada(antimony_model)
rr_tmp.setGlobalParameterByName("FEED", FEED)  # Ajustar el parámetro FEED según sea necesario

# ===========================
# STEADY STATE (UNA VEZ)
# ===========================

rr_tmp.simulate(0, 2000, 500)
value = rr_tmp.steadyState()
print("Steady state value:", value)
J = rr_tmp.getFullJacobian()
eigs = np.linalg.eigvals(J)
print("baseline lambda_max:", np.max(np.real(eigs)))

print("Steady state loaded")

lambda_values = []
yield_values = []
growth_values = []
# ===========================
# 2. PARÁMETROS
# ===========================
parameter_ids = rr_tmp.getGlobalParameterIds()
citramalate_parameters = ["Vmax_cim", "Km_accoa"]
growth_parameters = [
    "GROWTH_KmG6P", "GROWTH_KmE4P", "GROWTH_KmPGA3", "GROWTH_KmOAA",
    "GROWTH_KmAKG", "GROWTH_KmPYR", "GROWTH_KmR5P", "GROWTH_KmPEP",
    "GROWTH_KmGAP", "GROWTH_KmF6P", "GROWTH_KmNADPH", "GROWTH_KmACCOA",
    "GROWTH_KmNAD", "GROWTH_KmATP", "GROWTH_Vmax",
]

parameter_modes = {
    "A": citramalate_parameters,
    "B": citramalate_parameters + growth_parameters,
    "C": [p for p in parameter_ids if "Vmax" in p],
    "D": [p for p in parameter_ids if "Vmax" in p or "Km" in p],
    "E": [p for p in parameter_ids]
}
if PARAMETER_MODE not in parameter_modes:
    raise ValueError(f"PARAMETER_MODE no válido: {PARAMETER_MODE}. Usa A, B, C, D o E.")

# FEED se añade una sola vez al final; en el caso E ya está entre los
# parámetros globales y no debe duplicarse.
param_names = [p for p in parameter_modes[PARAMETER_MODE] if p != "FEED"]
param_names += ["FEED"]  # Agregar FEED a la lista de parámetros

base_params = np.array([
    rr_tmp[p]
    for p in param_names
])

param_dim = len(param_names)
feed_index = param_names.index("FEED")

print("Número de parámetros:", param_dim)

# ===========================
# 3. GENERADOR
# ===========================

class Generator(nn.Module):
    def __init__(self, noise_dim, param_dim):
        super().__init__()
        layers = []
        input_dim = noise_dim
        for hidden_dim in GENERATOR_HIDDEN_LAYERS:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.extend((nn.Linear(input_dim, param_dim), nn.Tanh()))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

# ===========================
# 4. UTILIDADES
# ===========================

def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data = flat_params[pointer:pointer+numel].view_as(p)
        pointer += numel


def generate_params(delta):
    params = base_params * ((delta / 4) + 1)
    params[feed_index] = 0.1 + (np.clip(delta[feed_index], -1, 1) + 1) * 0.25
    return params


# El modelo se compila una sola vez; en cada evaluación se devuelve a su estado
# inicial con resetToOrigin(), que es mucho más rápido que volver a cargarlo.
rr_eval = te.loada(antimony_model)


def build_rr(params):
    global rr_eval
    try:
        rr_eval.resetToOrigin()
    except Exception:
        # Si una simulación anterior dejó el modelo en mal estado, se recarga.
        rr_eval = te.loada(antimony_model)
    rr = rr_eval
    for name, value in zip(param_names, params):
        rr[name] = float(value)
    return rr

# ===========================
# 5. FITNESS
# ===========================

def compute_fitness(params):
    try:
        rr = build_rr(params)

        rr.simulate(0, 2000, 500)
        rr.steadyState()

        J = rr.getFullJacobian()

        eigs = np.linalg.eigvals(J)

        real_parts = np.real(eigs)

        # eliminar modos conservados
        real_parts = real_parts[np.abs(real_parts) > 1e-8]

        lam_slow = np.max(real_parts)

        citramalate_flux = rr.getValue("CITRA_SYN")
        growth_flux = rr.getValue("GROWTH")
        glucose_flux = rr.getValue("GLC_feed")
        cim_yield = abs(citramalate_flux) / (abs(glucose_flux) + 1e-12)

        yield_model = citramalate_flux / glucose_flux
        yield_exp = 0.58

        growth_model = growth_flux * 3600
        growth_exp = 0.25

        yield_error = (
            (yield_model - yield_exp)
            / yield_exp
        )**2

        growth_error = (
            (growth_model - growth_exp)
            / growth_exp
        )**2

        # Las métricas se guardan solo cuando la evaluación se completa, para que
        # las listas queden alineadas con fitness_values.
        lambda_values.append(lam_slow)
        yield_values.append(cim_yield)
        growth_values.append(growth_model)

        if lam_slow >= 0:
            return -1000


        fitness = -(
            yield_error +
            growth_error
        )

        print(
            f"yield_error={yield_error:.3f}",
            f"growth_error={growth_error:.3f}",
            f"fitness={fitness:.3f}"
        )

        return fitness

    except Exception as e:
        print(f"Error computing fitness: {e}")
        lambda_values.append(np.nan)
        yield_values.append(np.nan)
        growth_values.append(np.nan)
        return -100


def evaluate_original():
    """Fitness y métricas del modelo original (parámetros sin modificar, FEED=0.23)."""
    lambda_values.clear()
    yield_values.clear()
    growth_values.clear()
    fitness = compute_fitness(base_params.copy())
    return {
        "fitness": float(fitness),
        "lambda": float(lambda_values[-1]),
        "yield": float(yield_values[-1]),
        "growth": float(growth_values[-1]),
    }

# ===========================
# 6. ENTRENAMIENTO NES
# ===========================

def train(
    noise_dim=NOISE_DIM,
    population_size=POPULATION_SIZE,
    sigma=SIGMA,
    lr=LEARNING_RATE,
    generations=GENERATIONS,
    save_results=True,
    original=None,
):

    G = Generator(noise_dim, param_dim).to(device)

    theta = get_flat_params(G).clone()
    initial_learning_rate = lr
    best_fitness_history = []
    mean_fitness_history = []
    best_lambda_history = []
    mean_lambda_history = []
    best_fitness_lambda_history = []
    best_yield_history = []
    mean_yield_history = []
    best_growth_history = []
    mean_growth_history = []
    best_params_history = []  # parámetros del mejor individuo de cada generación

    for gen in range(generations):

        epsilons = []
        fitness_values = []
        population_params = []
        lambda_values.clear()  # se vacían las métricas de la generación anterior
        yield_values.clear()
        growth_values.clear()

        z = torch.randn(1, noise_dim).to(device)

        # ===== SAMPLE =====
        for i in range(population_size):

            eps = torch.randn_like(theta)
            theta_i = theta + sigma * eps
            set_flat_params(G, theta_i)

            delta = G(z).detach().cpu().numpy()[0]

            params = generate_params(delta)

            fitness = compute_fitness(params)

            epsilons.append(eps.cpu().numpy())
            fitness_values.append(fitness)
            population_params.append(params)

        fitness_values = np.array(fitness_values)
        best_index = np.argmax(fitness_values)
        print(f"Gen {gen} | best fitness {fitness_values.max():.4f}")
        print(f"Gen {gen} | mean fitness {fitness_values.mean():.4f}")
        best_fitness_history.append(fitness_values.max())
        mean_fitness_history.append(fitness_values.mean())
        best_params_history.append(population_params[best_index])
        # normalizar por rangos (fitness shaping): solo cuenta el orden de los
        # individuos, así las penalizaciones (-100, -1000) no dominan la
        # normalización. Los empates reciben el rango medio. Los rangos se
        # estandarizan para que tengan media nula y varianza unitaria.
        ranks = rankdata(fitness_values)
        fitness_values = (ranks - ranks.mean()) / (ranks.std() + 1e-8)

        # ===== GRADIENTE NES =====
        grad = np.zeros_like(theta.cpu().numpy())

        for i in range(population_size):
            grad += fitness_values[i] * epsilons[i]

        grad = grad / (population_size * sigma)

        # ===== UPDATE =====
        theta += lr * torch.tensor(grad, dtype=torch.float32)
        lr *= LEARNING_RATE_DECAY

        set_flat_params(G, theta)


        best_lambda = np.nanmin(lambda_values)
        mean_lambda = np.nanmean(lambda_values)
        # λ_slow del individuo con mejor fitness (no tiene por qué ser el mínimo)
        best_fitness_lambda = lambda_values[best_index]
        best_lambda_history.append(best_lambda)
        mean_lambda_history.append(mean_lambda)
        best_fitness_lambda_history.append(best_fitness_lambda)
        best_yield_history.append(yield_values[best_index])
        mean_yield_history.append(np.nanmean(yield_values))
        best_growth_history.append(growth_values[best_index])
        mean_growth_history.append(np.nanmean(growth_values))

        print(
            f"Gen {gen}"
            f" | best λ={best_lambda:.6f}"
            f" | mean λ={mean_lambda:.6f}"
            f" | lambda mejor fitness={best_fitness_lambda:.6f}"
        )

    history = {
        "generation": np.arange(1, generations + 1),
        "best_fitness": np.array(best_fitness_history),
        "mean_fitness": np.array(mean_fitness_history),
        "best_lambda": np.array(best_lambda_history),
        "mean_lambda": np.array(mean_lambda_history),
        "best_fitness_lambda": np.array(best_fitness_lambda_history),
        "best_yield": np.array(best_yield_history),
        "mean_yield": np.array(mean_yield_history),
        "best_growth": np.array(best_growth_history),
        "mean_growth": np.array(mean_growth_history),
    }
    best_params_history = np.array(best_params_history)

    if save_results and OUTPUT_DIR:
        save_best_params(best_params_history, history["best_fitness"], OUTPUT_DIR, RUN_NAME)
        save_training_results(
            history,
            OUTPUT_DIR,
            RUN_NAME,
            population_size,
            sigma,
            initial_learning_rate,
            generations,
            original,
        )

    generations_axis = history["generation"]
    plt.figure(figsize=(10, 5))
    plt.plot(generations_axis, best_fitness_history, label="Best fitness", linewidth=2)
    plt.plot(generations_axis, mean_fitness_history, label="Mean fitness", linewidth=2)
    if original is not None:
        plt.axhline(original["fitness"], color="gray", linestyle="--", label="Original model")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness evolution by generation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if not OUTPUT_DIR:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(generations_axis, best_lambda_history, label="Best lambda", linewidth=2)
    plt.plot(generations_axis, mean_lambda_history, label="Mean lambda", linewidth=2)
    plt.plot(generations_axis, best_fitness_lambda_history, label="Lambda of best-fitness individual", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Lambda")
    plt.title("Lambda evolution by generation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if not OUTPUT_DIR:
        plt.show()
    else:
        plt.close()


    return G, history, best_params_history


def save_best_params(best_params_history, best_fitness, output_dir, run_name):
    """Guarda los parámetros de los mejores individuos.

    - <run>_best_params.csv: por parámetro, valor original, del mejor individuo
      de toda la ejecución y del mejor de la última generación, con su variación (%).
    - <run>_best_params_by_generation.csv: mejor individuo de cada generación.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_gen = int(np.argmax(best_fitness))
    overall = best_params_history[best_gen]
    last = best_params_history[-1]

    with (output_path / f"{run_name}_best_params.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "parameter", "original",
            f"best_overall_gen{best_gen + 1}", "best_overall_change_pct",
            "best_last_gen", "best_last_gen_change_pct",
        ])
        for name, base, b, l in zip(param_names, base_params, overall, last):
            writer.writerow([
                name, base,
                b, 100 * (b - base) / base,
                l, 100 * (l - base) / base,
            ])

    with (output_path / f"{run_name}_best_params_by_generation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["generation", "best_fitness"] + param_names)
        for gen, (fit, params) in enumerate(zip(best_fitness, best_params_history), 1):
            writer.writerow([gen, fit] + list(params))


def save_training_results(
    history, output_dir, run_name, population_size, sigma, learning_rate, generations,
    original=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"{run_name}_metrics.csv"
    fields = list(history)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(zip(*(history[field] for field in fields)))

    plt.figure(figsize=(10, 5))
    plt.plot(history["generation"], history["best_fitness"], label="Best fitness")
    plt.plot(history["generation"], history["mean_fitness"], label="Mean fitness")
    if original is not None:
        plt.axhline(original["fitness"], color="gray", linestyle="--", label="Original model")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"Fitness: {run_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{run_name}_fitness.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history["generation"], history["best_lambda"], label="Best lambda")
    plt.plot(history["generation"], history["mean_lambda"], label="Mean lambda")
    plt.plot(
        history["generation"], history["best_fitness_lambda"],
        label="Lambda of best-fitness individual",
    )
    plt.xlabel("Generation")
    plt.ylabel("Lambda")
    plt.title(f"Lambda: {run_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{run_name}_lambda.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history["generation"], history["best_yield"], label="Best yield")
    plt.plot(history["generation"], history["mean_yield"], label="Mean yield")
    plt.xlabel("Generation")
    plt.ylabel("Citramalate/glucose yield")
    plt.title(f"Yield: {run_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{run_name}_yield.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history["generation"], history["best_growth"], label="Best growth")
    plt.plot(history["generation"], history["mean_growth"], label="Mean growth")
    plt.xlabel("Generation")
    plt.ylabel("Growth rate (1/h)")
    plt.title(f"Growth: {run_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{run_name}_growth.png", dpi=150)
    plt.close()

    summary = {
        "run_name": run_name,
        "parameter_mode": PARAMETER_MODE,
        "parameter_count": param_dim,
        "population_size": population_size,
        "sigma": sigma,
        "learning_rate": learning_rate,
        "generations": generations,
        "hidden_layers": list(GENERATOR_HIDDEN_LAYERS),
        "seed": SEED,
        "original_fitness": original["fitness"] if original else None,
        "original_lambda": original["lambda"] if original else None,
        "original_yield": original["yield"] if original else None,
        "original_growth": original["growth"] if original else None,
        "final_best_fitness": float(history["best_fitness"][-1]),
        "final_mean_fitness": float(history["mean_fitness"][-1]),
        "final_best_lambda": float(history["best_lambda"][-1]),
        "final_best_fitness_lambda": float(history["best_fitness_lambda"][-1]),
        "final_best_yield": float(history["best_yield"][-1]),
        "final_best_growth": float(history["best_growth"][-1]),
        "final_mean_growth": float(history["mean_growth"][-1]),
    }
    (output_path / f"{run_name}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

# ===========================
# 7. RUN
# ===========================

if __name__ == "__main__":

    set_seed(SEED)
    print("Seed:", SEED)

    original = evaluate_original()
    print(
        "Original model:"
        f" fitness={original['fitness']:.4f}"
        f" | lambda_slow={original['lambda']:.6f}"
        f" | yield={original['yield']:.4f}"
        f" | growth={original['growth']:.4f} 1/h"
    )
    G, history, best_params_history = train(original=original)

    # Mejor individuo de toda la ejecución (no una muestra al azar del generador final)
    best_gen = int(np.argmax(history["best_fitness"]))
    best_params = best_params_history[best_gen]
    print(
        f"\nBest individual: generation {best_gen + 1}"
        f" | fitness={history['best_fitness'][best_gen]:.4f}"
    )

    print("\nParameter | original | best | change (%)")
    for name, base, val in zip(param_names, base_params, best_params):
        print(name, base, val, f"{100 * (val - base) / base:+.2f}")
