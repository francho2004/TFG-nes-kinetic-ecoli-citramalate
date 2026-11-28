"""Ejecuta barridos NES y guarda sus resultados en resultados_nes_semilla."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


# Edita estas variables para elegir el barrido. Cada grupo añade ejecuciones.
EXPERIMENT_GROUPS = ("cases", "sigma", "learning_rate", "architecture", "population")
CASES = ("A", "B", "C", "D", "E")
SIGMAS = (0.001, 0.01, 0.05, 0.1, 0.2)
LEARNING_RATES = (1e-4, 1e-3, 1e-2)
ARCHITECTURES = {
    "small": (64,),
    "medium": (128, 64),
    "current": (256, 512, 1024),
}
POPULATION_SIZES = (10, 20, 50, 100)
FINE_EXPERIMENT_GROUPS = ("sigma", "learning_rate", "population")
FINE_FACTORS = (0.8, 1.0, 1.2)

# Semilla común a todas las ejecuciones: así las diferencias entre
# configuraciones se deben a los hiperparámetros y no al azar.
SEED = 42

# Grupos numéricos en los que, si gana un extremo, se prueba un valor más allá.
# El nuevo valor sigue la misma progresión que los dos últimos valores probados
# (p. ej. learning rate 1e-3, 1e-4 -> 1e-5; sigma 0.1, 0.2 -> 0.4).
EXTENDABLE_GROUPS = {"sigma": "sigma", "learning_rate": "learning_rate", "population": "population"}

SCRIPT = Path(__file__).with_name("train_nes_generator.py")
RESULTS_DIR = Path(__file__).with_name("resultados_nes_semilla")

# Con semilla fija, la misma configuración da siempre el mismo resultado, así que
# no se vuelve a ejecutar: se reutiliza el resumen ya obtenido.
completed_configs = {}


def make_experiments(group, fixed):
    """Crea los candidatos de un grupo a partir de los mejores valores anteriores."""
    common = {key: value for key, value in fixed.items() if key != "name"}
    if group == "cases":
        values = ({"mode": case, "name": f"case_{case}"} for case in CASES)
    elif group == "population":
        values = (
            {"population": size, "name": f"population_{size}"}
            for size in POPULATION_SIZES
        )
    elif group == "sigma":
        values = (
            {"sigma": sigma, "name": f"sigma_{sigma:g}"}
            for sigma in SIGMAS
        )
    elif group == "learning_rate":
        values = (
            {"learning_rate": rate, "name": f"learning_rate_{rate:g}"}
            for rate in LEARNING_RATES
        )
    elif group == "architecture":
        values = (
            {"layers": layers, "name": f"architecture_{name}"}
            for name, layers in ARCHITECTURES.items()
        )
    else:
        raise ValueError(f"Grupo de experimentos no válido: {group}")

    experiments = []
    for candidate in values:
        experiment = common | candidate
        experiment["name"] = f"{group}_{candidate['name']}"
        experiments.append(experiment)
    return experiments


def make_fine_experiments(group, fixed):
    """Crea tres candidatos alrededor del mejor valor actual."""
    common = {key: value for key, value in fixed.items() if key != "name"}
    if group == "sigma":
        center = fixed["sigma"]
        values = (
            {"sigma": center * factor, "name": f"sigma_{center * factor:g}"}
            for factor in FINE_FACTORS
        )
    elif group == "learning_rate":
        center = fixed["learning_rate"]
        values = (
            {
                "learning_rate": center * factor,
                "name": f"learning_rate_{center * factor:g}",
            }
            for factor in FINE_FACTORS
        )
    elif group == "population":
        center = fixed["population"]
        values = (
            {
                "population": max(2, round(center * factor)),
                "name": f"population_{max(2, round(center * factor))}",
            }
            for factor in FINE_FACTORS
        )
    else:
        raise ValueError(f"Grupo de ajuste fino no válido: {group}")

    experiments = []
    for candidate in values:
        experiment = common | candidate
        experiment["name"] = f"fine_{group}_{candidate['name']}"
        experiments.append(experiment)
    return experiments


def make_extension(group, group_results, prefix):
    """Si el mejor valor está en un extremo del intervalo, crea una prueba más allá.

    Devuelve None si el mejor valor es interior (el óptimo queda acotado).
    """
    key = EXTENDABLE_GROUPS.get(group)
    if key is None:
        return None
    values = sorted({experiment[key] for experiment, _ in group_results})
    if len(values) < 2:
        return None
    best_experiment, _ = best_result(group_results)
    best_value = best_experiment[key]
    if best_value == values[-1]:
        new_value = best_value * best_value / values[-2]
    elif best_value == values[0]:
        new_value = best_value * best_value / values[1]
    else:
        return None
    if key == "population":
        new_value = max(2, round(new_value))
        if new_value in values:
            return None

    experiment = {k: v for k, v in best_experiment.items() if k != "name"}
    experiment[key] = new_value
    experiment["name"] = f"{prefix}{group}_{key}_{new_value:g}"
    return experiment


# Valores por defecto de train_nes_generator.py, usados cuando una
# ejecución no fija el hiperparámetro (p. ej. el caso ganador equivale a sigma_0.1).
DEFAULTS = {"population": 20, "sigma": 0.1, "learning_rate": 0.01, "layers": (256, 512, 1024)}


def config_key(experiment):
    config = DEFAULTS | {key: value for key, value in experiment.items() if key != "name"}
    return json.dumps(config, sort_keys=True)


def best_result(group_results):
    return max(group_results, key=lambda result: result[1]["final_best_fitness"])


def run_experiment(experiment, position):
    name = experiment["name"]
    key = config_key(experiment)
    if key in completed_configs:
        previous = completed_configs[key]
        print(
            f"[{position}] {name}: misma configuración que {previous['run_name']}, "
            "se reutiliza su resultado (semilla fija)"
        )
        return previous | {"run_name": name, "reused_from": previous["run_name"]}

    output_dir = RESULTS_DIR / name
    summary_path = output_dir / f"{name}_summary.json"
    metrics_path = output_dir / f"{name}_metrics.csv"
    fitness_path = output_dir / f"{name}_fitness.png"
    lambda_path = output_dir / f"{name}_lambda.png"
    yield_path = output_dir / f"{name}_yield.png"

    if all(
        path.is_file()
        for path in (summary_path, metrics_path, fitness_path, lambda_path, yield_path)
    ):
        print(f"[{position}] Omitiendo {name}: ya está completado")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        completed_configs[key] = summary
        return summary

    environment = os.environ.copy()
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "NES_PARAMETER_MODE": experiment["mode"],
            "NES_OUTPUT_DIR": str(output_dir),
            "NES_RUN_NAME": name,
            "NES_SEED": str(SEED),
        }
    )
    if "population" in experiment:
        environment["NES_POPULATION_SIZE"] = str(experiment["population"])
    if "sigma" in experiment:
        environment["NES_SIGMA"] = str(experiment["sigma"])
    if "learning_rate" in experiment:
        environment["NES_LEARNING_RATE"] = str(experiment["learning_rate"])
    if "layers" in experiment:
        environment["NES_HIDDEN_LAYERS"] = ",".join(map(str, experiment["layers"]))

    print(f"[{position}] Ejecutando {name}")
    subprocess.run([sys.executable, str(SCRIPT)], check=True, env=environment)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    completed_configs[key] = summary
    return summary


def run_group(group, experiments, fixed, prefix, label, summaries, position):
    """Ejecuta un grupo y, si el mejor valor está en un extremo, una prueba adicional."""
    print(
        f"\n{label} {group}: {len(experiments)} experimentos "
        f"con configuración fijada {fixed}"
    )
    group_results = []
    for experiment in experiments:
        position += 1
        summary = run_experiment(experiment, position)
        group_results.append((experiment, summary))
        summaries.append(summary)

    extension = make_extension(group, group_results, prefix)
    if extension is not None:
        print(
            f"El mejor valor de {group} está en un extremo del intervalo; "
            f"se prueba además {extension['name']}"
        )
        position += 1
        summary = run_experiment(extension, position)
        group_results.append((extension, summary))
        summaries.append(summary)

    best_experiment, best_summary = best_result(group_results)
    fixed = {key: value for key, value in best_experiment.items() if key != "name"}
    print(
        f"Mejor {label.lower()} {group}: {best_experiment['name']} "
        f"(fitness={best_summary['final_best_fitness']:.6f})"
    )
    return fixed, position


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fixed = {}
    summaries = []
    position = 0

    for group in EXPERIMENT_GROUPS:
        experiments = make_experiments(group, fixed)
        fixed, position = run_group(
            group, experiments, fixed, "", "Grupo", summaries, position
        )

    print("\nIniciando barrido fino alrededor de los mejores hiperparámetros")
    for group in FINE_EXPERIMENT_GROUPS:
        experiments = make_fine_experiments(group, fixed)
        fixed, position = run_group(
            group, experiments, fixed, "fine_", "Ajuste fino", summaries, position
        )

    print(f"\nConfiguración final: {fixed}")

    if summaries:
        summary_path = RESULTS_DIR / "experiments_summary.csv"
        fields = []
        for summary in summaries:
            fields += [field for field in summary if field not in fields]
        with summary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Resumen guardado en {summary_path}")


if __name__ == "__main__":
    main()
