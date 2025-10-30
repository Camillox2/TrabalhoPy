import argparse
import json
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from jinja2 import Template
except ImportError as exc:  # pragma: no cover - fail fast if jinja2 disappeared
    raise SystemExit("jinja2 is required to render the HTML report") from exc


RANDOM_STATE = 42


@dataclass
class ModelResult:
    name: str
    estimator: ImbPipeline
    best_params: Dict[str, object]
    cv_f1_macro: float
    test_metrics: Dict[str, float]
    classification_report_text: str
    confusion: np.ndarray


def load_data(dataset_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(dataset_path)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")
    features = df.drop(columns=["Class"])
    target = df["Class"]
    return features, target


def build_model_configs() -> Dict[str, Dict[str, object]]:
    base_steps = [
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
    ]

    configs = {
        "logistic_regression": {
            "pipeline": ImbPipeline(
                steps=[*base_steps, ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))]
            ),
            "param_grid": {
                "select__k": [10, 15, 20, "all"],
                "clf__C": [0.1, 1.0, 10.0],
            },
        },
        "random_forest": {
            "pipeline": ImbPipeline(
                steps=[
                    *base_steps,
                    (
                        "clf",
                        RandomForestClassifier(
                            class_weight=None,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "select__k": [15, 20, "all"],
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 12],
                "clf__min_samples_leaf": [1, 3],
            },
        },
        "gradient_boosting": {
            "pipeline": ImbPipeline(
                steps=[
                    *base_steps,
                    (
                        "clf",
                        GradientBoostingClassifier(random_state=RANDOM_STATE),
                    ),
                ]
            ),
            "param_grid": {
                "select__k": [15, 20, "all"],
                "clf__n_estimators": [150, 250],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
        },
        "svm": {
            "pipeline": ImbPipeline(
                steps=[
                    *base_steps,
                    (
                        "clf",
                        SVC(
                            probability=True,
                            kernel="rbf",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "select__k": [10, 15, 20],
                "clf__C": [0.5, 1.0, 5.0],
                "clf__gamma": ["scale", "auto"],
            },
        },
        "knn": {
            "pipeline": ImbPipeline(
                steps=[
                    *base_steps,
                    (
                        "clf",
                        KNeighborsClassifier(),
                    ),
                ]
            ),
            "param_grid": {
                "select__k": [10, 15, 20],
                "clf__n_neighbors": [5, 11, 15],
                "clf__weights": ["uniform", "distance"],
            },
        },
    }
    return configs


def run_model_search(
    name: str,
    config: Dict[str, object],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelResult:
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }
    grid = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["param_grid"],
        scoring=scoring,
        refit="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(x_train, y_train)
    best_estimator: ImbPipeline = grid.best_estimator_
    y_pred = best_estimator.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    }

    report_text = classification_report(y_test, y_pred, digits=4)
    confusion = confusion_matrix(y_test, y_pred)

    return ModelResult(
        name=name,
        estimator=best_estimator,
        best_params=grid.best_params_,
        cv_f1_macro=grid.best_score_,
        test_metrics=metrics,
        classification_report_text=report_text,
        confusion=confusion,
    )


def pick_select_k(value: object, fallback: int) -> object:
    if value == "all" or value is None:
        return "all"
    if isinstance(value, (int, np.integer)):
        return int(value)
    return fallback


def build_voting_ensemble(
    top_results: List[ModelResult],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    select_k: object,
) -> ModelResult:
    estimators = [
        (
            result.name,
            clone(result.estimator.named_steps["clf"]),
        )
        for result in top_results
    ]

    weights = [result.test_metrics["f1_macro"] for result in top_results]

    pipeline = ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=select_k)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "clf",
                VotingClassifier(
                    estimators=estimators,
                    voting="soft",
                    weights=weights,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    return ModelResult(
        name="voting_ensemble",
        estimator=pipeline,
        best_params={
            "select__k": select_k,
            "weights": weights,
            "estimators": [name for name, _ in estimators],
        },
        cv_f1_macro=float("nan"),
        test_metrics={},  # placeholder, will be filled by caller
        classification_report_text="",
        confusion=np.empty((0, 0)),
    )


def evaluate_model(model_result: ModelResult, x_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
    y_pred = model_result.estimator.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    }
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    model_result.test_metrics = metrics
    model_result.confusion = confusion
    model_result.classification_report_text = report
    return model_result


def plot_confusion_matrix(confusion: np.ndarray, labels: List[str], title: str, output_path: Path) -> None:
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.ylabel("Verdadeiro")
    plt.xlabel("Previsto")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def export_html_report(
    output_dir: Path,
    dataset_info: Dict[str, object],
    model_results: List[ModelResult],
    ensemble_result: ModelResult,
    confusion_paths: Dict[str, str],
    run_context: str,
) -> None:
    template_text = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Relatorio - Deteccao de Fraudes</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 24px; background-color: #f8fafc; color: #111827; }
        h1, h2, h3 { color: #1f2937; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
        th, td { border: 1px solid #cbd5f5; padding: 8px 12px; text-align: left; }
        th { background-color: #ede9fe; }
        .card { background: white; border-radius: 8px; padding: 16px 20px; box-shadow: 0 2px 6px rgba(79,70,229,0.1); margin-bottom: 24px; }
        .confusion { display: flex; flex-wrap: wrap; gap: 24px; }
        .confusion img { max-width: 360px; width: 100%; border: 1px solid #cbd5f5; border-radius: 8px; }
        pre { background-color: #1f2937; color: #f9fafb; padding: 16px; border-radius: 6px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>Deteccao de Fraudes em Cartoes de Credito</h1>
    <div class="card">
        <h2>Resumo do Dataset</h2>
        <p><strong>Modo de execucao:</strong> {{ run_context }}</p>
        <p><strong>Instancias:</strong> {{ dataset_info.samples }} | <strong>Features:</strong> {{ dataset_info.features }} | <strong>Fraudes (%)</strong> {{ dataset_info.fraud_percentage | round(2) }}</p>
        <table>
            <thead>
                <tr><th>Classe</th><th>Quantidade</th><th>Percentual</th></tr>
            </thead>
            <tbody>
                {% for label, stats in dataset_info.class_breakdown.items() %}
                <tr><td>{{ label }}</td><td>{{ stats.count }}</td><td>{{ (stats.count / dataset_info.samples * 100) | round(2) }}%</td></tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Modelos Individuais</h2>
        <table>
            <thead>
                <tr>
                    <th>Modelo</th>
                    <th>F1 Macro (CV)</th>
                    <th>Acuracia (Teste)</th>
                    <th>F1 Macro (Teste)</th>
                    <th>Recall Macro (Teste)</th>
                </tr>
            </thead>
            <tbody>
                {% for result in model_results %}
                <tr>
                    <td>{{ result.name }}</td>
                    <td>{{ result.cv_f1_macro | round(4) }}</td>
                    <td>{{ result.test_metrics.accuracy | round(4) }}</td>
                    <td>{{ result.test_metrics.f1_macro | round(4) }}</td>
                    <td>{{ result.test_metrics.recall_macro | round(4) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Ensemble (Voting Classifier)</h2>
        <p><strong>Estimadores combinados:</strong> {{ ensemble_result.best_params.estimators | join(", ") }}</p>
        <p><strong>F1 Macro (Teste):</strong> {{ ensemble_result.test_metrics.f1_macro | round(4) }} |
           <strong>Recall Macro:</strong> {{ ensemble_result.test_metrics.recall_macro | round(4) }} |
           <strong>Acuracia:</strong> {{ ensemble_result.test_metrics.accuracy | round(4) }}</p>
    </div>

    <div class="card">
        <h2>Matrizes de Confusao</h2>
        <div class="confusion">
            <div>
                <h3>Melhor Modelo Individual</h3>
                <img src="{{ confusion_paths.best }}" alt="Matriz de confusao - melhor modelo">
            </div>
            <div>
                <h3>Ensemble</h3>
                <img src="{{ confusion_paths.ensemble }}" alt="Matriz de confusao - ensemble">
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Relatorios Textuais</h2>
        {% for result in model_results %}
        <h3>{{ result.name }}</h3>
        <pre>{{ result.classification_report_text }}</pre>
        {% endfor %}
        <h3>voting_ensemble</h3>
        <pre>{{ ensemble_result.classification_report_text }}</pre>
    </div>
</body>
</html>
"""
    template = Template(template_text)
    rendered = template.render(
        dataset_info=dataset_info,
        model_results=model_results,
        ensemble_result=ensemble_result,
        confusion_paths=confusion_paths,
        run_context=run_context,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.html").write_text(rendered, encoding="utf-8")


def stringify_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def summarize_dataset(x: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    counts = y.value_counts()
    return {
        "samples": int(x.shape[0]),
        "features": int(x.shape[1]),
        "fraud_percentage": float(counts.get(1, 0) / x.shape[0] * 100),
        "class_breakdown": {int(label): {"count": int(count)} for label, count in counts.items()},
    }


def run_training_workflow(
    dataset_path: Path,
    output_dir: Path,
    generate_html: bool = True,
    open_report: bool = True,
    run_context: str = "CLI - pipeline",
) -> Dict[str, object]:
    x, y = load_data(dataset_path)
    dataset_info = summarize_dataset(x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    configs = build_model_configs()
    model_results: List[ModelResult] = []

    print("\n=== Ajuste de Modelos Individuais ===")
    for name, config in configs.items():
        print(f"\nTreinando {name} ...")
        result = run_model_search(name, config, x_train, y_train, x_test, y_test)
        model_results.append(result)
        print(f"Melhores hiperparametros: {json.dumps(result.best_params, indent=2)}")
        print(f"F1 Macro (CV): {result.cv_f1_macro:.4f}")
        print(f"Metricas de teste: {stringify_metrics(result.test_metrics)}")

    model_results.sort(key=lambda res: res.test_metrics["f1_macro"], reverse=True)
    best_model = model_results[0]

    select_k = pick_select_k(
        best_model.best_params.get("select__k"),
        fallback=x_train.shape[1],
    )

    ensemble = build_voting_ensemble(model_results[:3], x_train, y_train, select_k)
    ensemble = evaluate_model(ensemble, x_test, y_test)
    print("\n=== Ensemble (Voting Classifier) ===")
    print(f"Modelos combinados: {ensemble.best_params['estimators']}")
    print(f"Metricas de teste: {stringify_metrics(ensemble.test_metrics)}")

    labels = [str(cls) for cls in sorted(y.unique())]
    output_dir.mkdir(parents=True, exist_ok=True)
    best_confusion_path = output_dir / "confusion_best.png"
    ensemble_confusion_path = output_dir / "confusion_ensemble.png"
    plot_confusion_matrix(best_model.confusion, labels, f"Matriz de Confusao - {best_model.name}", best_confusion_path)
    plot_confusion_matrix(ensemble.confusion, labels, "Matriz de Confusao - Ensemble", ensemble_confusion_path)

    metrics_table = pd.DataFrame(
        [
            {
                "modelo": res.name,
                "f1_macro_cv": res.cv_f1_macro,
                **res.test_metrics,
            }
            for res in model_results + [ensemble]
        ]
    )
    metrics_csv_path = output_dir / "metrics.csv"
    metrics_table.to_csv(metrics_csv_path, index=False)
    print(f"Metricas consolidadas salvas em: {metrics_csv_path}")

    report_path = output_dir / "report.html"
    if generate_html:
        export_html_report(
            output_dir=output_dir,
            dataset_info=dataset_info,
            model_results=model_results,
            ensemble_result=ensemble,
            confusion_paths={
                "best": best_confusion_path.name,
                "ensemble": ensemble_confusion_path.name,
            },
            run_context=run_context,
        )
        print(f"Relatorio HTML gerado em: {report_path}")
        if open_report:
            try:
                webbrowser.open_new_tab(report_path.resolve().as_uri())
            except Exception as exc:  # pragma: no cover
                print(f"Aviso: nao foi possivel abrir automaticamente o relatorio ({exc})", file=sys.stderr)
    else:
        print("Geracao de HTML foi desativada (--no-html).")

    return {
        "dataset_info": dataset_info,
        "model_results": model_results,
        "ensemble": ensemble,
        "metrics_table": metrics_table,
        "output_dir": output_dir,
        "report_path": report_path if generate_html else None,
    }


def generate_notebook(
    dataset_path: Path,
    output_dir: Path,
    run_context: str = "CLI - notebook",
) -> Path:
    try:
        import nbformat as nbf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("nbformat is required to generate the notebook. Install it with 'pip install nbformat'.") from exc

    notebook_dir = Path("notebooks")
    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebook_dir / "fraud_detection.ipynb"

    dataset_literal = dataset_path.resolve().as_posix()
    output_literal = output_dir.resolve().as_posix()

    markdown_intro = f"""# Detecao de Fraudes em Cartoes

Notebook gerado automaticamente pelo CLI ({run_context}).

Use as celulas abaixo para reproduzir o treinamento, analisar as metricas e visualizar os artefatos gerados.
"""

    setup_code = f"""from pathlib import Path

DATASET_PATH = Path(r\"{dataset_literal}\")
OUTPUT_DIR = Path(r\"{output_literal}\")
"""

    run_code = """from py.main import run_training_workflow

result = run_training_workflow(
    dataset_path=DATASET_PATH,
    output_dir=OUTPUT_DIR,
    generate_html=True,
    open_report=False,
    run_context="Notebook"
)
metrics = result["metrics_table"]
metrics
"""

    confusion_code = """from IPython.display import Image, display

display(Image(filename=OUTPUT_DIR / "confusion_best.png"))
display(Image(filename=OUTPUT_DIR / "confusion_ensemble.png"))
"""

    html_preview_code = """from IPython.display import HTML

report_file = OUTPUT_DIR / "report.html"
if report_file.exists():
    HTML(report_file.read_text(encoding="utf-8"))
else:
    print("Relatorio HTML ainda nao foi gerado. Rode a celula anterior primeiro.")
"""

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(markdown_intro),
        nbf.v4.new_code_cell(setup_code.strip()),
        nbf.v4.new_code_cell(run_code.strip()),
        nbf.v4.new_code_cell(confusion_code.strip()),
        nbf.v4.new_code_cell(html_preview_code.strip()),
    ]
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {
        "name": "python",
        "version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }

    with notebook_path.open("w", encoding="utf-8") as fp:
        nbf.write(nb, fp)

    print(f"Notebook atualizado em: {notebook_path}")
    return notebook_path


MODE_MAP = {
    "1": "pipeline",
    "2": "notebook",
    "3": "both",
}


def prompt_mode_selection(initial: Optional[str]) -> str:
    if initial:
        if initial in MODE_MAP.values():
            return initial
        if initial in MODE_MAP:
            return MODE_MAP[initial]
        raise SystemExit(f"Modo invalido: {initial}. Use 1, 2, 3 ou as palavras pipeline/notebook/both.")

    menu = (
        "\nSelecione o modo de execucao:\n"
        "  1 - Treinamento CLI (pipeline)\n"
        "  2 - Gerar notebook Jupyter\n"
        "  3 - Executar pipeline e gerar notebook\n"
    )
    while True:
        try:
            choice = input(f"{menu}Opcao: ").strip()
        except EOFError:
            print("\nNenhuma opcao informada; usando 'pipeline'.")
            return "pipeline"
        if choice in MODE_MAP:
            return MODE_MAP[choice]
        if choice in MODE_MAP.values():
            return choice
        print("Opcao invalida. Digite 1, 2 ou 3.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento para deteccao de fraudes em cartoes.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("creditcard - menor balanceado.csv"),
        help="Caminho para o CSV com os dados.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Diretorio onde os relatorios e figuras serao gerados.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Nao gerar o relatorio HTML.",
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "notebook", "both"],
        help="Executa diretamente no modo escolhido (padrao: menu interativo).",
    )
    args = parser.parse_args()

    mode = prompt_mode_selection(args.mode)
    html_enabled = not args.no_html
    open_report = html_enabled

    if mode in {"pipeline", "both"}:
        context = "CLI - pipeline" if mode == "pipeline" else "CLI - pipeline + notebook"
        run_training_workflow(
            dataset_path=args.dataset,
            output_dir=args.output,
            generate_html=html_enabled,
            open_report=open_report,
            run_context=context,
        )

    if mode in {"notebook", "both"}:
        context = "CLI - notebook" if mode == "notebook" else "CLI - pipeline + notebook"
        generate_notebook(
            dataset_path=args.dataset,
            output_dir=args.output,
            run_context=context,
        )

    print("\nExecucao concluida.")
    if not args.no_html:
        report_path = (output_dir / "report.html").resolve()
        try:
            webbrowser.open_new_tab(report_path.as_uri())
        except Exception as exc:  # pragma: no cover - abrir browser nao deve quebrar execucao
            print(f"Aviso: nao foi possivel abrir automaticamente o relatorio ({exc})", file=sys.stderr)


if __name__ == "__main__":
    main()
