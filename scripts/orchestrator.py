#!/usr/bin/env python3
"""
Context Rot Research Orchestrator
Created: 2025-12-18

Configuration-driven workflow orchestrator that replaces run_full_research.sh.
Reads model and experiment definitions from YAML config files.

Usage:
    python scripts/orchestrator.py                      # Run all enabled
    python scripts/orchestrator.py --test               # Test mode
    python scripts/orchestrator.py --models gpt-oss-20b # Specific model
    python scripts/orchestrator.py --experiments niah   # Specific experiment
    python scripts/orchestrator.py --list               # Show available models/experiments
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = typer.Typer(
    name="orchestrator",
    help="Context Rot Research Orchestrator - Run experiments across models",
    add_completion=False
)
console = Console()


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path: Path) -> dict:
    """Load and validate YAML configuration."""
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required = ["models", "experiments", "settings"]
    for section in required:
        if section not in config:
            console.print(f"[red]Missing required config section: {section}[/red]")
            raise typer.Exit(1)

    return config


def get_enabled_models(config: dict, model_filter: Optional[list[str]] = None) -> dict:
    """Get models to run, filtered by enabled flag and optional filter list."""
    models = {}
    for name, spec in config["models"].items():
        if model_filter and name not in model_filter:
            continue
        if spec.get("enabled", True) or model_filter:
            models[name] = spec
    return models


def get_enabled_experiments(config: dict, exp_filter: Optional[list[str]] = None) -> dict:
    """Get experiments to run, filtered by enabled flag and optional filter list."""
    experiments = {}
    for name, spec in config["experiments"].items():
        if exp_filter and name not in exp_filter:
            continue
        if spec.get("enabled", True) or exp_filter:
            experiments[name] = spec
    return experiments


# =============================================================================
# WORKFLOW STATE MANAGEMENT
# =============================================================================

class WorkflowState:
    """Track completion of workflow steps for checkpoint/resume."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self._load()

    def _load(self) -> dict:
        default = {"completed": {}, "started_at": None}
        if self.state_file.exists():
            with open(self.state_file) as f:
                loaded = json.load(f)
            # Merge with defaults to handle partial/legacy state files
            for key in default:
                if key not in loaded:
                    loaded[key] = default[key]
            return loaded
        return default

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def is_complete(self, experiment: str, model: str, step: str) -> bool:
        key = f"{experiment}:{model}:{step}"
        return self.state["completed"].get(key, False)

    def mark_complete(self, experiment: str, model: str, step: str):
        key = f"{experiment}:{model}:{step}"
        self.state["completed"][key] = True
        self.save()

    def start_run(self):
        if not self.state.get("started_at"):
            self.state["started_at"] = datetime.now().isoformat()
            self.save()


# =============================================================================
# STEP EXECUTION
# =============================================================================

def run_python_script(
    script: str,
    args: list[str],
    cwd: Path,
    env_override: Optional[dict] = None
) -> bool:
    """Run a Python script with arguments. Returns success status."""
    cmd = [sys.executable, script] + args

    env = os.environ.copy()
    if env_override:
        env.update(env_override)

    console.print(f"[dim]$ {' '.join(cmd[:6])}...[/dim]")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=False,  # Let output stream to terminal
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        console.print(f"[red]Error running script: {e}[/red]")
        return False


def build_inference_args(
    model_spec: dict,
    exp_spec: dict,
    input_path: str,
    output_path: str,
    input_column: str,
    test_mode: bool,
    truncate: bool = False
) -> list[str]:
    """Build command-line arguments for inference script."""
    args = [
        "--provider", model_spec["provider"],
        "--model-name", model_spec["model_name"],
        "--input-path", input_path,
        "--output-path", output_path,
        "--input-column", input_column,
        "--output-column", exp_spec.get("inference", {}).get("output_column", "output"),
        "--max-context-length", str(model_spec["context_length"]),
        "--max-tokens-per-minute", str(model_spec["rate_limit"]),
    ]

    if test_mode:
        args.append("--test-mode")
    if truncate:
        args.append("--truncate-to-fit")

    # Experiment-specific args
    inference_spec = exp_spec.get("inference", {})
    if "common_word" in inference_spec:
        args.extend(["--common-word", inference_spec["common_word"]])
    if "modified_word" in inference_spec:
        args.extend(["--modified-word", inference_spec["modified_word"]])
    if "model_max_output_tokens" in inference_spec:
        args.extend(["--model-max-output-tokens", str(inference_spec["model_max_output_tokens"])])

    return args


def build_evaluation_args(
    exp_spec: dict,
    input_path: str,
    output_path: str,
    settings: dict
) -> list[str]:
    """Build command-line arguments for evaluation script."""
    eval_spec = exp_spec.get("evaluation", {})

    args = [
        "--input-path", input_path,
        "--output-path", output_path,
        "--model-name", eval_spec.get("judge_model", settings["default_judge_model"]),
    ]

    for key in ["output_column", "question_column", "correct_answer_column"]:
        if key in eval_spec:
            args.extend([f"--{key.replace('_', '-')}", eval_spec[key]])

    return args


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_niah_experiment(
    model_name: str,
    model_spec: dict,
    exp_spec: dict,
    config: dict,
    state: WorkflowState,
    test_mode: bool,
    file_prefix: str
):
    """Run NIAH extension experiment for a model."""
    settings = config["settings"]
    results_dir = PROJECT_ROOT / settings["results_dir"]

    # Step 1: Generate haystacks (if needed)
    gen_spec = exp_spec.get("generate", {})
    if gen_spec:
        output_check = PROJECT_ROOT / gen_spec.get("output_check", "")
        if not output_check.exists() or test_mode:
            console.print("[cyan]Generating haystacks...[/cyan]")
            args = []
            for key, val in gen_spec.get("args", {}).items():
                args.extend([f"--{key.replace('_', '-')}", str(PROJECT_ROOT / val) if "/" in str(val) else str(val)])
            if test_mode:
                args.append("--test-mode")

            success = run_python_script(
                gen_spec["script"],
                args,
                PROJECT_ROOT
            )
            if not success:
                console.print("[red]Haystack generation failed[/red]")
                return

    # Step 2: Inference
    if not state.is_complete("niah", model_name, "inference"):
        console.print(f"[cyan]Running NIAH inference for {model_name}...[/cyan]")

        inf_spec = exp_spec["inference"]
        output_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_niah_results.csv"

        args = build_inference_args(
            model_spec,
            exp_spec,
            str(PROJECT_ROOT / inf_spec["input_path"]),
            str(output_path),
            inf_spec["input_column"],
            test_mode
        )

        success = run_python_script(inf_spec["script"], args, PROJECT_ROOT, model_spec.get("env_override"))
        if success:
            state.mark_complete("niah", model_name, "inference")
        else:
            console.print("[red]Inference failed[/red]")
            return

    # Step 3: Evaluation
    if not state.is_complete("niah", model_name, "evaluation"):
        console.print(f"[cyan]Evaluating NIAH results for {model_name}...[/cyan]")

        eval_spec = exp_spec["evaluation"]
        input_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_niah_results.csv"
        output_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_niah_evaluated.csv"

        args = build_evaluation_args(exp_spec, str(input_path), str(output_path), settings)
        success = run_python_script(eval_spec["script"], args, PROJECT_ROOT)

        if success:
            state.mark_complete("niah", model_name, "evaluation")

    # Step 4: Visualization
    if not state.is_complete("niah", model_name, "visualization"):
        console.print(f"[cyan]Generating NIAH heatmap for {model_name}...[/cyan]")

        vis_spec = exp_spec["visualization"]
        input_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_niah_evaluated.csv"
        output_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_niah_heatmap.png"

        args = [
            "--csv-path", str(input_path),
            "--output-path", str(output_path),
            "--title", f"NIAH Performance - {model_spec.get('display_name', model_name)}"
        ]

        success = run_python_script(vis_spec["script"], args, PROJECT_ROOT)
        if success:
            state.mark_complete("niah", model_name, "visualization")


def run_longmemeval_experiment(
    model_name: str,
    model_spec: dict,
    exp_spec: dict,
    config: dict,
    state: WorkflowState,
    test_mode: bool,
    file_prefix: str
):
    """Run LongMemEval experiment for a model."""
    settings = config["settings"]
    results_dir = PROJECT_ROOT / settings["results_dir"]

    for variant_name, variant_spec in exp_spec.get("variants", {}).items():
        # Inference
        step_key = f"{variant_name}_inference"
        if not state.is_complete("longmemeval", model_name, step_key):
            console.print(f"[cyan]Running LongMemEval {variant_name} inference for {model_name}...[/cyan]")

            inf_spec = exp_spec["inference"]
            output_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_longmemeval_{variant_name}_results.csv"

            args = build_inference_args(
                model_spec,
                exp_spec,
                str(PROJECT_ROOT / variant_spec["input_path"]),
                str(output_path),
                variant_spec["input_column"],
                test_mode,
                truncate=variant_spec.get("truncate_to_fit", False)
            )

            success = run_python_script(inf_spec["script"], args, PROJECT_ROOT, model_spec.get("env_override"))
            if success:
                state.mark_complete("longmemeval", model_name, step_key)

        # Evaluation
        step_key = f"{variant_name}_evaluation"
        if not state.is_complete("longmemeval", model_name, step_key):
            console.print(f"[cyan]Evaluating LongMemEval {variant_name} for {model_name}...[/cyan]")

            eval_spec = exp_spec["evaluation"]
            input_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_longmemeval_{variant_name}_results.csv"
            output_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_longmemeval_{variant_name}_evaluated.csv"

            args = build_evaluation_args(exp_spec, str(input_path), str(output_path), settings)
            success = run_python_script(eval_spec["script"], args, PROJECT_ROOT)

            if success:
                state.mark_complete("longmemeval", model_name, step_key)

    # Visualization (after both variants complete)
    if not state.is_complete("longmemeval", model_name, "visualization"):
        console.print(f"[cyan]Generating LongMemEval comparison for {model_name}...[/cyan]")

        vis_spec = exp_spec["visualization"]
        focused_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_longmemeval_focused_evaluated.csv"
        full_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_longmemeval_full_evaluated.csv"
        output_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_longmemeval.png"

        args = [
            "--focused-path", str(focused_path),
            "--full-path", str(full_path),
            "--model-name", model_spec.get("display_name", model_name),
            "--output-path", str(output_path)
        ]

        success = run_python_script(vis_spec["script"], args, PROJECT_ROOT)
        if success:
            state.mark_complete("longmemeval", model_name, "visualization")


def run_repeated_words_experiment(
    model_name: str,
    model_spec: dict,
    exp_spec: dict,
    config: dict,
    state: WorkflowState,
    test_mode: bool,
    file_prefix: str
):
    """Run Repeated Words experiment for a model."""
    settings = config["settings"]
    results_dir = PROJECT_ROOT / settings["results_dir"]
    inf_spec = exp_spec["inference"]

    common_word = inf_spec.get("common_word", "apple")
    modified_word = inf_spec.get("modified_word", "apples")

    # Inference
    if not state.is_complete("repeated_words", model_name, "inference"):
        console.print(f"[cyan]Running Repeated Words inference for {model_name}...[/cyan]")

        output_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_repeated_words_{common_word}_{modified_word}.csv"

        args = [
            "--provider", model_spec["provider"],
            "--model-name", model_spec["model_name"],
            "--output-path", str(output_path),
            "--common-word", common_word,
            "--modified-word", modified_word,
            "--model-max-output-tokens", str(inf_spec.get("model_max_output_tokens", 32768)),
            "--max-context-length", str(model_spec["context_length"]),
            "--max-tokens-per-minute", str(model_spec["rate_limit"]),
        ]
        if test_mode:
            args.append("--test-mode")

        success = run_python_script(inf_spec["script"], args, PROJECT_ROOT, model_spec.get("env_override"))
        if success:
            state.mark_complete("repeated_words", model_name, "inference")

    # Evaluation
    if not state.is_complete("repeated_words", model_name, "evaluation"):
        console.print(f"[cyan]Evaluating Repeated Words for {model_name}...[/cyan]")

        eval_spec = exp_spec["evaluation"]
        input_path = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_repeated_words_{common_word}_{modified_word}.csv"
        output_dir = results_dir / f"{file_prefix}{model_name.replace('-', '_')}_repeated_words_{common_word}_{modified_word}_evaluated"

        args = [
            "--input-path", str(input_path),
            "--output-dir", str(output_dir),
            "--common-word", common_word,
            "--modified-word", modified_word,
            "--model-name", model_spec.get("display_name", model_name)
        ]

        success = run_python_script(eval_spec["script"], args, PROJECT_ROOT)
        if success:
            state.mark_complete("repeated_words", model_name, "evaluation")


# =============================================================================
# CLI COMMANDS
# =============================================================================

@app.command()
def run(
    config_file: Path = typer.Option(
        Path("config/research.yaml"),
        "--config", "-c",
        help="Path to config file"
    ),
    models: Optional[list[str]] = typer.Option(
        None,
        "--models", "-m",
        help="Specific models to run (can specify multiple)"
    ),
    experiments: Optional[list[str]] = typer.Option(
        None,
        "--experiments", "-e",
        help="Specific experiments to run (can specify multiple)"
    ),
    test_mode: bool = typer.Option(
        False,
        "--test", "-t",
        help="Run in test mode with reduced samples"
    ),
    truncate: bool = typer.Option(
        False,
        "--truncate", "-T",
        help="Truncate oversized prompts to fit context"
    ),
    reset_state: bool = typer.Option(
        False,
        "--reset",
        help="Reset workflow state and start fresh"
    )
):
    """Run context rot experiments."""
    config_path = PROJECT_ROOT / config_file
    config = load_config(config_path)
    settings = config["settings"]

    # Initialize workflow state
    state_file = PROJECT_ROOT / settings["state_file"]
    if reset_state and state_file.exists():
        state_file.unlink()
        console.print("[yellow]Workflow state reset[/yellow]")

    state = WorkflowState(state_file)
    state.start_run()

    # Get models and experiments to run
    enabled_models = get_enabled_models(config, models)
    enabled_experiments = get_enabled_experiments(config, experiments)

    if not enabled_models:
        console.print("[red]No models selected. Use --models or enable in config.[/red]")
        raise typer.Exit(1)

    if not enabled_experiments:
        console.print("[red]No experiments selected. Use --experiments or enable in config.[/red]")
        raise typer.Exit(1)

    # File prefix for test mode
    file_prefix = config.get("test_mode", {}).get("file_prefix", "test_") if test_mode else ""

    # Display run info
    console.print(Panel(
        f"[bold]Context Rot Research[/bold]\n\n"
        f"Models: {', '.join(enabled_models.keys())}\n"
        f"Experiments: {', '.join(enabled_experiments.keys())}\n"
        f"Test Mode: {test_mode}",
        title="Configuration"
    ))

    # Run experiments
    for exp_name, exp_spec in enabled_experiments.items():
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold blue]Experiment: {exp_spec.get('display_name', exp_name)}[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]\n")

        for model_name, model_spec in enabled_models.items():
            console.print(f"\n[bold cyan]Model: {model_spec.get('display_name', model_name)}[/bold cyan]\n")

            try:
                if exp_name == "niah":
                    run_niah_experiment(model_name, model_spec, exp_spec, config, state, test_mode, file_prefix)
                elif exp_name == "longmemeval":
                    run_longmemeval_experiment(model_name, model_spec, exp_spec, config, state, test_mode, file_prefix)
                elif exp_name == "repeated_words":
                    run_repeated_words_experiment(model_name, model_spec, exp_spec, config, state, test_mode, file_prefix)
                else:
                    console.print(f"[yellow]Unknown experiment type: {exp_name}[/yellow]")
            except Exception as e:
                console.print(f"[red]Error in {exp_name} for {model_name}: {e}[/red]")
                continue

    console.print("\n[bold green]All experiments complete![/bold green]")
    console.print(f"Results saved to: {settings['results_dir']}/")


@app.command()
def list_available(
    config_file: Path = typer.Option(
        Path("config/research.yaml"),
        "--config", "-c",
        help="Path to config file"
    )
):
    """List available models and experiments."""
    config_path = PROJECT_ROOT / config_file
    config = load_config(config_path)

    # Models table
    console.print("\n[bold]Available Models[/bold]\n")
    models_table = Table(show_header=True)
    models_table.add_column("Name")
    models_table.add_column("Provider")
    models_table.add_column("Context")
    models_table.add_column("Enabled")
    models_table.add_column("Description")

    for name, spec in config["models"].items():
        enabled = "[green]Yes[/green]" if spec.get("enabled", True) else "[dim]No[/dim]"
        models_table.add_row(
            name,
            spec["provider"],
            f"{spec['context_length']:,}",
            enabled,
            spec.get("description", "")
        )

    console.print(models_table)

    # Experiments table
    console.print("\n[bold]Available Experiments[/bold]\n")
    exp_table = Table(show_header=True)
    exp_table.add_column("Name")
    exp_table.add_column("Enabled")
    exp_table.add_column("Description")

    for name, spec in config["experiments"].items():
        enabled = "[green]Yes[/green]" if spec.get("enabled", True) else "[dim]No[/dim]"
        exp_table.add_row(
            name,
            enabled,
            spec.get("description", "")
        )

    console.print(exp_table)

    # Providers info
    console.print("\n[bold]Registered Providers[/bold]\n")
    try:
        from experiments.models.registry import provider_info
        info = provider_info()
        prov_table = Table(show_header=True)
        prov_table.add_column("Name")
        prov_table.add_column("Aliases")
        prov_table.add_column("Description")

        for name, details in info.items():
            aliases = ", ".join(details["aliases"]) if details["aliases"] else "-"
            prov_table.add_row(name, aliases, details["docstring"])

        console.print(prov_table)
    except ImportError:
        console.print("[dim]Could not load provider registry[/dim]")


@app.command()
def status(
    config_file: Path = typer.Option(
        Path("config/research.yaml"),
        "--config", "-c",
        help="Path to config file"
    )
):
    """Show workflow status and progress."""
    config_path = PROJECT_ROOT / config_file
    config = load_config(config_path)
    settings = config["settings"]

    state_file = PROJECT_ROOT / settings["state_file"]
    if not state_file.exists():
        console.print("[yellow]No workflow state found. Run 'orchestrator run' to start.[/yellow]")
        return

    state = WorkflowState(state_file)

    console.print(Panel(
        f"Started: {state.state.get('started_at', 'Unknown')}\n"
        f"Completed steps: {len(state.state.get('completed', {}))}",
        title="Workflow Status"
    ))

    # Show completion matrix
    table = Table(show_header=True, title="Completion Status")
    table.add_column("Experiment")
    table.add_column("Model")
    table.add_column("Step")
    table.add_column("Status")

    for key, completed in state.state.get("completed", {}).items():
        parts = key.split(":")
        if len(parts) == 3:
            exp, model, step = parts
            status = "[green]Complete[/green]" if completed else "[yellow]Pending[/yellow]"
            table.add_row(exp, model, step, status)

    console.print(table)


if __name__ == "__main__":
    app()
