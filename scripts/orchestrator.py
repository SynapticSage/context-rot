#!/usr/bin/env python3
"""
Context Rot Research Orchestrator
Created: 2025-12-18
Modified: 2025-12-22 (added generate command with datasets.yaml config)

Configuration-driven workflow orchestrator that replaces run_full_research.sh.
Reads model and experiment definitions from YAML config files.

Usage:
    python scripts/orchestrator.py generate             # Generate all datasets
    python scripts/orchestrator.py generate niah        # Generate NIAH dataset
    python scripts/orchestrator.py run                  # Run all enabled experiments
    python scripts/orchestrator.py run --test           # Test mode
    python scripts/orchestrator.py run --models gpt-oss-20b
    python scripts/orchestrator.py list                 # Show available models/experiments
    python scripts/orchestrator.py full                 # Generate + Run + Evaluate
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

def load_config(config_path: Path, required_sections: list[str] = None) -> dict:
    """Load and validate YAML configuration."""
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required sections
    if required_sections is None:
        required_sections = ["models", "experiments", "settings"]
    for section in required_sections:
        if section not in config:
            console.print(f"[red]Missing required config section: {section}[/red]")
            raise typer.Exit(1)

    return config


def load_datasets_config(config_path: Path = None) -> dict:
    """Load datasets.yaml configuration."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "datasets.yaml"
    return load_config(config_path, required_sections=[])


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
        "--model", model_spec["model_name"],
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
        "--model", eval_spec.get("judge_model", settings["default_judge_model"]),
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
            "--model", model_spec.get("display_name", model_name),
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
            "--model", model_spec["model_name"],
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
            "--model", model_spec.get("display_name", model_name)
        ]

        success = run_python_script(eval_spec["script"], args, PROJECT_ROOT)
        if success:
            state.mark_complete("repeated_words", model_name, "evaluation")


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_niah_dataset(
    datasets_config: dict,
    test_mode: bool = False,
    force: bool = False
) -> bool:
    """Generate NIAH haystacks from datasets.yaml config."""
    niah = datasets_config.get("niah_extension", {})
    if not niah:
        console.print("[yellow]No niah_extension config found in datasets.yaml[/yellow]")
        return False

    defaults = datasets_config.get("defaults", {})
    output_dir = PROJECT_ROOT / defaults.get("output_dir", "data") / "niah_prompts"

    # Determine output file and check if exists
    suffix = "shuffled" if niah.get("shuffled", False) else "sequential"
    output_file = output_dir / f"niah_prompts_{suffix}.csv"

    if output_file.exists() and not force:
        console.print(f"[dim]Dataset exists: {output_file.relative_to(PROJECT_ROOT)}[/dim]")
        console.print("[dim]Use --force to regenerate[/dim]")
        return True

    # Build arguments from config
    if test_mode:
        test_config = niah.get("test", {})
        trials = test_config.get("trials_per_cell", defaults.get("test_trials_per_cell", 1))
    else:
        trials = niah.get("trials_per_cell", defaults.get("trials_per_cell", 5))

    args = [
        "--haystack-folder", str(PROJECT_ROOT / niah["haystack_folder"]),
        "--needle", niah["needle"].strip(),
        "--question", niah["question"],
        "--output-folder", str(output_dir),
        "--trials-per-cell", str(trials),
    ]

    if niah.get("shuffled", False):
        args.append("--shuffled")

    if niah.get("distractors"):
        args.append("--distractors")
        args.extend(niah["distractors"])

    if test_mode:
        args.append("--test-mode")

    console.print(f"[cyan]Generating NIAH dataset ({trials} trials/cell)...[/cyan]")

    success = run_python_script(
        "experiments/niah_extension/run/create_haystacks.py",
        args,
        PROJECT_ROOT
    )

    if success:
        console.print(f"[green]Generated: {output_file.relative_to(PROJECT_ROOT)}[/green]")
    return success


def generate_repeated_words_data(
    datasets_config: dict,
    test_mode: bool = False
) -> dict:
    """Get repeated words config (data is generated inline during run)."""
    rw = datasets_config.get("repeated_words", {})
    return {
        "common_word": rw.get("common_word", "apple"),
        "modified_word": rw.get("modified_word", "apples"),
        "context_lengths": rw.get("context_lengths", [1000, 2000, 4000, 8000, 16000, 32000])
    }


# =============================================================================
# CLI COMMANDS
# =============================================================================

@app.command()
def generate(
    experiment: Optional[str] = typer.Argument(
        None,
        help="Specific experiment to generate data for (niah, longmemeval, repeated_words)"
    ),
    config_file: Path = typer.Option(
        Path("config/datasets.yaml"),
        "--config", "-c",
        help="Path to datasets config file"
    ),
    test_mode: bool = typer.Option(
        False,
        "--test", "-t",
        help="Generate test-sized datasets"
    ),
    trials: Optional[int] = typer.Option(
        None,
        "--trials", "-n",
        help="Override trials per cell (NIAH only)"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Regenerate even if data exists"
    )
):
    """Generate experiment datasets from config.

    Examples:
        orchestrator generate                    # Generate all datasets
        orchestrator generate niah              # Generate NIAH only
        orchestrator generate niah --trials 30  # 30 trials per cell
        orchestrator generate --test            # Test-sized datasets
    """
    config_path = PROJECT_ROOT / config_file
    datasets_config = load_datasets_config(config_path)

    # Override trials if specified
    if trials and "niah_extension" in datasets_config:
        datasets_config["niah_extension"]["trials_per_cell"] = trials

    experiments_to_generate = []
    if experiment:
        # Map aliases
        exp_map = {
            "niah": "niah_extension",
            "niah_extension": "niah_extension",
            "longmemeval": "longmemeval",
            "repeated_words": "repeated_words",
            "rw": "repeated_words"
        }
        if experiment.lower() not in exp_map:
            console.print(f"[red]Unknown experiment: {experiment}[/red]")
            console.print(f"Available: {', '.join(exp_map.keys())}")
            raise typer.Exit(1)
        experiments_to_generate = [exp_map[experiment.lower()]]
    else:
        experiments_to_generate = ["niah_extension", "longmemeval", "repeated_words"]

    console.print(Panel(
        f"[bold]Dataset Generation[/bold]\n\n"
        f"Experiments: {', '.join(experiments_to_generate)}\n"
        f"Test Mode: {test_mode}\n"
        f"Force: {force}",
        title="Configuration"
    ))

    results = {}

    for exp in experiments_to_generate:
        if exp == "niah_extension":
            results[exp] = generate_niah_dataset(datasets_config, test_mode, force)
        elif exp == "longmemeval":
            console.print("[dim]LongMemEval uses pre-existing data (no generation needed)[/dim]")
            results[exp] = True
        elif exp == "repeated_words":
            console.print("[dim]Repeated Words generates data inline during run[/dim]")
            rw_config = generate_repeated_words_data(datasets_config, test_mode)
            console.print(f"[dim]  Words: {rw_config['common_word']} / {rw_config['modified_word']}[/dim]")
            results[exp] = True

    # Summary
    all_success = all(results.values())
    if all_success:
        console.print("\n[bold green]Dataset generation complete![/bold green]")
    else:
        failed = [k for k, v in results.items() if not v]
        console.print(f"\n[bold red]Some datasets failed: {', '.join(failed)}[/bold red]")
        raise typer.Exit(1)


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


def get_csv_path_for_step(exp: str, model: str, step: str, results_dir: Path) -> Optional[Path]:
    """Map experiment/model/step to the corresponding CSV file."""
    model_slug = model.replace("-", "_")

    # Map step names to file patterns
    if exp == "niah":
        if step == "inference":
            return results_dir / f"{model_slug}_niah_results.csv"
        elif step == "evaluation":
            return results_dir / f"{model_slug}_niah_evaluated.csv"
    elif exp == "longmemeval":
        if step == "focused_inference":
            return results_dir / f"{model_slug}_longmemeval_focused_results.csv"
        elif step == "full_inference":
            return results_dir / f"{model_slug}_longmemeval_full_results.csv"
        elif step == "focused_evaluation":
            return results_dir / f"{model_slug}_longmemeval_focused_evaluated.csv"
        elif step == "full_evaluation":
            return results_dir / f"{model_slug}_longmemeval_full_evaluated.csv"
    elif exp == "repeated_words":
        if step == "inference":
            return results_dir / f"{model_slug}_repeated_words_apple_apples.csv"

    return None


def get_error_stats(csv_path: Path, output_column: str = "output") -> Optional[str]:
    """Calculate error/empty percentage from a CSV file."""
    if not csv_path.exists():
        return None

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)

        if output_column not in df.columns:
            return None

        total = len(df)
        if total == 0:
            return "0/0"

        errors = df[output_column].astype(str).str.startswith('ERROR').sum()
        empty = df[output_column].isna().sum()
        failed = errors + empty

        if failed == 0:
            return "[green]0%[/green]"

        pct = (failed / total) * 100
        color = "red" if pct > 10 else "yellow"
        return f"[{color}]{failed}/{total} ({pct:.1f}%)[/{color}]"
    except Exception:
        return "[dim]?[/dim]"


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
    results_dir = PROJECT_ROOT / settings["results_dir"]

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
    table.add_column("Errors")

    for key, completed in state.state.get("completed", {}).items():
        parts = key.split(":")
        if len(parts) == 3:
            exp, model, step = parts
            status_str = "[green]Complete[/green]" if completed else "[yellow]Pending[/yellow]"

            # Get error stats for inference/evaluation steps
            csv_path = get_csv_path_for_step(exp, model, step, results_dir)
            error_str = get_error_stats(csv_path) if csv_path else "[dim]-[/dim]"

            table.add_row(exp, model, step, status_str, error_str or "[dim]-[/dim]")

    console.print(table)


@app.command()
def full(
    research_config: Path = typer.Option(
        Path("config/research.yaml"),
        "--research-config",
        help="Path to research config file"
    ),
    datasets_config: Path = typer.Option(
        Path("config/datasets.yaml"),
        "--datasets-config",
        help="Path to datasets config file"
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
    trials: Optional[int] = typer.Option(
        None,
        "--trials", "-n",
        help="Override trials per cell for NIAH"
    ),
    force_regenerate: bool = typer.Option(
        False,
        "--force-generate", "-f",
        help="Force regenerate datasets even if they exist"
    ),
    reset_state: bool = typer.Option(
        False,
        "--reset",
        help="Reset workflow state and start fresh"
    )
):
    """Full pipeline: generate datasets + run experiments + evaluate.

    This is the recommended way to run a complete research workflow.

    Examples:
        orchestrator full                           # Full run, all experiments
        orchestrator full --test                    # Quick validation
        orchestrator full --trials 30              # 30 trials per NIAH cell
        orchestrator full -m gpt-oss-20b -e niah   # Specific model/experiment
    """
    console.print(Panel(
        "[bold]Full Research Pipeline[/bold]\n\n"
        "Step 1: Generate datasets\n"
        "Step 2: Run inference\n"
        "Step 3: Evaluate results\n"
        "Step 4: Generate visualizations",
        title="Workflow"
    ))

    # Step 1: Generate datasets
    console.print("\n[bold blue]Step 1: Dataset Generation[/bold blue]\n")

    ds_config_path = PROJECT_ROOT / datasets_config
    ds_config = load_datasets_config(ds_config_path)

    if trials and "niah_extension" in ds_config:
        ds_config["niah_extension"]["trials_per_cell"] = trials

    # Map experiment filter to dataset names
    exp_to_dataset = {
        "niah": "niah_extension",
        "longmemeval": "longmemeval",
        "repeated_words": "repeated_words"
    }

    datasets_to_gen = []
    if experiments:
        for exp in experiments:
            if exp in exp_to_dataset:
                datasets_to_gen.append(exp_to_dataset[exp])
    else:
        datasets_to_gen = list(exp_to_dataset.values())

    for ds in datasets_to_gen:
        if ds == "niah_extension":
            generate_niah_dataset(ds_config, test_mode, force_regenerate)
        elif ds == "longmemeval":
            console.print("[dim]LongMemEval: Using pre-existing data[/dim]")
        elif ds == "repeated_words":
            console.print("[dim]Repeated Words: Data generated inline[/dim]")

    # Step 2-4: Run experiments (uses the run command logic)
    console.print("\n[bold blue]Step 2-4: Inference, Evaluation, Visualization[/bold blue]\n")

    # Load research config and run
    config_path = PROJECT_ROOT / research_config
    config = load_config(config_path)
    settings = config["settings"]

    # Initialize workflow state
    state_file = PROJECT_ROOT / settings["state_file"]
    if reset_state and state_file.exists():
        state_file.unlink()
        console.print("[yellow]Workflow state reset[/yellow]")

    state = WorkflowState(state_file)
    state.start_run()

    enabled_models = get_enabled_models(config, models)
    enabled_experiments = get_enabled_experiments(config, experiments)

    if not enabled_models:
        console.print("[red]No models selected[/red]")
        raise typer.Exit(1)

    if not enabled_experiments:
        console.print("[red]No experiments selected[/red]")
        raise typer.Exit(1)

    file_prefix = config.get("test_mode", {}).get("file_prefix", "test_") if test_mode else ""

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
            except Exception as e:
                console.print(f"[red]Error in {exp_name} for {model_name}: {e}[/red]")
                continue

    console.print("\n[bold green]Full pipeline complete![/bold green]")
    console.print(f"Results saved to: {settings['results_dir']}/")


if __name__ == "__main__":
    app()
