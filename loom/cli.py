"""Click-based CLI entry point for Loom."""

from __future__ import annotations

import json
from pathlib import Path

import click
from dotenv import load_dotenv

from loom.schema.loader import ConfigError, load_config

# Load .env file so API keys are available without manual export.
load_dotenv()


@click.group()
@click.version_option(package_name="loom")
def cli() -> None:
    """Loom — runtime scaffolding for faithful long-horizon persona simulation."""


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str) -> None:
    """Validate a YAML persona config file."""
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    n_dims = len(cfg.trajectory.dimensions)
    n_phases = len(cfg.trajectory.phases)
    click.echo(
        f"Config valid: {cfg.persona.identity.name} "
        f"({n_dims} dimensions, {n_phases} phases, "
        f"mode={cfg.trajectory.mode})"
    )


@cli.command(name="dry-run")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--turns", default=10, help="Number of turns to simulate.")
@click.option(
    "--json-output", is_flag=True, default=False,
    help="Output as JSON instead of human-readable text.",
)
def dry_run(config_path: str, turns: int, json_output: bool) -> None:
    """Show assembled prompts at each turn without making API calls."""
    from loom.assembler.prompt_builder import PromptBuilder

    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    builder = PromptBuilder(cfg)
    expected = cfg.trajectory.expected_turns or turns

    all_turns: list[dict] = []
    for t in range(min(turns, expected)):
        blocks, state = builder.build(t)

        turn_data = {
            "turn": t,
            "phase": state.phase,
            "intensities": state.intensities,
            "injection_type": str(state.injection_type),
            "must_do": blocks.must_do,
            "how": blocks.how,
            "context": blocks.context,
        }

        if json_output:
            all_turns.append(turn_data)
        else:
            click.echo(f"\n{'='*72}")
            click.echo(
                f"TURN {t}  |  Phase: {state.phase}  |  "
                f"Injection: {state.injection_type}"
            )
            click.echo(
                "Intensities: "
                + ", ".join(f"{k}={v:.2f}" for k, v in state.intensities.items())
            )
            click.echo(f"{'='*72}")

            if state.injection_type == "full":
                click.echo("\n--- MUST DO ---")
                click.echo(blocks.must_do)
                click.echo("\n--- HOW ---")
                click.echo(blocks.how)
                click.echo("\n--- CONTEXT ---")
                click.echo(blocks.context)
            elif state.injection_type == "reminder":
                reminder = builder.format_reminder(state)
                click.echo(f"\n--- REMINDER (appended to user message) ---")
                click.echo(reminder)
            else:
                click.echo("(no injection this turn)")

    if json_output:
        click.echo(json.dumps(all_turns, indent=2))

    click.echo(f"\nDry-run complete: {min(turns, expected)} turns simulated.")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--target", required=True, help="Interlocutor model (e.g. openai/gpt-4.1).")
@click.option("--persona-model", required=True, help="Persona model (e.g. openai/gpt-4.1).")
@click.option("--turns", default=None, type=int, help="Override expected_turns from config.")
@click.option("--rollouts", default=1, help="Number of rollouts to produce.")
@click.option("--adversary", default="none", help="Adversary type: none, contradiction, helpfulness.")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
@click.option("--output", required=True, type=click.Path(), help="Output directory for transcripts.")
@click.option("--score", is_flag=True, default=False, help="Run scoring inline after rollout.")
def run(
    config_path: str,
    target: str,
    persona_model: str,
    turns: int | None,
    rollouts: int,
    adversary: str,
    seed: int | None,
    output: str,
    score: bool,
) -> None:
    """Execute multi-turn persona rollouts."""
    from loom.runner.rollout import RolloutRunner

    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_turns = turns or cfg.trajectory.expected_turns or 20

    runner = RolloutRunner(
        config=cfg,
        target_model=target,
        persona_model=persona_model,
        seed=seed,
    )

    for i in range(rollouts):
        rollout_seed = (seed or 0) + i
        click.echo(f"\nRollout {i}/{rollouts} (seed={rollout_seed})...")
        transcript = runner.execute(
            n_turns=n_turns,
            seed=rollout_seed,
        )
        out_path = output_dir / f"rollout_{i:03d}.json"
        out_path.write_text(
            json.dumps(transcript.to_dict(), indent=2),
            encoding="utf-8",
        )
        click.echo(f"  Saved: {out_path}")

    click.echo(f"\nDone. {rollouts} rollout(s) saved to {output_dir}")
