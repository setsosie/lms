"""CLI entry point for LMS experiments."""

import argparse
import asyncio
import json
import math
from datetime import datetime
from pathlib import Path

from lms.agent import Agent
from lms.config import Config
from lms.goals import get_goal, list_goals, Goal
from lms.lean.mock import MockLeanVerifier
from lms.lean.mcp import MCPLeanVerifier
from lms.lean.real import RealLeanVerifier
from lms.metrics import analyze_library, print_analysis
from lms.prompts import get_all_versions
from lms.providers import create_provider
from lms.society import Society, BudgetExceeded


async def run_experiment(
    n_agents: int,
    n_generations: int,
    provider_name: str,
    output_dir: Path,
    config: Config,
    max_tokens: int | None = None,
    use_real_verifier: bool = False,
    mixed_providers: bool = False,
    goal: Goal | None = None,
    verifier_type: str = "mock",
    iterative_mode: bool = False,
    max_attempts: int = 5,
) -> None:
    """Run an LMS experiment.

    Args:
        n_agents: Number of agents
        n_generations: Number of generations
        provider_name: LLM provider name (ignored if mixed_providers=True)
        output_dir: Directory to save results
        config: Configuration object
        max_tokens: Maximum tokens to use (None = unlimited)
        use_real_verifier: Use real LEAN 4 verifier instead of mock (deprecated, use verifier_type)
        mixed_providers: Use all available providers (one agent per provider)
        goal: Optional goal to work towards
        verifier_type: Verifier to use: "mock", "real", or "mcp"
        iterative_mode: If True, each agent gets multiple verification attempts
        max_attempts: Max attempts per agent in iterative mode
    """
    print(f"\nStarting LMS Experiment")
    print(f"  Agents: {n_agents}")
    print(f"  Generations: {n_generations}")

    # Choose verifier
    # Path to the Lean project for auto-rebuild on new imports
    lean_project_dir = Path(__file__).parent.parent / "lean"

    if use_real_verifier or verifier_type == "real":
        verifier = RealLeanVerifier(project_dir=lean_project_dir)
        verifier_name = "LEAN 4 (direct)"
    elif verifier_type == "mcp":
        verifier = MCPLeanVerifier()
        verifier_name = "LEAN 4 (MCP)"
    else:
        verifier = MockLeanVerifier()
        verifier_name = "Mock"

    # Show goal if present
    if goal:
        print(f"  Goal: {goal.name}")
        print(f"  Definitions: {len(goal.definitions)} items")

    # Create providers
    if mixed_providers:
        available = config.available_providers()
        if len(available) < n_agents:
            raise ValueError(
                f"Mixed mode requires {n_agents} providers but only {len(available)} configured: {available}"
            )
        # Use first n_agents providers
        provider_names = available[:n_agents]
        providers = [
            create_provider(name, config.get_provider_config(name))
            for name in provider_names
        ]

        # Create society with multiple providers
        society = Society(
            n_agents=n_agents,
            providers=providers,
            verifier=verifier,
            max_tokens=max_tokens,
            goal=goal,
        )
        provider_name = "mixed"  # For metadata
    else:
        # Single provider mode
        provider_config = config.get_provider_config(provider_name)
        provider = create_provider(provider_name, provider_config)

        # Create society
        society = Society(
            n_agents=n_agents,
            provider=provider,
            verifier=verifier,
            max_tokens=max_tokens,
            goal=goal,
        )

    # Enable iterative mode if requested
    if iterative_mode:
        society.iterative_mode = True
        society.max_attempts = max_attempts

    # Print society members (agents)
    print(f"  Society Members:")
    for i, agent in enumerate(society.agents):
        print(f"    - {agent.id}: {society.providers[i].config.model}")

    print(f"  Output: {output_dir}")
    if max_tokens:
        print(f"  Budget: {max_tokens:,} tokens")
    print(f"  Verifier: {verifier_name}")
    if iterative_mode:
        print(f"  Mode: ITERATIVE ({max_attempts} attempts/agent)")
    print()

    # Run experiment
    print("Running generations...")
    checkpoint_interval = 5  # Save every N generations
    try:
        for gen in range(n_generations):
            print(f"  Generation {gen + 1}/{n_generations}...", end=" ", flush=True)
            result = await society.run_generation(gen)
            # Basic stats
            score = math.sqrt(result.artifacts_created * result.artifacts_verified) if result.artifacts_verified > 0 else 0
            output = (
                f"Created: {result.artifacts_created}, "
                f"Verified: {result.artifacts_verified}, "
                f"Score: {score:.1f}"
            )
            # Add review stats if peer review was used
            if result.reviews_total > 0:
                output += (
                    f", Reviews: {result.reviews_approved}A/{result.reviews_modified}M/{result.reviews_rejected}R"
                )
            output += f", Tokens: {result.tokens_used:,}"
            print(output)

            # Periodic checkpoint
            if (gen + 1) % checkpoint_interval == 0:
                output_dir.mkdir(parents=True, exist_ok=True)
                society.save(output_dir)
                print(f"    [Checkpoint saved at gen {gen + 1}]")
    except BudgetExceeded as e:
        print(f"\n  Budget exceeded: {e}")
        print("  Saving checkpoint for later resumption...")
    except Exception as e:
        print(f"\n  Error: {e}")
        print("  Saving checkpoint before exit...")
        output_dir.mkdir(parents=True, exist_ok=True)
        society.save(output_dir)

    # Analyze results
    analysis = analyze_library(society.library, society.results)
    print_analysis(analysis)

    # Print cumulative score
    total_created = len(society.library)
    total_verified = len(society.library.get_verified())
    cumulative_score = math.sqrt(total_created * total_verified) if total_verified > 0 else 0
    print(f"\nCollective Score: {cumulative_score:.1f}")
    print(f"  (sqrt({total_created} created × {total_verified} verified))")

    # Print goal progress if applicable
    if goal:
        print(f"\n{'='*50}")
        print("Goal Progress")
        print(f"{'='*50}")
        print(f"Progress: {goal.progress():.0%}")
        for defn in goal.definitions:
            status = "[DONE]" if defn.formalized else "[    ]"
            milestone = "MILESTONE: " if "MILESTONE" in defn.name else ""
            print(f"  {status} {milestone}{defn.tag}: {defn.name.split(':')[-1].strip()[:40]}")
        print(f"{'='*50}")

    # Print token summary
    print(f"\nToken Usage: {society.total_tokens_used:,} total")
    if max_tokens:
        print(f"  Budget: {max_tokens:,} ({100*society.total_tokens_used/max_tokens:.1f}% used)")
    # Per-agent breakdown
    if society.artifacts_by_agent:
        print(f"  Per-agent breakdown:")
        for agent_id in sorted(society.artifacts_by_agent.keys()):
            stats = society.artifacts_by_agent[agent_id]
            tokens = society.tokens_by_agent.get(agent_id, 0)
            ver_rate = 100 * stats["verified"] / stats["created"] if stats["created"] > 0 else 0
            reviews = society.reviews_by_agent.get(agent_id, {})
            print(f"    {agent_id}:")
            line = f"      Created: {stats['created']:>2}  Verified: {stats['verified']:>2} ({ver_rate:>4.0f}%)  Referenced: {stats['referenced']:>2}"
            if reviews:
                line += f"  Reviews: {reviews.get('given', 0)} given"
            line += f"  Tokens: {tokens:,}"
            print(line)

    # Save results (checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    society.save(output_dir)

    # Save experiment metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "n_agents": n_agents,
        "n_generations": n_generations,
        "generations_completed": society.current_generation,
        "provider": provider_name,
        "prompt_versions": get_all_versions(),
        "total_tokens_used": society.total_tokens_used,
        "max_tokens": max_tokens,
        "analysis": {
            "total_artifacts": analysis.total_artifacts,
            "verified_artifacts": analysis.verified_artifacts,
            "reuse_rate": analysis.reuse_rate,
            "fresh_creation_rate": analysis.fresh_creation_rate,
            "verification_rate": analysis.verification_rate,
            "growth_rate": analysis.growth_rate,
            "potential_tasmania_effect": analysis.potential_tasmania_effect,
        },
    }
    # Add provider details
    if mixed_providers:
        metadata["providers"] = [
            {"name": p.name, "model": p.config.model}
            for p in society.providers
        ]
    else:
        metadata["model"] = society.provider.config.model

    # Add per-agent breakdowns
    if society.tokens_by_agent:
        metadata["tokens_by_agent"] = society.tokens_by_agent
    if society.artifacts_by_agent:
        metadata["artifacts_by_agent"] = society.artifacts_by_agent
    if society.reviews_by_agent:
        metadata["reviews_by_agent"] = society.reviews_by_agent

    # Add goal info if present
    if goal:
        metadata["goal"] = {
            "name": goal.name,
            "source": goal.source,
            "progress": goal.progress(),
            "definitions": [
                {
                    "tag": d.tag,
                    "name": d.name,
                    "formalized": d.formalized,
                    "artifact_ids": d.artifact_ids,
                }
                for d in goal.definitions
            ],
        }
        # Save goal state separately for potential resumption
        goal.save(output_dir / "goal.json")

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nResults saved to: {output_dir}")
    if society.current_generation < n_generations:
        print(f"  To resume: uv run python -m lms.run --resume {output_dir}")


async def resume_experiment(
    checkpoint_dir: Path,
    target_generations: int,
    config: Config,
    max_tokens: int | None = None,
    use_real_verifier: bool = False,
    verifier_type: str = "mock",
    iterative_mode: bool = False,
    max_attempts: int = 5,
) -> None:
    """Resume an experiment from a checkpoint.

    Args:
        checkpoint_dir: Directory with saved checkpoint
        target_generations: Total generations to reach
        config: Configuration object
        max_tokens: New max tokens (None = use from checkpoint)
        use_real_verifier: Use real LEAN 4 verifier (deprecated, use verifier_type)
        verifier_type: Verifier to use: "mock", "real", or "mcp"
        iterative_mode: If True, each agent gets multiple verification attempts
        max_attempts: Max attempts per agent in iterative mode
    """
    # Load metadata to get provider info
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    provider_name = metadata["provider"]
    n_agents = metadata["n_agents"]

    print(f"\nResuming LMS Experiment from checkpoint")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Completed generations: {metadata['generations_completed']}")
    print(f"  Target generations: {target_generations}")
    print(f"  Tokens used so far: {metadata['total_tokens_used']:,}")
    if iterative_mode:
        print(f"  Mode: ITERATIVE ({max_attempts} attempts/agent)")
    print()

    # Handle mixed provider mode
    if provider_name == "mixed":
        available = config.available_providers()
        if len(available) < n_agents:
            raise ValueError(
                f"Mixed mode requires {n_agents} providers but only {len(available)} configured"
            )
        providers = [
            create_provider(name, config.get_provider_config(name))
            for name in available[:n_agents]
        ]
        provider = providers[0]  # For Society.load compatibility
    else:
        provider_config = config.get_provider_config(provider_name)
        provider = create_provider(provider_name, provider_config)
        providers = None

    # Choose verifier
    lean_project_dir = Path(__file__).parent.parent / "lean"

    if use_real_verifier or verifier_type == "real":
        verifier = RealLeanVerifier(project_dir=lean_project_dir)
    elif verifier_type == "mcp":
        verifier = MCPLeanVerifier()
    else:
        verifier = MockLeanVerifier()

    # Load society from checkpoint
    society = Society.load(checkpoint_dir, provider, verifier)

    # If mixed mode, update the providers list
    if providers:
        society.providers = providers
        society.agents = [
            Agent(
                id=f"agent-{i}-{providers[i].name}",
                provider=providers[i],
                generation=society.current_generation,
            )
            for i in range(n_agents)
        ]

    # Update max_tokens if specified
    if max_tokens:
        society.max_tokens = max_tokens

    # Enable iterative mode if requested
    if iterative_mode:
        society.iterative_mode = True
        society.max_attempts = max_attempts

    # Continue running
    print(f"Continuing from generation {society.current_generation}...")
    try:
        results = await society.run_from_checkpoint(target_generations)
        for result in results:
            print(
                f"  Generation {result.generation + 1}: "
                f"Created: {result.artifacts_created}, "
                f"Verified: {result.artifacts_verified}, "
                f"Tokens: {result.tokens_used:,}"
            )
    except BudgetExceeded as e:
        print(f"\n  Budget exceeded: {e}")

    # Analyze and save
    analysis = analyze_library(society.library, society.results)
    print_analysis(analysis)

    print(f"\nToken Usage: {society.total_tokens_used:,} total")

    # Save updated checkpoint
    society.save(checkpoint_dir)

    # Save updated goal progress
    if society.goal:
        society.goal.save(checkpoint_dir / "goal.json")

    # Update metadata
    metadata["timestamp"] = datetime.now().isoformat()
    metadata["generations_completed"] = society.current_generation
    metadata["total_tokens_used"] = society.total_tokens_used
    metadata["analysis"] = {
        "total_artifacts": analysis.total_artifacts,
        "verified_artifacts": analysis.verified_artifacts,
        "reuse_rate": analysis.reuse_rate,
        "fresh_creation_rate": analysis.fresh_creation_rate,
        "verification_rate": analysis.verification_rate,
        "growth_rate": analysis.growth_rate,
        "potential_tasmania_effect": analysis.potential_tasmania_effect,
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nCheckpoint updated: {checkpoint_dir}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LMS: LLM Mathematical Society Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--agents",
        "-a",
        type=int,
        default=3,
        help="Number of agents in the society",
    )

    parser.add_argument(
        "--generations",
        "-g",
        type=int,
        default=6,
        help="Number of generations to run",
    )

    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default=None,
        help="LLM provider (anthropic, openai, google)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: experiments/run_<timestamp>)",
    )

    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Path to .env file",
    )

    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=None,
        help="Maximum tokens to use (budget limit)",
    )

    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )

    parser.add_argument(
        "--real-verifier",
        action="store_true",
        help="Use real LEAN 4 verifier instead of mock",
    )

    parser.add_argument(
        "--mixed",
        "-m",
        action="store_true",
        help="Use all available providers (one agent per provider, heterogeneous society)",
    )

    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help=f"Goal to work towards. Available: {', '.join(list_goals())}",
    )

    parser.add_argument(
        "--verifier",
        type=str,
        choices=["mock", "real", "mcp"],
        default="mock",
        help="Verifier type: mock (heuristic), real (direct lean), mcp (lean-lsp)",
    )

    parser.add_argument(
        "--list-goals",
        action="store_true",
        help="List available goals and exit",
    )

    parser.add_argument(
        "--iterative",
        "-i",
        action="store_true",
        help="Enable iterative mode: each agent gets 5 verification attempts per generation",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Max verification attempts per agent in iterative mode",
    )

    args = parser.parse_args()

    # Handle --list-goals
    if args.list_goals:
        print("Available goals:")
        for goal_name in list_goals():
            goal_obj = get_goal(goal_name)
            print(f"  {goal_name}: {goal_obj.name}")
            print(f"    {len(goal_obj.definitions)} definitions/milestones")
        return

    # Load config
    config = Config.from_env(Path(args.env) if args.env else None)

    # Resume from checkpoint?
    if args.resume:
        checkpoint_dir = Path(args.resume)
        if not checkpoint_dir.exists():
            print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
            return

        asyncio.run(
            resume_experiment(
                checkpoint_dir=checkpoint_dir,
                target_generations=args.generations,
                config=config,
                max_tokens=args.max_tokens,
                use_real_verifier=args.real_verifier,
                verifier_type=args.verifier,
                iterative_mode=args.iterative,
                max_attempts=args.max_attempts,
            )
        )
        return

    # New experiment
    available = config.available_providers()

    if not available:
        print("Error: No API keys configured. Add keys to .env file.")
        print("  ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY")
        return

    # Handle mixed mode
    if args.mixed:
        if len(available) < args.agents:
            print(f"Error: Mixed mode requires {args.agents} providers but only {len(available)} configured.")
            print(f"Available providers: {', '.join(available)}")
            return
        provider = "mixed"
    else:
        # Determine provider
        provider = args.provider or config.default_provider
        if provider not in available:
            print(f"Error: Provider '{provider}' not configured.")
            print(f"Available providers: {', '.join(available)}")
            return

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("experiments") / f"run_{timestamp}"

    # Load goal if specified
    goal = None
    if args.goal:
        try:
            goal = get_goal(args.goal)
            print(f"Loaded goal: {goal.name}")
        except ValueError as e:
            print(f"Error: {e}")
            return

    # Run experiment
    asyncio.run(
        run_experiment(
            n_agents=args.agents,
            n_generations=args.generations,
            provider_name=provider,
            output_dir=output_dir,
            config=config,
            max_tokens=args.max_tokens,
            use_real_verifier=args.real_verifier,
            mixed_providers=args.mixed,
            goal=goal,
            verifier_type=args.verifier,
            iterative_mode=args.iterative,
            max_attempts=args.max_attempts,
        )
    )


if __name__ == "__main__":
    main()
