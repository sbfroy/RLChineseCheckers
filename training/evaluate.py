"""
Evaluation suite for the Chinese Checkers agent.

Runs matches against baseline agents and tracks metrics.
"""

import sys
import os
import time
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.local_game import LocalGame
from agents.random_agent import RandomAgent
from agents.heuristic_agent import GreedyProgressAgent, HeuristicAgent


def play_match(
    agent_a,
    agent_b,
    num_games: int = 10,
    num_players: int = 2,
    max_moves: int = 300,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Play a series of games between two agents.

    agent_a plays as the first colour in each game.
    Each agent must have a select_action(game, colour) method.

    Returns match statistics.
    """
    results = {
        "a_wins": 0, "b_wins": 0, "draws": 0,
        "a_scores": [], "b_scores": [],
        "game_lengths": [], "game_times": [],
    }

    for game_num in range(num_games):
        game = LocalGame(num_players=num_players, max_moves=max_moves)
        game.reset()

        colour_a = game.turn_order[0]
        colour_b = game.turn_order[1]

        agents = {colour_a: agent_a, colour_b: agent_b}

        start_time = time.time()

        while not game.done:
            colour = game.current_colour()
            agent = agents[colour]

            try:
                pin_id, to_index = agent.select_action(game, colour)
                state, done, info = game.step(pin_id, to_index)
            except Exception as e:
                if verbose:
                    print(f"  Error for {colour}: {e}")
                break

        game_time = time.time() - start_time
        results["game_lengths"].append(game.move_count)
        results["game_times"].append(game_time)

        scores = game.compute_scores()

        if game.winner == colour_a:
            results["a_wins"] += 1
        elif game.winner == colour_b:
            results["b_wins"] += 1
        else:
            results["draws"] += 1

        if colour_a in scores:
            results["a_scores"].append(scores[colour_a]["final_score"])
        if colour_b in scores:
            results["b_scores"].append(scores[colour_b]["final_score"])

        if verbose:
            winner_str = game.winner if game.winner else "draw"
            print(
                f"  Game {game_num+1}: winner={winner_str}, "
                f"moves={game.move_count}, time={game_time:.1f}s, "
                f"a_score={scores.get(colour_a, {}).get('final_score', 0):.0f}, "
                f"b_score={scores.get(colour_b, {}).get('final_score', 0):.0f}"
            )

    total = num_games
    results["a_win_rate"] = results["a_wins"] / total
    results["b_win_rate"] = results["b_wins"] / total
    results["avg_game_length"] = sum(results["game_lengths"]) / total
    results["avg_game_time"] = sum(results["game_times"]) / total
    results["avg_a_score"] = sum(results["a_scores"]) / len(results["a_scores"]) if results["a_scores"] else 0
    results["avg_b_score"] = sum(results["b_scores"]) / len(results["b_scores"]) if results["b_scores"] else 0

    return results


def evaluate_agent(
    agent,
    num_games: int = 10,
    num_players: int = 2,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate an agent against all baselines.

    Returns results dict keyed by opponent name.
    """
    baselines = {
        "random": RandomAgent(),
        "greedy": GreedyProgressAgent(),
        "heuristic": HeuristicAgent(),
    }

    all_results = {}

    for name, baseline in baselines.items():
        if verbose:
            print(f"\nvs {name}:")
        results = play_match(
            agent_a=agent,
            agent_b=baseline,
            num_games=num_games,
            num_players=num_players,
            verbose=verbose,
        )
        all_results[name] = results

        if verbose:
            print(
                f"  Win rate: {results['a_win_rate']:.0%} "
                f"({results['a_wins']}W / {results['b_wins']}L / {results['draws']}D)"
            )
            print(
                f"  Avg score: agent={results['avg_a_score']:.0f}, "
                f"baseline={results['avg_b_score']:.0f}"
            )
            print(f"  Avg game length: {results['avg_game_length']:.0f} moves")

    return all_results


def run_baseline_benchmark(num_games: int = 20, verbose: bool = True):
    """
    Benchmark baselines against each other.
    Useful for understanding the performance ladder.
    """
    agents = {
        "random": RandomAgent(),
        "greedy": GreedyProgressAgent(),
        "heuristic": HeuristicAgent(),
    }

    if verbose:
        print("=== Baseline Benchmark ===\n")

    for name_a, agent_a in agents.items():
        for name_b, agent_b in agents.items():
            if name_a >= name_b:
                continue
            if verbose:
                print(f"{name_a} vs {name_b}:")
            results = play_match(
                agent_a=agent_a,
                agent_b=agent_b,
                num_games=num_games,
                verbose=False,
            )
            if verbose:
                print(
                    f"  {name_a}: {results['a_win_rate']:.0%} "
                    f"({results['a_wins']}W), "
                    f"avg_score={results['avg_a_score']:.0f}"
                )
                print(
                    f"  {name_b}: {results['b_win_rate']:.0%} "
                    f"({results['b_wins']}W), "
                    f"avg_score={results['avg_b_score']:.0f}"
                )
                print(f"  Avg game length: {results['avg_game_length']:.0f}\n")
