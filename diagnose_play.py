#!/usr/bin/env python3
"""
Diagnostic game logger for the Phase 0 RL agent.

Plays a configurable matrix of matchups and writes per-game JSON logs rich
enough to diagnose:
  - where the agent stalls (last turn it made positive distance progress)
  - how often its move matches the greedy / heuristic baselines
    (counterfactual queries on every move)
  - per-move distance progress
  - multi-player congestion behaviour (4P / 6P)
  - self-play behaviour (does it play different games against itself?)
  - endgame solver activation rate
  - inference latency (vs the 2000 ms competition budget)

Output layout (under --out-dir, defaults to logs/diagnostic/<timestamp>):
  run_metadata.json          # config + final aggregate metrics
  summary.jsonl              # one line per completed game (top metrics)
  games/<matchup>__g<n>.json # full per-move record per game

Usage on the school machine:
  python3.10 diagnose_play.py \
    --checkpoint checkpoints/sup_20260418_090744/model_best.pt \
    --device cuda \
    --mcts-sims 20 \
    --games-per-matchup 3
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from env.local_game import LocalGame
from agents.chinese_checkers_agent import ChineseCheckersAgent
from agents.heuristic_agent import GreedyProgressAgent, HeuristicAgent
from agents.random_agent import RandomAgent


# ---------- per-state diagnostics ----------

def board_summary(game: LocalGame, colour: str) -> Dict[str, Any]:
    """Compact snapshot of a colour's situation at one point in time."""
    info = game.compute_distance_info(colour)
    return {
        "pins_in_goal": info["pins_in_goal"],
        "pins_in_home": info["pins_in_home"],
        "total_distance": int(info["total_distance"]),
        "max_distance": int(info["max_distance"]),
        "min_distance": int(info["min_distance"]),
    }


def safe_query(agent, game: LocalGame, colour: str) -> Optional[Tuple[int, int]]:
    """Ask a baseline what it would do here. None on any error."""
    try:
        return agent.select_action(game, colour)
    except Exception:
        return None


# ---------- one game ----------

def run_game(
    *,
    game_id: str,
    matchup: str,
    num_players: int,
    rl_agent: ChineseCheckersAgent,
    rl_seats: List[int],
    opponent_factory,
    max_moves: int = 300,
    log_baselines: bool = True,
) -> Dict[str, Any]:
    """Play one game; return a structured per-move log + final result."""
    game = LocalGame(num_players=num_players, max_moves=max_moves)
    game.reset()

    rl_colours = [game.turn_order[s] for s in rl_seats]

    seat_assignments: Dict[str, str] = {}
    agent_for: Dict[str, Any] = {}
    for colour in game.turn_order:
        if colour in rl_colours:
            seat_assignments[colour] = "rl"
            agent_for[colour] = rl_agent
        else:
            seat_assignments[colour] = baseline_label(matchup)
            agent_for[colour] = opponent_factory()

    # Persistent baseline-query agents for counterfactuals.
    cf_greedy = GreedyProgressAgent()
    cf_heuristic = HeuristicAgent()

    initial_state = {c: board_summary(game, c) for c in game.colours}

    moves: List[Dict[str, Any]] = []
    crash: Optional[str] = None
    rl_latencies: List[float] = []

    while not game.done:
        colour = game.current_colour()
        is_rl = colour in rl_colours
        a = agent_for[colour]

        legal = game.get_legal_moves(colour)
        legal_count = sum(len(v) for v in legal.values())

        before_summary = board_summary(game, colour)

        endgame_active = False
        if is_rl and rl_agent.endgame_solver is not None:
            try:
                endgame_active = rl_agent.endgame_solver.is_active(game, colour)
            except Exception:
                endgame_active = False

        cf_greedy_choice = safe_query(cf_greedy, game, colour) if log_baselines else None
        cf_heur_choice = safe_query(cf_heuristic, game, colour) if log_baselines else None

        t0 = time.time()
        try:
            pin_id, to_idx = a.select_action(game, colour)
        except Exception as e:
            crash = (
                f"select_action failure: {type(e).__name__}: {e} "
                f"(colour={colour}, move={game.move_count})\n"
                f"{traceback.format_exc()}"
            )
            break
        elapsed = time.time() - t0
        if is_rl:
            rl_latencies.append(elapsed)

        try:
            _, _, info = game.step(pin_id, to_idx)
        except Exception as e:
            crash = (
                f"step failure: {type(e).__name__}: {e} "
                f"(colour={colour}, move={game.move_count}, "
                f"chosen=({pin_id},{to_idx}))\n"
                f"{traceback.format_exc()}"
            )
            break

        after_summary = board_summary(game, colour)

        rec: Dict[str, Any] = {
            "turn": game.move_count,
            "colour": colour,
            "is_rl": is_rl,
            "agent_type": seat_assignments[colour],
            "legal_count": legal_count,
            "endgame_active": endgame_active,
            "pin_id": pin_id,
            "from_index": info["from_index"],
            "to_index": info["to_index"],
            "before": before_summary,
            "after": after_summary,
            "dist_improvement": before_summary["total_distance"] - after_summary["total_distance"],
            "pins_in_goal_delta": after_summary["pins_in_goal"] - before_summary["pins_in_goal"],
            "latency_s": round(elapsed, 4),
        }
        if log_baselines:
            rec["greedy_choice"] = list(cf_greedy_choice) if cf_greedy_choice else None
            rec["heuristic_choice"] = list(cf_heur_choice) if cf_heur_choice else None
            rec["matches_greedy"] = (cf_greedy_choice == (pin_id, to_idx))
            rec["matches_heuristic"] = (cf_heur_choice == (pin_id, to_idx))
        moves.append(rec)

    final_pins = {c: game.get_pin_positions(c) for c in game.colours}

    scores: Dict[str, Dict[str, float]] = {}
    try:
        scores = game.compute_scores()
    except Exception:
        pass

    rl_diag: Dict[str, Dict[str, Any]] = {}
    for rl_c in rl_colours:
        rl_moves = [m for m in moves if m["colour"] == rl_c]
        last_progress_turn = None
        for m in rl_moves:
            if m["dist_improvement"] > 0:
                last_progress_turn = m["turn"]
        gm = sum(1 for m in rl_moves if m.get("matches_greedy"))
        hm = sum(1 for m in rl_moves if m.get("matches_heuristic"))
        avg_legal = sum(m["legal_count"] for m in rl_moves) / max(1, len(rl_moves))
        endgame_used = sum(1 for m in rl_moves if m.get("endgame_active"))
        rl_diag[rl_c] = {
            "moves": len(rl_moves),
            "last_progress_turn": last_progress_turn,
            "stall_duration": (
                game.move_count - last_progress_turn
                if last_progress_turn is not None else None
            ),
            "greedy_match_rate": round(gm / max(1, len(rl_moves)), 3) if log_baselines else None,
            "heuristic_match_rate": round(hm / max(1, len(rl_moves)), 3) if log_baselines else None,
            "avg_legal_count": round(avg_legal, 1),
            "endgame_active_count": endgame_used,
        }

    rl_lat: Dict[str, float] = {}
    if rl_latencies:
        sorted_t = sorted(rl_latencies)
        p95_idx = min(int(0.95 * len(sorted_t)), len(sorted_t) - 1)
        rl_lat = {
            "moves": len(rl_latencies),
            "avg_s": round(sum(rl_latencies) / len(rl_latencies), 4),
            "p95_s": round(sorted_t[p95_idx], 4),
            "max_s": round(max(rl_latencies), 4),
        }

    return {
        "game_id": game_id,
        "matchup": matchup,
        "num_players": num_players,
        "rl_colours": rl_colours,
        "seat_assignments": seat_assignments,
        "completed": game.done and crash is None,
        "crash": crash,
        "winner": game.winner,
        "status": game.status,
        "move_count": game.move_count,
        "initial_state": initial_state,
        "final_pins": final_pins,
        "scores": scores,
        "rl_diagnostics": rl_diag,
        "rl_latency": rl_lat,
        "moves": moves,
    }


# ---------- matchup parsing ----------

def baseline_label(matchup: str) -> str:
    if "self" in matchup:
        return "rl"
    if "random" in matchup:
        return "random"
    if "heuristic" in matchup:
        return "heuristic"
    return "greedy"


def parse_matchups(spec: List[str], rl_agent) -> List[Tuple[str, int, List[int], Any]]:
    """Each spec item: '<np>p_vs_<random|greedy|heuristic>' or '<np>p_self_play'."""
    out = []
    for s in spec:
        if "p_" not in s:
            raise ValueError(f"Bad matchup spec: {s}. Use e.g. 2p_vs_greedy or 4p_self_play.")
        np_part, rest = s.split("p_", 1)
        try:
            num_players = int(np_part)
        except ValueError:
            raise ValueError(f"Bad player count in matchup: {s}")
        if num_players not in (2, 4, 6):
            raise ValueError(f"Player count must be 2/4/6 in matchup: {s}")
        if "self" in rest:
            rl_seats = list(range(num_players))
            opp = (lambda _ra=rl_agent: _ra)
        elif "random" in rest:
            rl_seats = [0]
            opp = (lambda: RandomAgent())
        elif "heuristic" in rest:
            rl_seats = [0]
            opp = (lambda: HeuristicAgent())
        elif "greedy" in rest:
            rl_seats = [0]
            opp = (lambda: GreedyProgressAgent())
        else:
            raise ValueError(f"Unknown opponent in matchup: {s}")
        out.append((s, num_players, rl_seats, opp))
    return out


# ---------- summarising / printing ----------

def build_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    rl_score = None
    rl_pins = None
    if result["rl_colours"] and result["scores"]:
        rl_c = result["rl_colours"][0]
        sc = result["scores"].get(rl_c, {})
        rl_score = sc.get("final_score")
        rl_pins = sc.get("pins_in_goal")
    rl_diag = (
        next(iter(result["rl_diagnostics"].values()), {})
        if result["rl_diagnostics"] else {}
    )
    return {
        "game_id": result["game_id"],
        "matchup": result["matchup"],
        "num_players": result["num_players"],
        "completed": result["completed"],
        "crash": (result["crash"].splitlines()[0] if result["crash"] else None),
        "winner": result["winner"],
        "status": result["status"],
        "move_count": result["move_count"],
        "rl_score": rl_score,
        "rl_pins_in_goal": rl_pins,
        "greedy_match_rate": rl_diag.get("greedy_match_rate"),
        "heuristic_match_rate": rl_diag.get("heuristic_match_rate"),
        "stall_duration": rl_diag.get("stall_duration"),
        "endgame_active_count": rl_diag.get("endgame_active_count"),
        "avg_legal_count": rl_diag.get("avg_legal_count"),
        "rl_avg_latency_s": result["rl_latency"].get("avg_s") if result["rl_latency"] else None,
        "rl_max_latency_s": result["rl_latency"].get("max_s") if result["rl_latency"] else None,
        "wall_time_s": result.get("wall_time_s"),
    }


def print_short(result: Dict[str, Any], summary: Dict[str, Any], wall: float):
    if result["crash"]:
        print(f"    CRASH: {result['crash'].splitlines()[0]}")
        return
    rl_c = result["rl_colours"][0]
    sc = result["scores"].get(rl_c, {})
    gm = summary["greedy_match_rate"]
    hm = summary["heuristic_match_rate"]
    gm_str = f"{gm:.2f}" if gm is not None else "  -  "
    hm_str = f"{hm:.2f}" if hm is not None else "  -  "
    print(
        f"    done in {wall:5.1f}s  moves={result['move_count']:3d}  "
        f"winner={(result['winner'] or 'draw'):>10s}  "
        f"RL_score={sc.get('final_score', 0):7.1f}  "
        f"pins={sc.get('pins_in_goal', 0):2d}/10  "
        f"greedy_match={gm_str}  heur_match={hm_str}"
    )


def aggregate_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_matchup: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        by_matchup.setdefault(e["matchup"], []).append(e)
    out: Dict[str, Any] = {}
    for m, items in by_matchup.items():
        valid = [x for x in items if not x["crash"]]
        if not valid:
            out[m] = {"games": len(items), "all_crashed": True}
            continue
        scores = [x["rl_score"] for x in valid if x["rl_score"] is not None]
        pins = [x["rl_pins_in_goal"] for x in valid if x["rl_pins_in_goal"] is not None]
        moves = [x["move_count"] for x in valid]
        gms = [x["greedy_match_rate"] for x in valid if x["greedy_match_rate"] is not None]
        hms = [x["heuristic_match_rate"] for x in valid if x["heuristic_match_rate"] is not None]
        stalls = [x["stall_duration"] for x in valid if x["stall_duration"] is not None]
        endgames = [x["endgame_active_count"] for x in valid if x["endgame_active_count"] is not None]
        winners = [x["winner"] for x in valid]
        rl_wins = sum(1 for x in valid if x["winner"] is not None and x["winner"] == _first_rl_colour(x))
        out[m] = {
            "games": len(items),
            "completed": len(valid),
            "crashed": len(items) - len(valid),
            "rl_wins": rl_wins,
            "draws": sum(1 for w in winners if w is None),
            "mean_rl_score": round(mean(scores), 1) if scores else None,
            "min_rl_score": min(scores) if scores else None,
            "max_rl_score": max(scores) if scores else None,
            "mean_rl_pins_in_goal": round(mean(pins), 2) if pins else None,
            "mean_move_count": round(mean(moves), 1) if moves else None,
            "mean_greedy_match_rate": round(mean(gms), 3) if gms else None,
            "mean_heuristic_match_rate": round(mean(hms), 3) if hms else None,
            "mean_stall_duration": round(mean(stalls), 1) if stalls else None,
            "mean_endgame_active_count": round(mean(endgames), 1) if endgames else None,
        }
    return out


def _first_rl_colour(summary_entry: Dict[str, Any]) -> Optional[str]:
    # Summary entries don't carry rl_colours directly; for the win count we
    # rely on the convention that seat 0 is RL except in self-play (where
    # any winner is "an RL win" by definition).
    if "self" in summary_entry["matchup"]:
        return summary_entry["winner"]  # any winner counts as an RL win
    # seat 0 is always 'red' (turn order starts with red for our matchups)
    return "red"


def fmt_or_dash(v, spec: str) -> str:
    if v is None:
        return "  -  "
    return ("{" + spec + "}").format(v)


def print_aggregates(agg: Dict[str, Any]):
    header = (
        f"  {'matchup':<22s} {'ok':>5s} {'crash':>5s} "
        f"{'RLscore':>8s} {'pins':>5s} {'moves':>5s} "
        f"{'greedy':>7s} {'heur':>7s} {'stall':>6s} {'endgm':>6s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for matchup, m in agg.items():
        if m.get("all_crashed"):
            print(f"  {matchup:<22s}  ALL CRASHED ({m['games']} games)")
            continue
        print(
            f"  {matchup:<22s} "
            f"{m['completed']:>5d} {m['crashed']:>5d} "
            f"{fmt_or_dash(m.get('mean_rl_score'), ':>8.1f')} "
            f"{fmt_or_dash(m.get('mean_rl_pins_in_goal'), ':>5.1f')} "
            f"{fmt_or_dash(m.get('mean_move_count'), ':>5.0f')} "
            f"{fmt_or_dash(m.get('mean_greedy_match_rate'), ':>7.2f')} "
            f"{fmt_or_dash(m.get('mean_heuristic_match_rate'), ':>7.2f')} "
            f"{fmt_or_dash(m.get('mean_stall_duration'), ':>6.0f')} "
            f"{fmt_or_dash(m.get('mean_endgame_active_count'), ':>6.1f')}"
        )


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mcts-sims", type=int, default=20,
                        help="Phase 0 sim count — 20 is the validated good setting")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--matchups", nargs="+",
        default=[
            "2p_vs_random",
            "2p_vs_greedy",
            "2p_vs_heuristic",
            "2p_self_play",
            "4p_vs_greedy",
            "6p_vs_greedy",
        ],
        help="e.g. 2p_vs_greedy 4p_vs_heuristic 6p_self_play",
    )
    parser.add_argument("--games-per-matchup", type=int, default=3)
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--out-dir", default=None,
                        help="defaults to logs/diagnostic/<timestamp>")
    parser.add_argument("--no-baselines", action="store_true",
                        help="skip greedy/heuristic counterfactual queries (faster)")
    parser.add_argument("--no-endgame", action="store_true",
                        help="disable the endgame solver in the RL agent")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("logs", "diagnostic", run_id)
    games_dir = os.path.join(out_dir, "games")
    os.makedirs(games_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.jsonl")
    metadata_path = os.path.join(out_dir, "run_metadata.json")

    print(f"Diagnostic run id: {run_id}")
    print(f"Output: {out_dir}")
    print(f"Loading agent: {args.checkpoint} "
          f"(device={args.device}, mcts_sims={args.mcts_sims}, "
          f"temperature={args.temperature}, "
          f"endgame={'OFF' if args.no_endgame else 'ON'})")

    rl_agent = ChineseCheckersAgent(
        checkpoint_path=args.checkpoint,
        mcts_simulations=args.mcts_sims,
        temperature=args.temperature,
        device=args.device,
        use_endgame=not args.no_endgame,
    )

    matchups = parse_matchups(args.matchups, rl_agent)
    print(f"Matchups: {[m[0] for m in matchups]}")
    print(f"Games per matchup: {args.games_per_matchup}")
    print(f"Total games: {args.games_per_matchup * len(matchups)}\n")

    metadata = {
        "run_id": run_id,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "mcts_sims": args.mcts_sims,
        "temperature": args.temperature,
        "use_endgame": not args.no_endgame,
        "matchups": [m[0] for m in matchups],
        "games_per_matchup": args.games_per_matchup,
        "max_moves": args.max_moves,
        "log_baselines": not args.no_baselines,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    # truncate summary file
    open(summary_path, "w").close()

    summary_entries: List[Dict[str, Any]] = []
    overall_t0 = time.time()

    for matchup_id, np_, rl_seats, opp_factory in matchups:
        print(f"\n=== {matchup_id} ({np_}P, RL seats={rl_seats}) ===")
        for i in range(args.games_per_matchup):
            game_id = f"{matchup_id}__g{i+1}"
            print(f"  [{game_id}] starting ...", flush=True)
            t0 = time.time()
            result = run_game(
                game_id=game_id,
                matchup=matchup_id,
                num_players=np_,
                rl_agent=rl_agent,
                rl_seats=rl_seats,
                opponent_factory=opp_factory,
                max_moves=args.max_moves,
                log_baselines=not args.no_baselines,
            )
            wall = time.time() - t0
            result["wall_time_s"] = round(wall, 1)

            game_path = os.path.join(games_dir, f"{game_id}.json")
            with open(game_path, "w") as f:
                json.dump(result, f, indent=2)

            summary = build_summary(result)
            summary_entries.append(summary)
            with open(summary_path, "a") as f:
                f.write(json.dumps(summary) + "\n")

            print_short(result, summary, wall)

    metadata["finished_at"] = datetime.now().isoformat()
    metadata["wall_time_s"] = round(time.time() - overall_t0, 1)
    metadata["aggregate"] = aggregate_metrics(summary_entries)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 90)
    print(" AGGREGATE")
    print("=" * 90)
    print_aggregates(metadata["aggregate"])
    print(f"\nTotal wall time: {metadata['wall_time_s']:.1f}s")
    print(f"Logs written to: {out_dir}")

    any_crash = any(e["crash"] for e in summary_entries)
    sys.exit(1 if any_crash else 0)


if __name__ == "__main__":
    main()
