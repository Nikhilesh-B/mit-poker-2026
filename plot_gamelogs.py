"""
Utility to plot cumulative winnings from Pokerbots gamelog files.

Usage examples:
  python plot_gamelogs.py --input gamelog.txt
  python plot_gamelogs.py --input gamelogs --output plots
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend so plots can be generated headlessly.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import GAME_LOG_DIR

ROUND_HEADER = re.compile(
    r"^Round #(?P<round>\d+),\s*(?P<p1>.+?)\s+\((-?\d+)\),\s*(?P<p2>.+?)\s+\((-?\d+)\)"
)
AWARD_LINE = re.compile(r"^(?P<name>.+?) awarded (?P<delta>-?\d+)")


def parse_gamelog(log_path: Path) -> Tuple[List[int], List[str], Dict[str, List[int]]]:
    """
    Parse a gamelog.txt file and return rounds, player order, and cumulative totals.
    """
    rounds: List[int] = []
    players: List[str] = []
    cumulative: Dict[str, int] = {}
    histories: Dict[str, List[int]] = {}
    current_round: Optional[int] = None
    last_recorded_round: Optional[int] = None

    def ensure_player(name: str) -> None:
        if name not in cumulative:
            cumulative[name] = 0
            histories[name] = []
            players.append(name)

    def record_round() -> None:
        nonlocal last_recorded_round
        if current_round is None or current_round == last_recorded_round or not players:
            return
        rounds.append(current_round)
        for player in players:
            histories.setdefault(player, [])
            histories[player].append(cumulative.get(player, 0))
        last_recorded_round = current_round

    with log_path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            round_match = ROUND_HEADER.match(line)
            if round_match:
                record_round()
                current_round = int(round_match.group("round"))
                if not players:
                    first = round_match.group("p1").strip()
                    second = round_match.group("p2").strip()
                    ensure_player(first)
                    ensure_player(second)
                continue

            award_match = AWARD_LINE.match(line)
            if award_match:
                name = award_match.group("name").strip()
                delta = int(award_match.group("delta"))
                ensure_player(name)
                cumulative[name] += delta

    record_round()
    return rounds, players, histories


def normalize_histories(rounds: List[int], players: List[str], histories: Dict[str, List[int]]) -> None:
    """
    Ensure each player history is aligned with the number of rounds parsed.
    """
    for player in players:
        history = histories.get(player, [])
        if not history:
            histories[player] = [0 for _ in rounds]
            continue
        if len(history) < len(rounds):
            histories[player] = history + [history[-1]] * (len(rounds) - len(history))
        else:
            histories[player] = history[: len(rounds)]


def sanitize_for_filename(text: str) -> str:
    """Make a string safe for filenames."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    safe = re.sub(r"_{2,}", "_", safe).strip("_")
    return safe or "player"


def plot_rounds(
    rounds: List[int], players: List[str], histories: Dict[str, List[int]], output_path: Path
) -> None:
    plt.figure(figsize=(10, 6))
    for player in players:
        plt.plot(rounds, histories[player], marker="o", linewidth=2, label=player)
    plt.xlabel("Round")
    plt.ylabel("Cumulative winnings")
    plt.title(output_path.stem.replace("_", " "))
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def generate_plot_for_log(
    log_path: Path,
    output_dir: Optional[Path] = None,
    formats: Optional[Sequence[str]] = None,
) -> List[Path]:
    """
    Generate cumulative winnings plots for a single gamelog.

    Args:
        log_path: Path to the gamelog.txt file.
        output_dir: Directory to place plots. Defaults to log directory.
        formats: Iterable of formats to save (e.g., ["png", "pdf"]).
    """
    rounds, players, histories = parse_gamelog(log_path)
    if not rounds:
        raise ValueError(f"No rounds found in {log_path}")
    normalize_histories(rounds, players, histories)

    formats = list(formats) if formats else ["png"]
    output_dir = output_dir or log_path.parent

    p1 = sanitize_for_filename(players[0]) if players else "player1"
    p2 = sanitize_for_filename(players[1]) if len(players) > 1 else "player2"
    base_name = f"{log_path.stem}_{p1}_vs_{p2}_cumulative"

    outputs: List[Path] = []
    for fmt in formats:
        output_path = output_dir / f"{base_name}.{fmt}"
        plot_rounds(rounds, players, histories, output_path)
        outputs.append(output_path)
    return outputs


def process_file(log_path: Path, output_path: Optional[Path] = None) -> Path:
    outputs = generate_plot_for_log(
        log_path,
        output_dir=output_path.parent if output_path else None,
        formats=[output_path.suffix.lstrip(".")] if output_path and output_path.suffix else ["png"],
    )
    return outputs[0]


def collect_log_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.glob("*.txt") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cumulative winnings plots from Pokerbots gamelog files."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=GAME_LOG_DIR,
        help="Path to a gamelog.txt file or a directory containing multiple gamelogs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output image path (single file) or directory (when input is a directory).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    outputs: List[Path] = []
    if input_path.is_dir():
        logs = collect_log_files(input_path)
        if not logs:
            raise SystemExit(f"No .txt gamelog files found in directory: {input_path}")
        output_dir = Path(args.output) if args.output else input_path / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        for log_file in logs:
            outputs.extend(
                generate_plot_for_log(
                    log_file,
                    output_dir=output_dir,
                    formats=["png"],
                )
            )
    else:
        output_path = None
        if args.output:
            output_candidate = Path(args.output)
            if output_candidate.suffix:
                output_path = output_candidate
            else:
                output_path = output_candidate / f"{input_path.stem}_cumulative.png"
        outputs.extend(
            generate_plot_for_log(
                input_path,
                output_dir=output_path.parent if output_path else None,
                formats=[output_path.suffix.lstrip(".")] if output_path and output_path.suffix else ["png"],
            )
        )

    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

