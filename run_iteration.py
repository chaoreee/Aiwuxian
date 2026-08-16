from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def extract_report(output: str) -> dict[str, dict[str, float]]:
    starts = [index for index, char in enumerate(output) if char == "{"]
    for start in reversed(starts):
        try:
            value = json.loads(output[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("evaluate_hybrid.py did not emit a JSON report")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one reproducible spatial-hole experiment and append its result to the iteration log."
    )
    parser.add_argument("--id", required=True, help="experiment id, for example R2-021")
    parser.add_argument("--note", required=True)
    parser.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    evaluator_args = args.evaluator_args
    if evaluator_args and evaluator_args[0] == "--":
        evaluator_args = evaluator_args[1:]

    root = Path(__file__).resolve().parent
    command = [sys.executable, str(root / "evaluate_hybrid.py"), *evaluator_args]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    report = extract_report(completed.stdout)
    alpha, best = max(report.items(), key=lambda item: item[1]["score_upa"])
    record = {
        "id": args.id,
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": args.note,
        "command": command,
        "best_alpha": alpha,
        "best": best,
        "report": report,
    }
    experiment_dir = root / "experiments"
    experiment_dir.mkdir(exist_ok=True)
    with (experiment_dir / "iterations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    command_text = " ".join(command[1:])
    markdown = (
        f"\n### {args.id}（自动记录）\n\n"
        f"- 时间：`{record['time']}`\n"
        f"- 假设：{args.note}\n"
        f"- 命令：`{command_text}`\n"
        f"- 最佳 alpha：`{alpha}`\n"
        f"- UPA-PAS / PDP / NMSE / 分数："
        f"`{best['upa_pas']:.6f} / {best['pdp']:.6f} / "
        f"{best['calibrated_nmse']:.6f} / {best['score_upa']:.6f}`\n"
    )
    with (root / "ITERATION_LOG.md").open("a", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"recorded {args.id}: score_upa={best['score_upa']:.6f}")


if __name__ == "__main__":
    main()
