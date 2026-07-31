"""
Autonomous research loop for TrimNN subgraph matching.

Uses the Anthropic Claude API to iteratively improve train.py,
running overnight with prompt caching for efficiency.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... python research_loop.py

Stop with Ctrl+C at any time — the last good commit is always preserved.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR     = Path(__file__).parent
TRAIN_FILE     = SCRIPT_DIR / "train_trimnn.py"
RESULTS_FILE   = SCRIPT_DIR / "results.tsv"
LOG_FILE       = SCRIPT_DIR / "run.log"
PROGRAM_FILE   = SCRIPT_DIR / "program.md"

MODEL          = "claude-opus-4-6"
TRAIN_TIMEOUT  = 660          # 11 minutes — time budget (5min) + overhead
MAX_ITERATIONS = 9999         # effectively infinite; Ctrl+C to stop

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git(*args, check=True):
    return subprocess.run(
        ["git", "-C", str(SCRIPT_DIR)] + list(args),
        capture_output=True, text=True, check=check
    )


def current_commit():
    return git("rev-parse", "--short", "HEAD").stdout.strip()


def init_results_tsv():
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text("commit\tvf2_spearman\tstatus\tdescription\n")
        print(f"Initialized {RESULTS_FILE}")


def _tail_results(results_text: str, n: int = 20) -> str:
    """Return the header + last n data rows of results.tsv."""
    if not results_text:
        return "(no results yet)"
    lines = results_text.splitlines()
    header = lines[:1]
    rows   = lines[1:]
    return "\n".join(header + rows[-n:])


def log_result(commit, vf2_spearman, status, description):
    with RESULTS_FILE.open("a") as f:
        f.write(f"{commit}\t{vf2_spearman}\t{status}\t{description}\n")


def read_vf2_spearman(log_text: str) -> float | None:
    m = re.search(r"^vf2_spearman:\s+([\d.]+)", log_text, re.MULTILINE)
    return float(m.group(1)) if m else None


def run_training() -> tuple[str, float | None]:
    """Run train.py, return (log_text, vf2_spearman or None)."""
    print("  Running: python train.py  (up to 11 min timeout) ...")
    result = subprocess.run(
        [sys.executable, str(TRAIN_FILE)],
        capture_output=True, text=True,
        timeout=TRAIN_TIMEOUT,
        cwd=str(SCRIPT_DIR),
    )
    log_text = result.stdout + result.stderr
    LOG_FILE.write_text(log_text)
    vf2_spearman = read_vf2_spearman(log_text)
    return log_text, vf2_spearman


# ---------------------------------------------------------------------------
# Claude interaction  (with prompt caching on program.md)
# ---------------------------------------------------------------------------

def ask_claude_for_improvement(
    client: anthropic.Anthropic,
    program_md: str,
    current_train_py: str,
    results_history: str,
    last_log_tail: str,
    iteration: int,
) -> str:
    """
    Ask Claude for a single concrete improvement to train.py.
    Returns the full new train.py source code.
    program.md is cached as part of the system prompt (stable across iters).
    """

    # The system prompt is stable → prompt cache will kick in after 2nd call
    system_prompt = (
        "You are an autonomous ML researcher optimizing a neural subgraph matching model.\n\n"
        "## Task specification\n\n"
        + program_md
        + "\n\n## Instructions\n\n"
        "Output ONLY the complete, runnable Python source for train.py — "
        "no markdown fences, no prose before or after. "
        "The file must import from prepare.py exactly as the original does. "
        "Make exactly ONE focused improvement per iteration. "
        "Prefer simplicity: a small improvement from clean code beats a large "
        "improvement from fragile complexity."
    )

    user_message = (
        f"## Iteration {iteration}\n\n"
        f"### Current train.py\n```python\n{current_train_py}\n```\n\n"
        f"### Results so far (last 20)\n{_tail_results(results_history)}\n\n"
        f"### Last run log (tail)\n```\n{last_log_tail[-1500:]}\n```\n\n"
        "Choose ONE improvement, implement it in train.py, and output the full file."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},   # cache the stable system prompt
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    # Extract text from response (skip thinking blocks)
    new_code = ""
    for block in response.content:
        if block.type == "text":
            new_code += block.text

    # Strip accidental markdown fences
    if "```python" in new_code:
        new_code = new_code.split("```python", 1)[1].split("```")[0]
    elif new_code.strip().startswith("```"):
        new_code = new_code.strip()[3:].split("```")[0]

    return new_code.strip()


def extract_description(train_py: str) -> str:
    """Pull a short description from the first docstring comment, if any."""
    m = re.search(r'#\s*Experiment:\s*(.+)', train_py)
    if m:
        return m.group(1).strip()[:80]
    # Fallback: first non-empty comment or docstring line
    for line in train_py.splitlines():
        line = line.strip()
        if line.startswith("#") and len(line) > 2:
            return line[1:].strip()[:80]
    return "(no description)"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set ANTHROPIC_API_KEY environment variable first.")

    client = anthropic.Anthropic(api_key=api_key)

    init_results_tsv()

    program_md = PROGRAM_FILE.read_text()

    best_vf2_spearman = -1.0
    last_log_tail     = "(first run)"
    iteration         = 0

    print(f"=== TrimNN AutoResearch Loop — model: {MODEL} ===")
    print(f"Working dir : {SCRIPT_DIR}")
    print(f"Results     : {RESULTS_FILE}")
    print(f"Press Ctrl+C to stop at any time.\n")

    # Check if there are existing results to seed best_vf2_spearman
    if RESULTS_FILE.exists():
        lines = RESULTS_FILE.read_text().splitlines()
        for line in lines[1:]:   # skip header
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    v = float(parts[1])
                    if v > best_vf2_spearman:
                        best_vf2_spearman = v
                except ValueError:
                    pass
        if best_vf2_spearman > -1.0:
            print(f"Resuming — best vf2_spearman so far: {best_vf2_spearman:.6f}\n")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}  |  best vf2_spearman so far: "
              f"{best_vf2_spearman:.6f}" if best_vf2_spearman > -1.0
              else f"Iteration {iteration}  |  no result yet")
        print(f"{'='*60}")

        current_train_py = TRAIN_FILE.read_text()
        results_history  = RESULTS_FILE.read_text() if RESULTS_FILE.exists() else ""

        # 1. Ask Claude for improvement
        print("  Consulting Claude for improvement ...")
        t0 = time.time()
        try:
            new_train_py = ask_claude_for_improvement(
                client, program_md,
                current_train_py, results_history, last_log_tail,
                iteration,
            )
        except Exception as exc:
            print(f"  Claude API error: {exc}")
            print("  Waiting 30s before retry ...")
            time.sleep(30)
            continue
        print(f"  Claude responded in {time.time()-t0:.1f}s")

        if not new_train_py or len(new_train_py) < 200:
            print("  Response too short / empty — skipping iteration.")
            continue

        # 2. Write new train.py and commit
        TRAIN_FILE.write_text(new_train_py + "\n")
        description = extract_description(new_train_py)

        git("add", "train.py")
        git("commit", "-m", f"iter{iteration}: {description}")
        commit = current_commit()
        print(f"  Committed {commit}: {description}")

        # 3. Run training
        try:
            log_text, vf2_spearman = run_training()
        except subprocess.TimeoutExpired:
            print("  TIMEOUT — reverting commit.")
            log_result(commit, "timeout", "crash", description)
            git("reset", "--hard", "HEAD~1")
            last_log_tail = "Training timed out."
            continue
        except Exception as exc:
            print(f"  Run error: {exc} — reverting.")
            log_result(commit, "error", "crash", description)
            git("reset", "--hard", "HEAD~1")
            last_log_tail = str(exc)
            continue

        # 4. Check result
        tail_lines = log_text.strip().splitlines()[-40:]
        last_log_tail = "\n".join(tail_lines)

        if vf2_spearman is None:
            print("  CRASH (no vf2_spearman in log) — reverting.")
            print("  Log tail:")
            print(last_log_tail[-800:])
            log_result(commit, "crash", "crash", description)
            git("reset", "--hard", "HEAD~1")
            continue

        print(f"  vf2_spearman = {vf2_spearman:.6f}  (best = "
              f"{best_vf2_spearman:.6f})")

        if vf2_spearman > best_vf2_spearman:
            best_vf2_spearman = vf2_spearman
            log_result(commit, f"{vf2_spearman:.6f}", "keep", description)
            print(f"  *** IMPROVED — keeping commit {commit} ***")
        else:
            log_result(commit, f"{vf2_spearman:.6f}", "revert", description)
            git("reset", "--hard", "HEAD~1")
            print(f"  No improvement — reverted.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user. Last good commit is preserved.")
