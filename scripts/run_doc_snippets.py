"""Execute the ```python blocks of README.md and docs/ so documentation cannot drift from the code.

The blocks on one page run in order in a single fresh interpreter, so a later
block may use names from an earlier one. Mark a fragment, a placeholder or a
slow block by putting ``<!-- doctest: skip -->`` on the line right before its
opening fence.

Usage: python scripts/run_doc_snippets.py [--root REPO] [--python EXE] [--timeout S]
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

FENCE = re.compile(r"^```python\s*$")
SKIP = "<!-- doctest: skip -->"


def python_blocks(path: Path) -> list[str]:
    """Return the runnable ```python blocks of a Markdown file, in order."""
    lines = path.read_text().splitlines()
    blocks, i = [], 0
    while i < len(lines):
        if FENCE.match(lines[i]):
            skipped = i > 0 and lines[i - 1].strip() == SKIP
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            if not skipped:
                blocks.append("\n".join(lines[i + 1 : j]))
            i = j
        i += 1
    return blocks


def main() -> int:
    """Run every page's blocks and report pass/fail per page."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--python", default=sys.executable, help="Interpreter to run the snippets")
    parser.add_argument("--timeout", type=float, default=900.0, help="Seconds per page")
    args = parser.parse_args()

    pages = [args.root / "README.md", *sorted((args.root / "docs").rglob("*.md"))]
    failures = 0
    for page in pages:
        blocks = python_blocks(page)
        if not blocks:
            continue
        name = page.relative_to(args.root)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [args.python, "-c", "\n\n".join(blocks)],
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            ok, detail = proc.returncode == 0, proc.stderr[-2000:]
        except subprocess.TimeoutExpired:
            ok, detail = False, f"timed out after {args.timeout:.0f}s"
        status = "ok  " if ok else "FAIL"
        print(f"{status}  {name}  ({len(blocks)} blocks, {time.perf_counter() - start:.0f}s)")
        if not ok:
            failures += 1
            print("      " + detail.replace("\n", "\n      "))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
