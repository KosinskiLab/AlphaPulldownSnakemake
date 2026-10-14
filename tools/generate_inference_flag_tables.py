#!/usr/bin/env python3
"""Generate the workflow's inference-flag tables from an AlphaPulldown checkout.

``workflow/rules/common.smk`` warns at parse time about ``structure_inference_arguments``
the prediction container's backend will reject. It needs AlphaPulldown's per-backend
flag sets (``FLAGS_BY_BACKEND`` in ``alphapulldown/prediction/inference_flags.py``), but
AlphaPulldown is not importable where the workflow parses: it lives in the container. So
the sets are copied into a generated block of ``common.smk``, which records the
AlphaPulldown commit it came from, and CI regenerates that block from the recorded
commit and fails on any difference.

The source is read, not imported: its syntax tree is evaluated for literal sets, ``|``
unions and the dict of backends, so the checkout needs no dependencies, no submodules
and no installation.

Updating, for a new AlphaPulldown release the workflow targets:

1. Check out the intended AlphaPulldown revision, clean, without submodules:
       git clone https://github.com/KosinskiLab/AlphaPulldown.git /tmp/ap
       git -C /tmp/ap checkout <tag or commit>
2. Regenerate:
       python tools/generate_inference_flag_tables.py --source-checkout /tmp/ap
3. Inspect the diff of workflow/rules/common.smk: each added or removed flag should
   be one the release meant to add or remove.
4. Run the workflow tests and the default-container compatibility tests
   (python -m pytest test), and bump the default containers in the same change if the
   new flags need them.

``--check`` regenerates without writing and exits 1 on any difference; CI runs it
against the commit the block records (``--print-recorded-commit``).
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
TARGET = REPOSITORY / "workflow" / "rules" / "common.smk"
SOURCE_FILE = "alphapulldown/prediction/inference_flags.py"
SOURCE_REPOSITORY = "KosinskiLab/AlphaPulldown"
BEGIN = "# BEGIN GENERATED: inference flags"
END = "# END GENERATED: inference flags"
_RECORDED_COMMIT = re.compile(rf"{re.escape(SOURCE_REPOSITORY)}@([0-9a-f]{{40}})")


class GenerationError(Exception):
    """The source or the target cannot be used; the message says why."""


def _git(checkout: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise GenerationError(f"{checkout} is not a usable git checkout: {detail.strip()}")
    return result.stdout.strip()


def _evaluate(node: ast.AST, names: dict[str, object]) -> object:
    """Literal sets of strings, ``frozenset(...)``, ``|`` unions, names already
    evaluated, and dicts of those; anything else is an error, not a guess."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return frozenset(_evaluate(element, names) for element in node.elts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("frozenset", "set")
        and len(node.args) == 1
        and not node.keywords
    ):
        return frozenset(_evaluate(node.args[0], names))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _evaluate(node.left, names) | _evaluate(node.right, names)
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Dict) and all(key is not None for key in node.keys):
        return {_evaluate(key, names): _evaluate(value, names) for key, value in zip(node.keys, node.values)}
    raise GenerationError(f"line {getattr(node, 'lineno', '?')}: cannot evaluate {ast.dump(node)[:80]}")


def read_flags(checkout: Path) -> tuple[str, dict[str, frozenset[str]]]:
    """The checkout's commit and its ``FLAGS_BY_BACKEND``."""
    checkout = checkout.resolve()
    source = checkout / SOURCE_FILE
    if not source.is_file():
        raise GenerationError(f"{source} does not exist: is {checkout} an AlphaPulldown checkout?")
    commit = _git(checkout, "rev-parse", "HEAD")
    if _git(checkout, "status", "--porcelain", "--", SOURCE_FILE):
        raise GenerationError(f"{source} has uncommitted changes; generate from a commit.")

    names: dict[str, object] = {}
    for statement in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target, value = statement.target, statement.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id.isupper():
            try:
                names[target.id] = _evaluate(value, names)
            except GenerationError:
                if target.id == "FLAGS_BY_BACKEND":
                    raise
    flags = names.get("FLAGS_BY_BACKEND")
    if not isinstance(flags, dict) or not flags or not all(
        isinstance(backend, str) and isinstance(accepted, frozenset)
        and all(isinstance(flag, str) for flag in accepted)
        for backend, accepted in flags.items()
    ):
        raise GenerationError(f"{source} defines no FLAGS_BY_BACKEND of flag-name sets.")
    return commit, flags


def render(commit: str, flags: dict[str, frozenset[str]]) -> str:
    """The generated block, deterministic: backends and flags sorted."""
    lines = [
        f"{BEGIN} (tools/generate_inference_flag_tables.py; do not edit by hand)",
        f"# Source: {SOURCE_REPOSITORY}@{commit}",
        f"#         {SOURCE_FILE}, FLAGS_BY_BACKEND",
        "ALLOWED_INFERENCE_FLAGS = {",
    ]
    for backend in sorted(flags):
        lines.append(f'    "{backend}": frozenset({{')
        lines.extend(f'        "{flag}",' for flag in sorted(flags[backend]))
        lines.append("    }),")
    lines += ["}", END]
    return "\n".join(lines) + "\n"


def _block_span(text: str) -> tuple[int, int]:
    begins = [match.start() for match in re.finditer(re.escape(BEGIN), text)]
    ends = [match.end() for match in re.finditer(re.escape(END), text)]
    if len(begins) != 1 or len(ends) != 1 or begins[0] > ends[0]:
        raise GenerationError(f"{TARGET} needs exactly one '{BEGIN}' ... '{END}' block.")
    end = ends[0] + (1 if text[ends[0]:ends[0] + 1] == "\n" else 0)
    return begins[0], end


def recorded_commit(text: str) -> str:
    start, end = _block_span(text)
    match = _RECORDED_COMMIT.search(text[start:end])
    if match is None:
        raise GenerationError(f"the generated block in {TARGET} records no source commit.")
    return match.group(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-checkout", type=Path, help="AlphaPulldown checkout to read")
    parser.add_argument("--check", action="store_true", help="fail on a difference; write nothing")
    parser.add_argument(
        "--print-recorded-commit", action="store_true",
        help="print the AlphaPulldown commit the block records, and exit",
    )
    parser.add_argument("--target", type=Path, default=TARGET, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        text = args.target.read_text(encoding="utf-8")
        if args.print_recorded_commit:
            print(recorded_commit(text))
            return 0
        if args.source_checkout is None:
            parser.error("--source-checkout is required")
        commit, flags = read_flags(args.source_checkout)
        start, end = _block_span(text)
        updated = text[:start] + render(commit, flags) + text[end:]
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if updated == text:
        print(f"{args.target.name}: inference flags match {SOURCE_REPOSITORY}@{commit}")
        return 0
    if args.check:
        sys.stdout.writelines(
            difflib.unified_diff(
                text.splitlines(keepends=True), updated.splitlines(keepends=True),
                f"{args.target.name} (recorded)", f"{args.target.name} (from {commit[:12]})",
            )
        )
        print(f"error: {args.target.name} differs from {SOURCE_REPOSITORY}@{commit}", file=sys.stderr)
        return 1
    args.target.write_text(updated, encoding="utf-8")
    print(f"{args.target.name}: inference flags regenerated from {SOURCE_REPOSITORY}@{commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
