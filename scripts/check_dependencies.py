#!/usr/bin/env python3
"""Enforce the two-layer dependency contract for this repo.

Shared definition: this file is authored in the prep-teaching-repo skill
(`scripts/check_dependencies.py`) and copied verbatim into each prepared repo so
learners can run it without the skill installed. Edit it in the skill, then
re-copy - never fork the rule in a repo.

Layer 1 (global toolchain): flake.nix + flake.lock - interpreter, uv, git only.
Layer 2 (project deps):     pyproject.toml + uv.lock - everything the code imports.

The checks below are what "every notebook passes the dependency checks" means:

  1. layout      - both layers present and locked; no competing manifest
                   (requirements.txt / environment.yml / Pipfile / poetry.lock)
                   committed next to pyproject.toml; .venv ignored.
  2. interpreter - flake.nix python3NN, requires-python, and .python-version agree.
  3. imports     - every third-party module imported by a .py file, a notebook code
                   cell, or a fenced python block inside a notebook markdown cell
                   maps to a package declared in pyproject.toml.
  4. stale-refs  - no prose or cell tells a reader to install from a file that does
                   not exist in the repo.
  5. installers  - setup instructions use uv, not pip. `pip install` is tolerated
                   only inside a cell or paragraph that is explicitly about
                   deploying to a pip-only host (Render, Railway, Heroku, ...),
                   where a requirements.txt is a generated export, not a source
                   of truth.

Run: uv run python scripts/check_dependencies.py
Exit code 0 = contract holds.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from pathlib import Path

# Repo root: an explicit argument (how the skill's contract script calls it),
# otherwise the parent of this script's directory (how a learner runs it in-repo).
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent

SCAN_GLOBS = ("*.py", "*.ipynb", "examples/*.py", "examples/*.ipynb", "homework/*.ipynb")
DOC_GLOBS = ("*.md", "examples/*.md", "homework/*.md")
EXCLUDE_DIRS = {".venv", "node_modules", "__pycache__", ".git", ".ipynb_checkpoints"}

COMPETING_MANIFESTS = ("requirements.txt", "environment.yml", "environment.yaml", "Pipfile", "Pipfile.lock", "poetry.lock", "setup.py", "setup.cfg")

# Fallback only - used when a chunk does not parse as Python (notebook magics,
# shell escapes, deliberately broken teaching examples). `from` requires a
# following `import` so that prose inside a string literal - "...different
# from a plain prompt." - is not read as an import of a package named `a`.
IMPORT_RE = re.compile(
    r"^[ \t]*(?:import[ \t]+([A-Za-z_][\w.]*)|from[ \t]+([A-Za-z_][\w.]*)[ \t]+import[ \t])",
    re.M,
)
MAGIC_RE = re.compile(r"^[ \t]*[!%]", re.M)
FENCED_RE = re.compile(r"```(?:python|py)\n(.*?)```", re.S)
PIP_RE = re.compile(r"(?:^|[!%\s`])(?:python[0-9.]* -m )?pip[0-9.]* install[^\n`\"]*", re.I)
INSTALL_FILE_RE = re.compile(r"(?:-r|--requirement)\s+([\w./-]+\.txt)")
DEPLOY_CONTEXT_RE = re.compile(r"render|railway|heroku|fly\.io|build command|deploy|uv pip compile", re.I)

# Module name -> distribution name, where they differ. Kept small on purpose:
# anything not listed is assumed to import under its own distribution name.
MODULE_TO_DIST = {
    "dotenv": "python-dotenv",
    "sentence_transformers": "sentence-transformers",
    "nest_asyncio": "nest-asyncio",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def iter_files(globs):
    for pattern in globs:
        for path in sorted(ROOT.glob(pattern)):
            if any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            yield path


def cells_of(path: Path):
    """Yield (label, text, kind) chunks for a source file or notebook."""
    if path.suffix == ".py":
        yield (path.name, path.read_text(encoding="utf-8"), "code")
        return
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb.get("cells", [])):
        text = "".join(cell.get("source", []))
        kind = "code" if cell.get("cell_type") == "code" else "markdown"
        yield (f"{path.name}[cell {i}]", text, kind)


def code_of(text: str, kind: str) -> str:
    """Executable python inside a chunk: the whole cell, or its fenced blocks."""
    if kind == "code":
        return text
    return "\n".join(FENCED_RE.findall(text))


def imports_in(code: str) -> set[str]:
    """Top-level module names imported by a chunk of Python.

    Parses with `ast` so that text inside string literals, comments, and
    docstrings is never mistaken for an import - notebooks are full of prose
    embedded in triple-quoted skill definitions. Only when the chunk does not
    parse (magics, shell escapes, intentionally broken examples) does this fall
    back to the line regex.
    """
    for candidate in (code, MAGIC_RE.sub("# ", code)):
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
        return found
    return {a or b for a, b in IMPORT_RE.findall(code) if (a or b)}


def local_module_names() -> set[str]:
    names = {p.stem for p in ROOT.glob("*.py")}
    names |= {p.stem for p in ROOT.glob("examples/*.py")}
    names |= {p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    return names


def declared_dependencies() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = list(data["project"].get("dependencies", []))
    for extra in data["project"].get("optional-dependencies", {}).values():
        deps.extend(extra)
    return {normalize(re.split(r"[<>=!~\[; ]", d, maxsplit=1)[0]) for d in deps}


def requires_python_minor() -> int | None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    m = re.search(r">=\s*3\.(\d+)", data["project"].get("requires-python", ""))
    return int(m.group(1)) if m else None


def workspace_owner() -> Path | None:
    """The uv workspace root above ROOT, if ROOT is a member of one.

    A member is not a standalone project: the root owns the single uv.lock and
    the flake, and the member legitimately has neither. Mirrors the same walk
    done by check_dependency_contract.sh so the two agree about what a member
    is allowed to be missing.
    """
    for probe in ROOT.parents:
        candidate = probe / "pyproject.toml"
        if candidate.exists() and re.search(r"^\[tool\.uv\.workspace\]", candidate.read_text(encoding="utf-8"), re.M):
            return probe
    return None


OWNER = workspace_owner()


def check_layout(fail):
    # Inside a workspace, the toolchain layer and the single lock live at the
    # root; requiring them here would flag a correctly structured member.
    required = ("pyproject.toml",) if OWNER else ("flake.nix", "flake.lock", "pyproject.toml", "uv.lock")
    for name in required:
        if not (ROOT / name).exists():
            fail("layout", f"missing {name} - both contract layers must be committed and locked")
    if OWNER and (ROOT / "uv.lock").exists():
        fail("layout", f"uv.lock inside a workspace member - the root at {OWNER} owns the single lock")
    for manifest in COMPETING_MANIFESTS:
        if (ROOT / manifest).exists():
            fail("layout", f"{manifest} competes with pyproject.toml as a source of truth - delete it")
    gitignore_path = (OWNER or ROOT) / ".gitignore"
    gitignore = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""
    if ".venv" not in gitignore:
        fail("layout", ".venv/ is not gitignored")
    flake_path = (OWNER or ROOT) / "flake.nix"
    flake = flake_path.read_text(encoding="utf-8") if flake_path.exists() else ""
    if re.search(r"python3\d*Packages\.\w+", flake):
        fail("layout", "flake.nix installs project libraries as Nix packages - layer 1 is toolchain only")
    for tool in ("uv", "git"):
        if not re.search(rf"pkgs\.{tool}\b", flake):
            fail("layout", f"flake.nix devShell does not provide `{tool}` - layer 1 is the toolchain layer")
    if "uv sync" in flake and "--frozen" not in flake:
        fail("layout", "flake.nix shellHook runs `uv sync` without --frozen - lock drift would pass silently")


def check_interpreter(fail):
    flake_path = (OWNER or ROOT) / "flake.nix"
    flake = flake_path.read_text(encoding="utf-8") if flake_path.exists() else ""
    pins = {int(m) for m in re.findall(r"pkgs\.python3(\d+)\b", flake)}
    if len(pins) > 1:
        fail("interpreter", f"flake.nix pins several interpreters: {sorted(pins)}")
    flake_minor = next(iter(pins), None)
    req_minor = requires_python_minor()
    pyver_file = ROOT / ".python-version"
    pyver_minor = None
    if pyver_file.exists():
        m = re.search(r"3\.(\d+)", pyver_file.read_text(encoding="utf-8"))
        pyver_minor = int(m.group(1)) if m else None

    seen = {k: v for k, v in
            {"flake.nix": flake_minor, "requires-python": req_minor, ".python-version": pyver_minor}.items()
            if v is not None}
    if len(set(seen.values())) > 1:
        detail = ", ".join(f"{k}=3.{v}" for k, v in seen.items())
        fail("interpreter", f"Python version disagreement ({detail}) - uv would silently use its own interpreter")


def check_imports(fail):
    declared = declared_dependencies()
    local = local_module_names()
    stdlib = set(sys.stdlib_module_names)
    for path in iter_files(SCAN_GLOBS):
        rel = path.relative_to(ROOT)
        for label, text, kind in cells_of(path):
            label = f"{rel}{label[len(path.name):]}"
            for top in sorted(imports_in(code_of(text, kind))):
                if top in stdlib or top in local or top.startswith("_"):
                    continue
                dist = normalize(MODULE_TO_DIST.get(top, top))
                if dist not in declared:
                    fail("imports", f"{label}: imports `{top}` ({dist}) but it is not declared in pyproject.toml")


def check_install_instructions(fail):
    for path in list(iter_files(SCAN_GLOBS)) + list(iter_files(DOC_GLOBS)):
        rel = path.relative_to(ROOT)
        chunks = ([(str(rel), path.read_text(encoding="utf-8"), "markdown")] if path.suffix == ".md"
                  else [(f"{rel}{label[len(path.name):]}", text, kind) for label, text, kind in cells_of(path)])
        for label, text, _kind in chunks:
            deploy = bool(DEPLOY_CONTEXT_RE.search(text))
            for target in INSTALL_FILE_RE.findall(text):
                if not (ROOT / target).exists() and not deploy:
                    fail("stale-refs", f"{label}: points a reader at `{target}`, which does not exist in this repo")
            for hit in PIP_RE.findall(text):
                if not deploy:
                    fail("installers", f"{label}: `{hit.strip()}` - use `uv add` / `uv sync`; pip is only for pip-only deploy hosts")


def main() -> int:
    failures: list[tuple[str, str]] = []

    def fail(check: str, message: str) -> None:
        failures.append((check, message))

    if not (ROOT / "pyproject.toml").exists():
        # No Python layer to check - an R-only repo locks with renv.lock, which
        # check_dependency_contract.sh verifies. Saying so beats a traceback.
        print("Dependency contract: SKIPPED (no pyproject.toml - not a Python project)")
        return 0

    check_layout(fail)
    check_interpreter(fail)
    check_imports(fail)
    check_install_instructions(fail)

    if failures:
        print(f"Dependency contract: {len(failures)} problem(s)\n")
        for check, message in failures:
            print(f"  [{check}] {message}")
        return 1
    print("Dependency contract: OK (layout, interpreter, imports, stale-refs, installers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
