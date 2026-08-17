"""Check the public artifact for secrets, local paths and broken Markdown links."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cmd",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = {
    "OpenAI-like key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "Bearer token": re.compile(r"\bBearer\s+[A-Za-z0-9._-]{24,}\b", re.I),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
MARKDOWN_LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
FORBIDDEN_NAMES = {
    ".env",
    "credentials.json",
    "token.json",
    "per_question_linked_probe.csv",
    "cohort_audit.csv",
}
FORBIDDEN_SUFFIXES = {".key", ".pem", ".sqlite", ".sqlite3", ".db"}


def candidate_files() -> list[Path]:
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    output = subprocess.check_output(command, cwd=ROOT, text=True, encoding="utf-8")
    return [ROOT / line for line in output.splitlines() if line.strip()]


def main() -> None:
    problems: list[str] = []
    files = candidate_files()
    for path in files:
        relative = path.relative_to(ROOT)
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden public artifact: {relative}")
        if path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
            problems.append(f"file exceeds 5 MiB: {relative}")
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if re.search(r"[A-Za-z]:\\Users\\[^\\\s]+", text):
            problems.append(f"absolute Windows user path: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{label}: {relative}")
        if path.suffix.lower() != ".md":
            continue
        for target in MARKDOWN_LINK.findall(text):
            target = unquote(target.split("#", 1)[0].strip())
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            linked = (path.parent / target).resolve()
            if not linked.exists():
                problems.append(f"broken Markdown link: {relative} -> {target}")

    if problems:
        print("\n".join(sorted(set(problems))))
        raise SystemExit(1)
    print(f"public artifact check passed: {len(files)} files")


if __name__ == "__main__":
    main()
