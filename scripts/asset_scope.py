"""Comparison assets require their corresponding empirical summary section."""

from pathlib import Path, PurePosixPath


def comparison_section(name):
    path = PurePosixPath(name)
    if path.as_posix() in {"paper/generated/method_numerical_comparison.tex", "docs/NUMERICAL_COMPARISON.md"}:
        return "method_numerical_comparison"
    if (path.parent == PurePosixPath("figures") and path.suffix in {".png", ".pdf"} or
            path.parent == PurePosixPath("paper/generated") and path.suffix == ".tex"):
        if path.stem == "privacy_calibration":
            return None
        for section in ["evaluation", "extensions", "privacy", "fashion"]:
            if path.stem.startswith(section+"_"):
                return section
    return None


def remove_absent_comparisons(summary, folder):
    """Remove only owned comparison outputs, retaining calibration and other files."""
    removed = []
    for path in sorted(Path(folder).glob("*")):
        section = comparison_section(path.as_posix())
        if path.is_file() and section is not None and section not in summary:
            path.unlink()
            removed.append(path.as_posix())
    return removed
