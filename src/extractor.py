import os
from pathlib import Path


def collect_python_code(
    root_dir=".",
    output_file="all_code.txt",
    exclude_folders=None
):
    if exclude_folders is None:
        exclude_folders = {
            "venv",
            "__pycache__",
            ".git",
            "pretrained_models"
        }

    root = Path(root_dir).resolve()
    output_file = Path(output_file).resolve()

    python_files = []

    # ---------------------------------------------------------
    # SAFE recursive walk (prevents entering forbidden dirs)
    # ---------------------------------------------------------
    for current_root, dirnames, filenames in os.walk(root, topdown=True):
        # prune directories IN-PLACE
        dirnames[:] = [
            d for d in dirnames
            if d not in exclude_folders
        ]

        for filename in filenames:
            if filename.endswith(".py"):
                python_files.append(Path(current_root) / filename)

    python_files.sort()

    # ---------------------------------------------------------
    # Write output file
    # ---------------------------------------------------------
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# ==== Collected Python Files ====\n")

        for p in python_files:
            out.write(str(p) + "\n")

        out.write("\n\n# ==== File Contents Below ====\n")

        for p in python_files:
            out.write(f"\n\n# ==== {p} ====\n\n")
            try:
                out.write(p.read_text(encoding="utf-8"))
            except Exception as e:
                out.write(f"\n# ERROR reading file: {e}\n")

    print(f"✅ All Python code collected into: {output_file}")


if __name__ == "__main__":
    collect_python_code()
