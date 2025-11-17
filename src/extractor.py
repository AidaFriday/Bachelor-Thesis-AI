import os
from pathlib import Path


def collect_python_code(
    root_dir=".",
    output_file="all_code.txt",
    exclude_folder="pretrained_models"
):
    root = Path(root_dir).resolve()
    output_file = Path(output_file).resolve()

    python_files = []

    # ---------------------------------------------------------
    # Collect all .py files (excluding specific folder)
    # ---------------------------------------------------------
    for path in root.rglob("*.py"):
        if exclude_folder in path.parts:
            continue
        python_files.append(path)

    # Sort results for stable ordering
    python_files = sorted(python_files)

    # ---------------------------------------------------------
    # Write output file
    # ---------------------------------------------------------
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# ==== Collected Python Files ====\n")

        for p in python_files:
            out.write(str(p) + "\n")

        out.write("\n\n# ==== File Contents Below ====\n")

        # Write contents
        for p in python_files:
            out.write(f"\n\n# ==== {p} ====\n\n")
            try:
                with open(p, "r", encoding="utf-8") as f:
                    out.write(f.read())
            except Exception as e:
                out.write(f"\n# ERROR reading file: {e}\n")

    print(f"✅ All Python code collected into: {output_file}")


if __name__ == "__main__":
    collect_python_code()
