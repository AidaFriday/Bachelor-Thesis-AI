import os


def collect_python_code(
    root_dir=".", output_file="all_code.txt", exclude_folder="pretrained_models"
):
    with open(output_file, "w", encoding="utf-8") as out:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Skip excluded folder
            if exclude_folder in dirpath.split(os.sep):
                continue

            for filename in filenames:
                if filename.endswith(".py"):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            code = f.read()
                        out.write(f"\n\n# ==== {file_path} ====\n\n")
                        out.write(code)
                        out.write("\n")
                    except Exception as e:
                        print(f"Skipping {file_path}: {e}")


if __name__ == "__main__":
    collect_python_code()
    print(
        "✅ All Python code has been collected into all_code.txt (excluding pretrained_models)"
    )
