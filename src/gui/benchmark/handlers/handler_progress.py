def handle_progress(data, progress):
    if not ("progress" in data and "total" in data):
        return False

    try:
        cur = int(data.get("progress", 0))
        tot = int(data.get("total", 1))
        pct = int(100 * cur / tot)

        progress.setMaximum(100)
        progress.setValue(pct)

        # Include run info if present
        run_str = ""
        if "run" in data and "num_runs" in data:
            run_str = f" | Run {data['run']}/{data['num_runs']}"

        # Auto-reset if new run starts
        if cur == 1:
            progress.reset()

        progress.setFormat(f"Processing {cur}/{tot} items ({pct}%)" + run_str)

    except Exception as e:
        print(f"[WARN] progress parse error: {e}")

    return True
