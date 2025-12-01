import os
import json
import numpy as np
import matplotlib.cm as cm


def handle_latency(data, fig, canvas, page_ref):
    latency_series_all = data.get("latency_series_all", [])
    dataset = data.get("dataset", "")
    model = data.get("model", "")

    fig.clear()
    ax = fig.add_subplot(111)

    num_runs = len(latency_series_all)
    cmap = cm.get_cmap("tab10", max(1, num_runs))
    styles = ["-", "--", "-.", ":"]

    # save reports into /latency_reports
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "latency_reports",
    )
    os.makedirs(base_dir, exist_ok=True)

    all_data = {"runs": []}

    for i, latencies in enumerate(latency_series_all):
        if not latencies:
            continue
        lat = np.array(latencies)
        avg_ms = float(np.mean(lat))

        ax.plot(
            range(1, len(lat) + 1),
            lat,
            linestyle=styles[i % len(styles)],
            color=cmap(i),
            linewidth=1.5,
            label=f"Run {i+1} – {avg_ms:.2f} ms",
        )

        all_data["runs"].append(
            {
                "run": i + 1,
                "avg_ms": avg_ms,
                "min": float(lat.min()),
                "max": float(lat.max()),
                "std": float(lat.std()),
            }
        )

    # save report
    with open(os.path.join(base_dir, "latency_report.json"), "w") as f:
        json.dump(all_data, f, indent=4)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.subplots_adjust(right=0.8)

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Latency – {model}")

    ax.grid(True)
    canvas.draw()
    return True
