import numpy as np
import matplotlib.cm as cm


def handle_fps(data, fig, canvas):
    fps_series_all = data.get("fps_series_all", [])
    run_avgs = data.get("runs", [])
    dataset = data.get("dataset", "")
    model = data.get("model", "")

    fig.clear()
    ax = fig.add_subplot(111)

    num_runs = len(fps_series_all)
    cmap = cm.get_cmap("tab10", num_runs)
    styles = ["-", "--", "-.", ":"]

    for i, fps_series in enumerate(fps_series_all):
        avg_fps = run_avgs[i] if i < len(run_avgs) else np.mean(fps_series)

        ax.plot(
            range(1, len(fps_series) + 1),
            fps_series,
            linestyle=styles[i % len(styles)],
            color=cmap(i),
            label=f"Run {i+1} ({avg_fps:.2f} FPS)",
        )

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.subplots_adjust(right=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("FPS")
    ax.set_title(f"Frames per Second – {model} ({dataset})")

    canvas.draw()
    return True
