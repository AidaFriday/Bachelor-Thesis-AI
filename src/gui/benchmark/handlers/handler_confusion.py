import numpy as np


def handle_confusion(data, fig, canvas):
    tp, tn, fp, fn = data["tp"], data["tn"], data["fp"], data["fn"]
    threshold = data["threshold"]
    model = data.get("model", "")
    dataset = data.get("dataset", "")

    fig.clear()
    ax = fig.add_subplot(111)

    cm = np.array(
        [
            [tp, fn],
            [fp, tn],
        ],
        dtype=float,
    )

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred POS", "Pred NEG"])
    ax.set_yticklabels(["Actual POS", "Actual NEG"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    ax.set_title(f"Confusion Matrix - {model} on {dataset}\nThr={threshold:.4f}")
    fig.colorbar(im, ax=ax)

    fig.subplots_adjust(right=0.75)

    metrics = (
        f"Accuracy: {data['accuracy']*100:.2f}%\n"
        f"Precision: {data['precision']*100:.2f}%\n"
        f"Recall (TAR): {data['recall']*100:.2f}%\n"
        f"Specificity (TNR): {data['specificity']*100:.2f}%\n"
        f"F1 Score: {data['f1']*100:.2f}%\n"
        f"FAR: {data['far']*100:.2f}%\n"
        f"FRR: {data['frr']*100:.2f}%"
    )

    fig.text(
        0.75,
        0.70,
        metrics,
        ha="left",
        va="center",
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1e1e1e", ec="#666666", alpha=0.92),
        transform=fig.transFigure,
    )

    canvas.draw()
    return True
