import os
import matplotlib.image as mpimg


def handle_accuracy_image(data, fig, canvas):
    roc_png = data.get("roc_png")

    fig.clear()

    if roc_png and os.path.isfile(roc_png):
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        img = mpimg.imread(roc_png)
        ax.imshow(img)
        ax.set_axis_off()
        canvas.draw()
        return True

    # fallback simple info
    ax = fig.add_subplot(111)
    ax.axis("off")

    acc = float(data.get("accuracy", 0)) * 100
    threshold = data.get("threshold", 0)
    pairs = data.get("pairs_tested", 0)

    msg = (
        f"Accuracy: {acc:.2f}%\n"
        f"Threshold: {threshold:.4f}\n"
        f"Pairs Tested: {pairs}"
    )

    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)

    canvas.d
