import os
import matplotlib.image as mpimg


def handle_accuracy_video(data, fig, canvas):
    roc_png = data.get("roc_png")

    fig.clear()

    if roc_png and os.path.isfile(roc_png):
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        img = mpimg.imread(roc_png)
        ax.imshow(img)
        ax.set_axis_off()
        canvas.draw()
        return True

    ax = fig.add_subplot(111)
    ax.axis("off")

    acc = float(data.get("accuracy", 0)) * 100
    auc = data.get("auc")
    eer = data.get("eer")

    msg = [f"Accuracy: {acc:.2f}%"]
    if auc:
        msg.append(f"AUC: {auc:.4f}")
    if eer:
        msg.append(f"EER: {float(eer)*100:.2f}%")

    ax.text(0.02, 0.98, "\n".join(msg), ha="left", va="top")

    canvas.draw()
    return True
