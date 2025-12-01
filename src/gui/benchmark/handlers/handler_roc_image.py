import os
import matplotlib.image as mpimg


def handle_roc_image(data, fig, canvas):
    img_path = data.get("path")
    if not img_path or not os.path.exists(img_path):
        return False

    fig.clear()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.set_axis_off()

    canvas.draw()
    return True
