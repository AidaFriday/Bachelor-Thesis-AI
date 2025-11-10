from scipy.io import loadmat
from pathlib import Path

META_PATH = Path(r"C:\programming\Datasets\meta_data\meta_and_splits.mat")
meta = loadmat(str(META_PATH), squeeze_me=True)

video_names = meta["video_names"]  # (3425,)
splits = meta["Splits"]  # (500, 3, 10)


def get_pairs_for_fold(k: int):
    """
    k: fold index 0..9
    returns list of (video_name1, video_name2, label)
    """
    fold = splits[:, :, k]  # shape (500, 3)

    idx1 = fold[:, 0].astype(int) - 1  # MATLAB -> Python
    idx2 = fold[:, 1].astype(int) - 1
    labels = fold[:, 2].astype(int)  # 1 = same, 0 = different

    pairs = []
    for i1, i2, lab in zip(idx1, idx2, labels):
        name1 = video_names[i1]
        name2 = video_names[i2]
        # in case scipy wraps them (here they’re already str, but safe):
        if not isinstance(name1, str):
            name1 = name1.item()
        if not isinstance(name2, str):
            name2 = name2.item()
        pairs.append((name1, name2, lab))
    return pairs


if __name__ == "__main__":
    pairs0 = get_pairs_for_fold(0)
    print("num pairs in fold 0:", len(pairs0))
    print("example pair:", pairs0[0])
