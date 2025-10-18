import os

dataset_path = r"C:\programming\Datasets\LFW\lfw-deepfunneled"

people = [
    p for p in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, p))
]
print(f"Found {len(people)} people")

sample_counts = []
for p in people[:10]:  # just check first 10
    imgs = [
        f
        for f in os.listdir(os.path.join(dataset_path, p))
        if f.lower().endswith(".jpg")
    ]
    sample_counts.append((p, len(imgs)))

for name, count in sample_counts:
    print(f"{name:25s}: {count} images")
