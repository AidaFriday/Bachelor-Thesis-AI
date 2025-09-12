from connector import load_model
from components.camera import run_camera_loop


def main():
    print("Choose model:")
    print("1 - ArcFace")
    print("2 - FaceNet")
    print("3 - MagFace")

    choice = input("Enter choice [1/2/3]: ").strip()
    mapping = {"1": "arcface", "2": "facenet", "3": "magface"}
    model_name = mapping.get(choice, "arcface")

    wrapper = load_model(model_name)
    print(f"[INFO] Loaded model: {wrapper.name}")

    run_camera_loop(wrapper)


if __name__ == "__main__":
    main()
