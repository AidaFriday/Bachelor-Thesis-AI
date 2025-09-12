from connector import load_model
from components.camera import run_camera_loop


def main():
    print("Choose model:")
    print("1 - arcface")
    print("2 - facenet")
    print("3 - sphereface")
    choice = input("Enter choice [1/2/3]: ").strip()

    mapping = {"1": "arcface", "2": "facenet", "3": "sphereface"}
    model_name = mapping.get(choice, "arcface")

    wrapper = load_model(model_name)
    print(f"Loaded model: {wrapper.name}")

    run_camera_loop(wrapper)


if __name__ == "__main__":
    main()
