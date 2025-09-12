import torch
from components.model_selector import select_model
from components.camera import run_camera_loop


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Choose model:")
    print("1 - ArcFace (buffalo_l local pack)")
    print("2 - FaceNet")
    print("3 - SphereFace")
    choice = input("Enter choice [1/2/3]: ").strip()

    app, encoder, out_size = select_model(choice, device)
    run_camera_loop(app, encoder, out_size, choice)


if __name__ == "__main__":
    main()
