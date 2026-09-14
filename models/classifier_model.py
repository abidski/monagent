import torch
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
)


MODEL_PATH = "models/best_metric_model.pth"
IMAGE_PATH = "data/000001.jpeg"

class_names = [
    "AbdomenCT",
    "BreastMRI",
    "CXR",
    "ChestCT",
    "Hand",
    "HeadCT",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image(image_path):

    # Recreate the architecture
    model = DenseNet121(
        spatial_dims=2, in_channels=1, out_channels=len(class_names)
    ).to(device)

    # Load trained weights
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    # Same preprocessing used in the tutorial
    transform = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
        ]
    )

    image = transform(image_path)

    # [1, 64, 64] -> [1, 1, 64, 64]
    image = image.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)

    predicted_index = probabilities.argmax(dim=1).item()

    return {
        "prediction": class_names[predicted_index],
        "confidence": probabilities[0, predicted_index].item(),
        "probabilities": {
            class_names[i]: probabilities[0, i].item() for i in range(len(class_names))
        },
    }


if __name__ == "__main__":
    result = predict_image(IMAGE_PATH)

    print("Prediction:", result["prediction"])
    print("Confidence:", f"{result['confidence']:.2%}")

    print("\nProbabilities:")
    for class_name, probability in result["probabilities"].items():
        print(f"  {class_name}: {probability:.2%}")
