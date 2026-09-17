from langchain_core.tools import tool
from models.classifier_model import predict_image


@tool
def analyze_medical_image(image_path) -> dict:
    """
    Analyze a 2D medical image using the MONAI MedNIST classifier.

    Supported image types:
        - JPEG (.jpeg, .jpg)
        - PNG (.png)

    The image should be a single grayscale medical image.
    The classifier predicts one of these MedNIST categories:
        - AbdomenCT
        - BreastMRI
        - CXR
        - ChestCT
        - Hand
        - HeadCT

    Args:
        image_path: Path to the medical image file.

    Returns:
        A dictionary containing the predicted image category,
        prediction confidence, and probabilities for all categories.
    """
    return predict_image(image_path)
