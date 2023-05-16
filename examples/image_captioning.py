import wyzard
import requests
import torch

# Load image from url
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = requests.get(url, stream=True).raw
device = "cuda" if torch.cuda.is_available() else "cpu"

# Start and load the model
detector = wyzard.ImageCaptioning(model="nlpconnect/vit-gpt2-image-captioning", device=device)
# Predicting the caption
results = detector.predict(image)

# Print the caption result
print(results)
