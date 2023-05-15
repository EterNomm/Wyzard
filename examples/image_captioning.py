import wyzard
import requests

# Load image from url
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = requests.get(url, stream=True).raw

# Start and load the model
detector = wyzard.ImageCaptioning(model="nlpconnect/vit-gpt2-image-captioning")
# Predicting the caption
results = detector.predict(image)

# Print the caption result
print(results)
