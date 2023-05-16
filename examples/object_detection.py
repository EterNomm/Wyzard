import wyzard
import requests
import torch

# Load image from url
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = requests.get(url, stream=True).raw

# Check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load detector
detector = wyzard.ObjectDetection(model:str="hustvl/yolos-small", device=device)
# Detect the image and save the output
detect = detector.detect(input_img=image, output_img="output.png", threshold:int=0.9, font_size:int=18, font_color:str="yellow")

# Print the coordinates
print(detector.coordinates(detect))
