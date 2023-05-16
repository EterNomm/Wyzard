import wyzard
import torch

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Input the text
text = "This text is written by human."

# Load GPT2 model
loader = wyzard.GPTDetector(model="gpt2", device=device)
# Check text
check = loader.check(text)

# Print output
print(check)
# Print more output details
print(loader.details())
