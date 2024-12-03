# https://huggingface.co/saltacc/anime-ai-detect/tree/main

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-classification", model="saltacc/anime-ai-detect")

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("saltacc/anime-ai-detect")
model = AutoModelForImageClassification.from_pretrained("saltacc/anime-ai-detect")