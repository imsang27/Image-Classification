# https://huggingface.co/saltacc/anime-ai-detect/tree/main

# Use a pipeline as a high-level helper
from transformers import pipeline
from PIL import Image

pipe = pipeline("image-classification", model="saltacc/anime-ai-detect")

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("saltacc/anime-ai-detect")
model = AutoModelForImageClassification.from_pretrained("saltacc/anime-ai-detect")

# 분류 파이프라인 로드
pipe = pipeline("image-classification", model="saltacc/anime-ai-detect")

# 분류할 이미지 로드
image = Image.open("example.jpg")

# 결과 예측
result = pipe(image)
print(result)