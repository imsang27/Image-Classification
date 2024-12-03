# https://huggingface.co/saltacc/anime-ai-detect/tree/main

import os
import json
from PIL import Image
import torch

# Load model directly
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

# 1. 설정
# 이미지가 저장된 폴더 경로
image_folder = "path_to_image_folder"  # 폴더 경로를 수정하세요.
output_file = "predictions_result.json"  # 결과를 저장할 JSON 파일 이름

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 모델과 프로세서 로드
use_pipeline = True  # True면 pipeline 사용, False면 직접 모델과 프로세서 사용

if use_pipeline:
    # Pipeline 로드
    pipe = pipeline("image-classification", model="saltacc/anime-ai-detect", device=0 if torch.cuda.is_available() else -1)
else:
    # 모델과 프로세서 로드
    processor = AutoImageProcessor.from_pretrained("saltacc/anime-ai-detect")
    model = AutoModelForImageClassification.from_pretrained("saltacc/anime-ai-detect")
    model = model.to(device)

# 3. 폴더 내의 이미지 처리
results = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 이미지 로드
        image_path = os.path.join(image_folder, filename)
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            continue

        # 4. 분류 수행
        if use_pipeline:
            # Pipeline 방식
            result = pipe(image)
        else:
            # 직접 모델과 프로세서 사용
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            result = probs.detach().cpu().numpy().tolist()

        # 결과 저장
        results.append({"filename": filename, "prediction": result})

# 5. 결과 출력
for res in results:
    print(f"Image: {res['filename']}, Prediction: {res['prediction']}")

# 6. 결과를 JSON 파일로 저장
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)
print(f"Predictions saved to {output_file}")