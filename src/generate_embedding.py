import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

# Student image
image_path = "data/students/krishn.jpg"

# Read image
img = cv2.imread(image_path)

# Detect face and generate embedding
faces = app.get(img)

if len(faces) == 0:
    print("No face detected!")
    exit()

# Take first detected face
embedding = faces[0].embedding

# Create embeddings folder
os.makedirs("data/embeddings", exist_ok=True)

# Save embedding
embedding_path = "data/embeddings/krishn.npy"

np.save(embedding_path, embedding)

print("Embedding saved successfully!")
print("Embedding shape:", embedding.shape)
print("Saved to:", embedding_path)