# Standalone vision scripts

CLI tools that complement the `nexus` package. Run directly without installing the package.

## ultra_fast_face_system.py

Multi-API face detection and clustering pipeline:

- Detector: ONNX UltraFace (preferred), fallback to `face_recognition`
- Embeddings: ArcFace ONNX (preferred), fallback to `face_recognition`
- Output: JSON metadata, face crops, optional DBSCAN clustering

```bash
pip install pillow numpy opencv-python tqdm onnxruntime face_recognition scikit-learn

python3 scripts/ultra_fast_face_system.py \
  --input /path/to/photos \
  --output /path/to/results \
  --workers 4
```

Optional env vars: `ULTRAFACE_ONNX`, `ARCFACE_ONNX` for model paths.