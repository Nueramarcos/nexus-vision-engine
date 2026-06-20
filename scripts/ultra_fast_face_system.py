#!/usr/bin/env python3
"""
Ultra-Fast Multi-API Face Recognition System
- Detector: ONNX UltraFace (preferred), fallback to face_recognition detector
- Embeddings: ArcFace ONNX (preferred), fallback to face_recognition.face_encodings
- Output: JSON metadata, face crops, optional clustering (DBSCAN)
- API: placeholders for Groq/OpenRouter/Ollama
"""

import os
import sys
import argparse
import json
import time
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict

from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import logging

# optional imports (onnxruntime, face_recognition, sklearn)
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

try:
    import face_recognition
    FR_AVAILABLE = True
except Exception:
    FR_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ultra-fast-face")

# -------------------------
# Utility helpers
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_image_cv(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Unable to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_image_pil(np_img: np.ndarray, out_path: Path, quality=90):
    img = Image.fromarray(np_img)
    img.save(out_path, quality=quality)

# -------------------------
# Simple ONNX UltraFace Detector
# -------------------------
class ONNXUltraFace:
    def __init__(self, model_path: str, providers=None, input_size=(320, 240)):
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime is required for ONNXUltraFace")
        self.model_path = model_path
        self.input_size = input_size  # (w,h)
        self.providers = providers or (["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"])
        self._sess = ort.InferenceSession(model_path, providers=self.providers)
        inp = self._sess.get_inputs()[0]
        self.input_name = inp.name
        # model-specific output handling will vary depending on the ONNX
        # Here we assume typical ultraface-like output - adjust if using a different model
        log.info(f"Loaded UltraFace ONNX: {self.model_path} (providers={self.providers})")

    def preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        iw, ih = self.input_size
        img_resized = cv2.resize(img, (iw, ih))
        # normalize (0..1) and transpose to CHW
        arr = img_resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :].astype(np.float32)
        return arr, (w, h)

    def postprocess(self, boxes_raw, scale):
        # Placeholder - depends on model; user should adapt if using a different ONNX
        # boxes_raw expected shape: (N, 6) -> [x1, y1, x2, y2, score, class]
        boxes = []
        ow, oh = scale
        for b in boxes_raw:
            x1, y1, x2, y2, score = float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])
            if score < 0.3:
                continue
            # map back to original image size
            x1 = max(0, min(ow, x1))
            x2 = max(0, min(ow, x2))
            y1 = max(0, min(oh, y1))
            y2 = max(0, min(oh, y2))
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(score)))
        return boxes

    def detect(self, img: np.ndarray, conf_threshold=0.3):
        arr, scale = self.preprocess(img)
        out_names = [o.name for o in self._sess.get_outputs()]
        out = self._sess.run(out_names, {self.input_name: arr})
        # Many ultraface models output boxes directly. We'll try to find a reasonable format.
        # Try to concatenate outputs and search any (N,6) shaped array
        boxes_raw = None
        for o in out:
            a = np.array(o)
            if a.ndim == 2 and a.shape[1] >= 5:
                boxes_raw = a
                break
        if boxes_raw is None:
            return []
        boxes = self.postprocess(boxes_raw, scale)
        # filter by conf_threshold
        boxes = [b for b in boxes if b[4] >= conf_threshold]
        return boxes

# -------------------------
# Fallback detectors / embeddings (face_recognition)
# -------------------------
class FaceRecognitionBackend:
    def __init__(self):
        if not FR_AVAILABLE:
            raise RuntimeError("face_recognition package required for fallback backend")
        log.info("Using face_recognition fallback backend")

    def detect(self, img: np.ndarray, conf_threshold=0.3):
        # face_recognition uses top/left/right/bottom
        rgb = img[:, :, ::-1] if img.shape[2] == 3 else img
        locs = face_recognition.face_locations(rgb, model="hog")  # or 'cnn' if installed
        res = []
        for (top, right, bottom, left) in locs:
            res.append((left, top, right, bottom, 1.0))
        return res

    def embed(self, img: np.ndarray, box: Tuple[int, int, int, int]):
        top, left, right, bottom = box[1], box[0], box[2], box[3]
        rgb = img[:, :, ::-1]
        crop = rgb[top:bottom, left:right]
        encs = face_recognition.face_encodings(rgb, known_face_locations=[(top, right, bottom, left)])
        return encs[0] if encs else None

# -------------------------
# Embedding using ONNX ArcFace (optional)
# -------------------------
class ONNXArcFace:
    def __init__(self, model_path: str, providers=None, input_size=(112,112)):
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime is required for ONNXArcFace")
        self.model_path = model_path
        self.input_size = input_size
        self.providers = providers or (["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"])
        self._sess = ort.InferenceSession(model_path, providers=self.providers)
        self.input_name = self._sess.get_inputs()[0].name
        log.info(f"Loaded ArcFace ONNX: {self.model_path}")

    def preprocess(self, crop_rgb: np.ndarray):
        # expects 112x112 RGB normalized [-1,1] or [0,1] depending on model
        w,h = self.input_size
        im = cv2.resize(crop_rgb, (w,h))
        im = im.astype(np.float32)
        im = im / 255.0
        im = np.transpose(im, (2,0,1))[np.newaxis, :].astype(np.float32)
        return im

    def embed(self, crop_rgb: np.ndarray):
        arr = self.preprocess(crop_rgb)
        out = self._sess.run(None, {self.input_name: arr})
        emb = np.array(out[0]).flatten()
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

# -------------------------
# High-level pipeline
# -------------------------
def process_single_image(path: Path, detector, embedder, out_dir: Path, resize_max: int = 1600, conf_threshold=0.3, verbose=False):
    try:
        img = load_image_cv(path)
    except Exception as e:
        return {"file": str(path), "error": str(e)}
    h0, w0 = img.shape[:2]
    scale = 1.0
    if max(h0, w0) > resize_max:
        scale = resize_max / max(h0, w0)
        img = cv2.resize(img, (int(w0*scale), int(h0*scale)))
    faces = detector.detect(img, conf_threshold=conf_threshold)
    metadata = {"file": str(path), "faces": []}
    for idx, (x1, y1, x2, y2, score) in enumerate(faces):
        # clip
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        # crop original-resolution coordinates if scaling happened
        if scale != 1.0:
            x1 = int(x1 / scale); y1 = int(y1 / scale); x2 = int(x2 / scale); y2 = int(y2 / scale)
            orig_img = load_image_cv(path)
            crop = orig_img[y1:y2, x1:x2]
        else:
            crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # embedding
        emb = None
        try:
            if embedder is not None:
                emb = embedder.embed(crop)
            elif FR_AVAILABLE:
                emb = FaceRecognitionBackend().embed(img, (x1,y1,x2,y2))
        except Exception as e:
            log.debug(f"Embed failed: {e}")
            emb = None
        face_id = f"{path.stem}_face{idx}"
        # write crop
        crops_dir = out_dir / "crops"
        ensure_dir(crops_dir)
        crop_path = crops_dir / f"{face_id}.jpg"
        save_image_pil(crop, crop_path)
        record = {
            "id": face_id,
            "bbox": [x1,y1,x2,y2],
            "score": float(score),
            "crop": str(crop_path),
            "embedding": emb.tolist() if emb is not None else None
        }
        metadata["faces"].append(record)
    # save metadata per image
    meta_dir = out_dir / "meta"
    ensure_dir(meta_dir)
    with open(meta_dir / (path.stem + ".json"), "w", encoding="utf8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata

def run_pipeline(args):
    in_dir = Path(args.input).expanduser()
    out_dir = Path(args.output).expanduser()
    ensure_dir(out_dir)
    image_paths = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp",".heic"}])
    log.info(f"Found {len(image_paths)} images in {in_dir}")

    # choose detector
    detector = None
    embedder = None

    # ONNX UltraFace (if model provided)
    if args.detector == "ultraface" and args.ultraface_model and ORT_AVAILABLE:
        providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if args.gpu and "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        detector = ONNXUltraFace(args.ultraface_model, providers=providers, input_size=(args.uf_width, args.uf_height))
    elif args.detector == "face_recognition" and FR_AVAILABLE:
        detector = FaceRecognitionBackend()
    else:
        # auto fallback
        if args.ultraface_model and ORT_AVAILABLE:
            detector = ONNXUltraFace(args.ultraface_model, providers=["CUDAExecutionProvider","CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"], input_size=(args.uf_width,args.uf_height))
        elif FR_AVAILABLE:
            detector = FaceRecognitionBackend()
        else:
            raise RuntimeError("No detector available. Install onnxruntime and/or face_recognition.")

    # choose embedder
    if args.arcface_model and ORT_AVAILABLE:
        embedder = ONNXArcFace(args.arcface_model, providers=["CUDAExecutionProvider","CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"])
    elif FR_AVAILABLE:
        embedder = None  # will use fallback face_recognition encoding inside process_single_image
    else:
        log.warning("No embedding backend available. Images will be processed but no embeddings will be created.")

    # concurrency
    workers = max(1, args.workers)
    log.info(f"Processing with {workers} workers (gpu={args.gpu})")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(process_single_image, p, detector, embedder, out_dir, args.resize_max, args.confidence, args.verbose): p for p in image_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing Photos", unit="img"):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                log.exception("Error processing image: %s", e)

    # collect embeddings
    all_emb = []
    mapping = []
    for meta in results:
        if not meta or "faces" not in meta:
            continue
        for face in meta["faces"]:
            if face.get("embedding") is not None:
                all_emb.append(np.array(face["embedding"], dtype=np.float32))
                mapping.append(face)

    # clustering
    if not args.no_clustering and SKLEARN_AVAILABLE and len(all_emb) > 0:
        X = np.vstack(all_emb)
        db = DBSCAN(eps=args.cluster_eps, min_samples=args.cluster_min).fit(X)
        labels = db.labels_
        for lbl, face in zip(labels, mapping):
            face["cluster"] = int(lbl)
        log.info(f"Clustering complete. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
    else:
        log.info("Skipping clustering (no sklearn or disabled or no embeddings)")

    # write aggregated index
    index_path = out_dir / "index.json"
    with open(index_path, "w", encoding="utf8") as f:
        json.dump({"images": results}, f, indent=2)

    log.info(f"Finished. Metadata written to {out_dir}")
    return out_dir

# -------------------------
# API helpers - placeholders
# -------------------------
def call_groq_api(prompt: str, api_key: Optional[str]=None, **kwargs):
    # Placeholder - user must provide groq client or httpx request
    # Example (pseudo):
    # import groq
    # client = groq.Client(api_key=api_key)
    # resp = client.generate(prompt)
    # return resp
    raise NotImplementedError("Add your Groq client call here using your GROQ_API_KEY env var")

def call_openrouter(prompt: str, api_key: Optional[str]=None, **kwargs):
    # Placeholder for OpenRouter
    raise NotImplementedError("Add your OpenRouter call here")

def call_ollama(prompt: str, **kwargs):
    # Placeholder for Ollama local client
    raise NotImplementedError("Add your Ollama call here")

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Ultra-Fast Multi-API Face Recognition System")
    p.add_argument("--input", "-i", required=True, help="Input directory with images")
    p.add_argument("--output", "-o", required=True, help="Output directory for results")
    p.add_argument("--detector", choices=["ultraface","face_recognition","auto"], default="auto")
    p.add_argument("--ultraface-model", default=os.getenv("ULTRAFACE_ONNX", ""), help="Path to UltraFace ONNX model")
    p.add_argument("--arcface-model", default=os.getenv("ARCFACE_ONNX", ""), help="Path to ArcFace ONNX model")
    p.add_argument("--uf-width", type=int, default=320, help="UltraFace input width")
    p.add_argument("--uf-height", type=int, default=240, help="UltraFace input height")
    p.add_argument("--workers", "-w", type=int, default=4, help="Number of worker threads")
    p.add_argument("--gpu", action="store_true", help="Use GPU providers if available (onnxruntime-gpu)")
    p.add_argument("--resize-max", type=int, default=1600, help="Max dimension to resize images to for speed")
    p.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    p.add_argument("--no-clustering", action="store_true", help="Disable clustering of embeddings")
    p.add_argument("--cluster-eps", type=float, default=0.6, help="DBSCAN eps")
    p.add_argument("--cluster-min", type=int, default=2, help="DBSCAN min_samples")
    p.add_argument("--verbose", action="store_true", help="Verbose debug prints")
    return p.parse_args()

def main():
    args = parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)
    start = time.time()
    try:
        out = run_pipeline(args)
    except Exception as e:
        log.exception("Pipeline failed: %s", e)
        sys.exit(2)
    log.info(f"Total time: {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()

