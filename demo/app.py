import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from src.inference.predictor import Predictor
from src.utils.config import get_device, load_config

CONFIG_PATH = os.environ.get("DURIAN_CONFIG", "configs/config_resnet18.yaml")
CHECKPOINT_PATH = os.environ.get(
    "DURIAN_CHECKPOINT",
    "models/checkpoints/resnet18/best_model.pth",
)
UPLOAD_DIR = Path("demo/static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
_predictor = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        config = load_config(CONFIG_PATH).raw
        # Prefer running the demo on GPU when available to leverage your NVIDIA CUDA setup.
        # Falls back to CPU automatically if a compatible GPU is not detected by PyTorch.
        device = get_device(prefer_gpu=True)
        _predictor = Predictor(config=config, checkpoint_path=CHECKPOINT_PATH, device=device)
    return _predictor


@app.route("/", methods=["GET", "POST"])
def index():
    predictor = None
    prediction = None
    image_url = None
    error = None

    try:
        predictor = get_predictor()
    except FileNotFoundError:
        error = (
            f"Không tìm thấy checkpoint tại {CHECKPOINT_PATH}. "
            "Vui lòng train model trước khi chạy demo."
        )

    if request.method == "POST" and predictor and not error:
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Vui lòng chọn ảnh lá sầu riêng."
        else:
            filename = secure_filename(file.filename)
            saved_path = UPLOAD_DIR / f"{uuid4().hex}_{filename}"
            file.save(saved_path)
            result = predictor.predict(str(saved_path))
            prediction = {
                "class": result["predicted_class"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            }
            image_url = url_for("static", filename=f"uploads/{saved_path.name}")

    return render_template(
        "index.html",
        prediction=prediction,
        image_url=image_url,
        error=error,
        class_names=predictor.class_names if predictor else [],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


