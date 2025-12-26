import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from src.inference.predictor import Predictor
from src.utils.config import get_device, load_config

# Available models configuration
AVAILABLE_MODELS = {
    "resnet18": {
        "display_name": "ResNet-18",
        "config": "configs/config_resnet18.yaml",
        "checkpoint": "models/checkpoints/resnet18/best_model.pth",
    },
    "resnet34": {
        "display_name": "ResNet-34",
        "config": "configs/config_resnet34.yaml",
        "checkpoint": "models/checkpoints/resnet34/best_model.pth",
    },
    "resnet50": {
        "display_name": "ResNet-50",
        "config": "configs/config_resnet50.yaml",
        "checkpoint": "models/checkpoints/resnet50/best_model.pth",
    },
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "config": "configs/config_efficientnet_b0.yaml",
        "checkpoint": "models/checkpoints/efficientnet_b0/best_model.pth",
    },
    "mobilenetv3_large": {
        "display_name": "MobileNetV3-Large",
        "config": "configs/config_mobilenetv3.yaml",
        "checkpoint": "models/checkpoints/mobilenetv3_large/best_model.pth",
    },
}

# Disease information in Vietnamese
DISEASE_INFO = {
    "ALGAL_LEAF_SPOT": {
        "name": "Bệnh đốm tảo",
        "short_desc": "Đốm xanh xám trên lá do tảo gây ra",
        "description": "Bệnh đốm tảo (Cephaleuros virescens) tạo ra các đốm tròn màu xanh xám hoặc nâu đỏ trên bề mặt lá. Thường xuất hiện trong điều kiện ẩm ướt, thông gió kém.",
        "treatment": "Cắt tỉa lá bệnh, cải thiện thông gió, phun thuốc gốc đồng như Bordeaux hoặc Copper oxychloride.",
    },
    "ALLOCARIDARA_ATTACK": {
        "name": "Bọ trĩ tấn công",
        "short_desc": "Lá bị hư hại do bọ trĩ gây ra",
        "description": "Bọ trĩ (Allocaridara malayensis) hút nhựa từ lá non, làm lá bị quăn, biến dạng và có màu nâu bạc. Gây ảnh hưởng nghiêm trọng đến sự phát triển của cây.",
        "treatment": "Phun thuốc trừ sâu như Imidacloprid, Abamectin. Loại bỏ lá bị nhiễm nặng và vệ sinh vườn thường xuyên.",
    },
    "HEALTHY_LEAF": {
        "name": "Lá khỏe mạnh",
        "short_desc": "Lá sầu riêng khỏe mạnh, không bệnh",
        "description": "Lá có màu xanh đậm đồng đều, bóng mượt, không có dấu hiệu của bệnh hay sâu hại. Đây là trạng thái lý tưởng của lá sầu riêng.",
        "treatment": None,
    },
    "LEAF_BLIGHT": {
        "name": "Bệnh cháy lá",
        "short_desc": "Lá bị cháy nâu, khô héo",
        "description": "Bệnh cháy lá (Rhizoctonia solani hoặc Phytophthora) làm lá chuyển màu nâu từ mép hoặc đầu lá, sau đó lan rộng và khô héo. Thường gặp trong mùa mưa.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc diệt nấm Mancozeb, Metalaxyl. Tránh tưới nước lên lá vào buổi chiều tối.",
    },
    "PHOMOPSIS_LEAF_SPOT": {
        "name": "Bệnh đốm lá Phomopsis",
        "short_desc": "Đốm nâu do nấm Phomopsis",
        "description": "Nấm Phomopsis gây ra các đốm tròn hoặc bất định màu nâu với viền đậm hơn. Bệnh thường xuất hiện ở lá già và lan sang lá non trong điều kiện ẩm ướt.",
        "treatment": "Phun thuốc diệt nấm như Carbendazim, Thiophanate-methyl. Loại bỏ lá rụng và lá bệnh để giảm nguồn bệnh.",
    },
}

DEFAULT_MODEL = os.environ.get("DURIAN_MODEL", "resnet18")
UPLOAD_DIR = Path("demo/static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
_predictors = {}


def get_predictor(model_name: str = None) -> Predictor:
    """Get or create a predictor for the specified model."""
    global _predictors

    if model_name is None:
        model_name = DEFAULT_MODEL

    if model_name not in AVAILABLE_MODELS:
        model_name = DEFAULT_MODEL

    if model_name not in _predictors:
        model_info = AVAILABLE_MODELS[model_name]
        config_path = model_info["config"]
        checkpoint_path = model_info["checkpoint"]

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint không tồn tại: {checkpoint_path}")

        config = load_config(config_path).raw
        device = get_device(prefer_gpu=True)
        _predictors[model_name] = Predictor(config=config, checkpoint_path=checkpoint_path, device=device)

    return _predictors[model_name]


def get_available_models_with_status():
    """Get available models with their availability status."""
    models = {}
    for name, info in AVAILABLE_MODELS.items():
        checkpoint_exists = Path(info["checkpoint"]).exists()
        models[name] = {
            "display_name": info["display_name"] + (" ✓" if checkpoint_exists else " (chưa train)"),
            "available": checkpoint_exists,
        }
    return models


@app.route("/", methods=["GET", "POST"])
def index():
    predictor = None
    prediction = None
    image_url = None
    error = None
    current_model = request.form.get("model", DEFAULT_MODEL) if request.method == "POST" else DEFAULT_MODEL

    available_models = get_available_models_with_status()

    try:
        predictor = get_predictor(current_model)
    except FileNotFoundError as e:
        error = (
            f"Không tìm thấy checkpoint cho model {current_model}. "
            "Vui lòng train model trước khi chạy demo hoặc chọn model khác."
        )

    if request.method == "POST" and predictor and not error:
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Vui lòng chọn ảnh lá sầu riêng để phân tích."
        else:
            filename = secure_filename(file.filename)
            saved_path = UPLOAD_DIR / f"{uuid4().hex}_{filename}"
            file.save(saved_path)

            try:
                result = predictor.predict(str(saved_path))
                prediction = {
                    "class": result["predicted_class"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                }
                image_url = url_for("static", filename=f"uploads/{saved_path.name}")
            except Exception as e:
                error = f"Lỗi khi phân tích ảnh: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        image_url=image_url,
        error=error,
        class_names=predictor.class_names if predictor else [],
        disease_info=DISEASE_INFO,
        available_models=available_models,
        current_model=current_model,
    )


if __name__ == "__main__":
    print("=" * 50)
    print("🌿 Durian Disease Detection Demo")
    print("=" * 50)
    print("\nAvailable models:")
    for name, info in get_available_models_with_status().items():
        status = "✓ Ready" if info["available"] else "✗ Not trained"
        print(f"  - {info['display_name'].replace(' ✓', '').replace(' (chưa train)', '')}: {status}")
    print("\n" + "=" * 50)
    print("Starting server at http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)


