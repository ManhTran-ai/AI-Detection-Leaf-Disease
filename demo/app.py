import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, redirect, render_template, request, url_for, session
from werkzeug.utils import secure_filename

from src.inference.predictor import Predictor
from src.detection.detector import DetectionPredictor
from src.detection.segmentor import SegmentationPredictor, draw_segmentation_masks
from src.detection.utils import CLASS_NAMES as DETECTION_CLASS_NAMES, DISEASE_INFO_VN, draw_detections
from src.utils.config import get_device, load_config

import cv2
import numpy as np

# =============================================================================
# Model Configuration
# =============================================================================

AVAILABLE_CLASSIFICATION_MODELS = {
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

AVAILABLE_DETECTION_MODELS = {
    "yolov8n": {
        "display_name": "YOLOv8-Nano (Nhanh)",
        "checkpoint": "models/detection/yolov8n_disease/weights/best.pt",
        "description": "Mô hình nhẹ, tốc độ nhanh nhất",
    },
    "yolov8s": {
        "display_name": "YOLOv8-Small (Cân bằng)",
        "checkpoint": "models/detection/yolov8s_disease/weights/best.pt",
        "description": "Cân bằng giữa tốc độ và độ chính xác",
    },
    "yolov8m": {
        "display_name": "YOLOv8-Medium (Chính xác)",
        "checkpoint": "models/detection/yolov8m_disease/weights/best.pt",
        "description": "Độ chính xác cao hơn, yêu cầu VRAM nhiều hơn",
    },
}

AVAILABLE_SEGMENTATION_MODELS = {
    "yolov8n-seg": {
        "display_name": "YOLOv8n-Seg (Nhanh)",
        "checkpoint": "models/segmentation/yolov8n_seg_disease/weights/best.pt",
        "description": "Instance Segmentation - Mô hình nhẹ, tốc độ nhanh",
    },
    "yolov8s-seg": {
        "display_name": "YOLOv8s-Seg (Cân bằng)",
        "checkpoint": "models/segmentation/yolov8s_seg_disease/weights/best.pt",
        "description": "Instance Segmentation - Cân bằng giữa tốc độ và độ chính xác",
    },
    "yolov8m-seg": {
        "display_name": "YOLOv8m-Seg (Chính xác)",
        "checkpoint": "models/segmentation/yolov8m_seg_disease/weights/best.pt",
        "description": "Instance Segmentation - Độ chính xác cao, VRAM cao hơn",
    },
}

# =============================================================================
# Disease Information
# =============================================================================

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

# =============================================================================
# Flask App Setup
# =============================================================================

DEFAULT_MODEL = os.environ.get("DURIAN_MODEL", "resnet18")
UPLOAD_DIR = Path("demo/static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "durian-disease-detection-secret-key-2024")
_class_predictors = {}
_detection_predictors = {}

# Detection color mapping for visualization
DETECTION_COLORS = {
    "ALGAL_LEAF_SPOT": "#00C800",
    "ALLOCARIDARA_ATTACK": "#FFA500",
    "HEALTHY_LEAF": "#00FF00",
    "LEAF_BLIGHT": "#FF0000",
    "PHOMOPSIS_LEAF_SPOT": "#C800C8",
}


# =============================================================================
# Predictor Management
# =============================================================================

def get_classification_predictor(model_name: str) -> Predictor:
    """Get or create a classification predictor."""
    global _class_predictors

    if model_name not in AVAILABLE_CLASSIFICATION_MODELS:
        model_name = DEFAULT_MODEL

    if model_name not in _class_predictors:
        model_info = AVAILABLE_CLASSIFICATION_MODELS[model_name]
        config_path = model_info["config"]
        checkpoint_path = model_info["checkpoint"]

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint không tồn tại: {checkpoint_path}")

        config = load_config(config_path).raw
        device = get_device(prefer_gpu=True)
        _class_predictors[model_name] = Predictor(
            config=config, checkpoint_path=checkpoint_path, device=device
        )

    return _class_predictors[model_name]


def get_detection_predictor(model_name: str) -> DetectionPredictor:
    """Get or create a detection predictor."""
    global _detection_predictors

    if model_name not in AVAILABLE_DETECTION_MODELS:
        model_name = "yolov8n"

    if model_name not in _detection_predictors:
        model_info = AVAILABLE_DETECTION_MODELS[model_name]
        checkpoint_path = model_info["checkpoint"]

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Detection model không tồn tại: {checkpoint_path}")

        device = "0" if get_device().type == "cuda" else "cpu"
        _detection_predictors[model_name] = DetectionPredictor(
            model_path=checkpoint_path,
            class_names=DETECTION_CLASS_NAMES,
            conf_threshold=0.25,
            iou_threshold=0.45,
            device=device,
        )

    return _detection_predictors[model_name]


_seg_predictors = {}


def get_segmentation_predictor(model_name: str) -> SegmentationPredictor:
    """Get or create a segmentation predictor."""
    global _seg_predictors

    if model_name not in AVAILABLE_SEGMENTATION_MODELS:
        model_name = "yolov8n-seg"

    if model_name not in _seg_predictors:
        model_info = AVAILABLE_SEGMENTATION_MODELS[model_name]
        checkpoint_path = model_info["checkpoint"]

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Segmentation model không tồn tại: {checkpoint_path}")

        device = "0" if get_device().type == "cuda" else "cpu"
        _seg_predictors[model_name] = SegmentationPredictor(
            model_path=checkpoint_path,
            class_names=DETECTION_CLASS_NAMES,
            conf_threshold=0.25,
            iou_threshold=0.45,
            device=device,
        )

    return _seg_predictors[model_name]


def get_available_models_with_status():
    """Get available models with their availability status."""
    cls_models = {}
    for name, info in AVAILABLE_CLASSIFICATION_MODELS.items():
        checkpoint_exists = Path(info["checkpoint"]).exists()
        cls_models[name] = {
            "display_name": info["display_name"],
            "available": checkpoint_exists,
            "status_text": "✓" if checkpoint_exists else "(chưa train)",
        }

    det_models = {}
    for name, info in AVAILABLE_DETECTION_MODELS.items():
        checkpoint_exists = Path(info["checkpoint"]).exists()
        det_models[name] = {
            "display_name": info["display_name"],
            "description": info["description"],
            "available": checkpoint_exists,
            "status_text": "✓" if checkpoint_exists else "(chưa train)",
        }

    seg_models = {}
    for name, info in AVAILABLE_SEGMENTATION_MODELS.items():
        checkpoint_exists = Path(info["checkpoint"]).exists()
        seg_models[name] = {
            "display_name": info["display_name"],
            "description": info["description"],
            "available": checkpoint_exists,
            "status_text": "✓" if checkpoint_exists else "(chưa train)",
        }

    return cls_models, det_models, seg_models


# =============================================================================
# Routes
# =============================================================================

@app.route("/", methods=["GET"])
def index():
    detection_mode = request.args.get("mode", session.get("last_mode", "classification"))
    current_cls_model = request.args.get("cls_model", session.get("last_cls_model", DEFAULT_MODEL))
    current_det_model = request.args.get("det_model", session.get("last_det_model", "yolov8n"))
    current_seg_model = request.args.get("seg_model", session.get("last_seg_model", "yolov8n-seg"))

    prediction = session.pop("prediction", None)
    detection_result = session.pop("detection_result", None)
    segmentation_result = session.pop("segmentation_result", None)
    image_url = session.pop("image_url", None)
    annotated_image_url = session.pop("annotated_image_url", None)
    error = session.pop("error", None)

    available_cls_models, available_det_models, available_seg_models = get_available_models_with_status()

    return render_template(
        "index.html",
        prediction=prediction,
        detection_result=detection_result,
        segmentation_result=segmentation_result,
        image_url=image_url,
        annotated_image_url=annotated_image_url,
        error=error,
        disease_info=DISEASE_INFO,
        detection_colors=DETECTION_COLORS,
        available_cls_models=available_cls_models,
        available_det_models=available_det_models,
        available_seg_models=available_seg_models,
        current_cls_model=current_cls_model,
        current_det_model=current_det_model,
        current_seg_model=current_seg_model,
        detection_mode=detection_mode,
    )


@app.route("/predict", methods=["POST"])
def predict():
    detection_mode = request.form.get("mode", "classification")
    current_cls_model = request.form.get("cls_model", DEFAULT_MODEL)
    current_det_model = request.form.get("det_model", "yolov8n")
    current_seg_model = request.form.get("seg_model", "yolov8n-seg")

    session["last_mode"] = detection_mode
    session["last_cls_model"] = current_cls_model
    session["last_det_model"] = current_det_model
    session["last_seg_model"] = current_seg_model

    file = request.files.get("image")
    if not file or file.filename == "":
        session["error"] = "Vui lòng chọn ảnh lá sầu riêng để phân tích."
        return redirect(url_for("index", mode=detection_mode))

    filename = secure_filename(file.filename)
    saved_path = UPLOAD_DIR / f"{uuid4().hex}_{filename}"
    file.save(saved_path)
    image_url = url_for("static", filename=f"uploads/{saved_path.name}")

    predictor = None
    det_predictor = None
    seg_predictor = None
    annotated_image_url = None
    prediction = None
    detection_result = None
    segmentation_result = None

    try:
        if detection_mode == "detection":
            det_predictor = get_detection_predictor(current_det_model)
        elif detection_mode == "segmentation":
            seg_predictor = get_segmentation_predictor(current_seg_model)
        else:
            predictor = get_classification_predictor(current_cls_model)
    except FileNotFoundError as e:
        session["error"] = str(e)
        return redirect(url_for("index", mode=detection_mode))

    try:
        if detection_mode == "segmentation" and seg_predictor:
            result = seg_predictor.predict(str(saved_path), return_image=True)
            predictions_list = result["predictions"]
            annotated_img = result["annotated_image"]

            if annotated_img is not None:
                annotated_name = f"seg_{uuid4().hex}_{filename}"
                annotated_path = UPLOAD_DIR / annotated_name
                annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(annotated_path), annotated_bgr)
                annotated_image_url = url_for("static", filename=f"uploads/{annotated_name}")

            segmentation_result = {
                "num_detections": result["num_detections"],
                "predictions": predictions_list,
                "disease_counts": {},
                "total_area": sum(p.get("mask_area_pixels", 0) for p in predictions_list),
            }

            for pred in predictions_list:
                cls_name = pred["class_name"]
                segmentation_result["disease_counts"][cls_name] = (
                    segmentation_result["disease_counts"].get(cls_name, 0) + 1
                )

            session["segmentation_result"] = segmentation_result

        elif detection_mode == "detection" and det_predictor:
            result = det_predictor.predict(str(saved_path), return_image=True)
            predictions_list = result["predictions"]
            annotated_img = result["annotated_image"]

            if annotated_img is not None:
                annotated_name = f"det_{uuid4().hex}_{filename}"
                annotated_path = UPLOAD_DIR / annotated_name
                cv2.imwrite(str(annotated_path), annotated_img)
                annotated_image_url = url_for("static", filename=f"uploads/{annotated_name}")

            detection_result = {
                "num_detections": result["num_detections"],
                "predictions": predictions_list,
                "disease_counts": {},
            }

            for pred in predictions_list:
                cls_name = pred["class_name"]
                detection_result["disease_counts"][cls_name] = (
                    detection_result["disease_counts"].get(cls_name, 0) + 1
                )

            session["detection_result"] = detection_result

        elif detection_mode == "classification" and predictor:
            result = predictor.predict(str(saved_path))
            prediction = {
                "class": result["predicted_class"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            }
            session["prediction"] = prediction

        session["image_url"] = image_url
        session["annotated_image_url"] = annotated_image_url

    except Exception as e:
        session["error"] = f"Lỗi khi phân tích ảnh: {str(e)}"
        session["image_url"] = image_url
        session["annotated_image_url"] = None
        return redirect(url_for("index", mode=detection_mode))

    return redirect(url_for("index", mode=detection_mode))


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Durian Disease Detection Demo - Classification + Detection + Segmentation")
    print("=" * 60)

    cls_models, det_models, seg_models = get_available_models_with_status()

    print("\n[Classification Models]")
    for name, info in cls_models.items():
        print(f"  {info['display_name']}: {info['status_text']}")

    print("\n[Detection Models - YOLOv8]")
    for name, info in det_models.items():
        print(f"  {info['display_name']}: {info['status_text']}")
        print(f"    {info['description']}")

    print("\n[Segmentation Models - YOLOv8 Instance Segmentation]")
    for name, info in seg_models.items():
        print(f"  {info['display_name']}: {info['status_text']}")
        print(f"    {info['description']}")

    print("\n" + "=" * 60)
    print("Starting server at http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
