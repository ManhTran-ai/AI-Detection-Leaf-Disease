# 🌿 Durian Leaf Disease Detection

Hệ thống phát hiện bệnh lá sầu riêng sử dụng Deep Learning với nhiều kiến trúc CNN (ResNet18/34/50 và Custom CNN).

## 📋 Tổng Quan

- Phân loại 5 nhóm: `ALGAL_LEAF_SPOT`, `ALLOCARIDARA_ATTACK`, `HEALTHY_LEAF`, `LEAF_BLIGHT`, `PHOMOPSIS_LEAF_SPOT`
- Hỗ trợ huấn luyện, đánh giá, suy luận, trực quan hóa và Grad-CAM
- Pipeline đầy đủ: chuẩn bị dữ liệu → huấn luyện → đánh giá → suy luận → báo cáo

## 🚀 Cài Đặt

```bash
git clone <your-repo-url>
cd AI-Durian-Disease-Detection
python -m venv venv
venv\Scripts\activate  # Windows (hoặc source venv/bin/activate)
pip install -r requirements.txt
```

## 📊 Dataset

Cấu trúc mong đợi:

```
data/
  raw/
    ALGAL_LEAF_SPOT/
    ALLOCARIDARA_ATTACK/
    HEALTHY_LEAF/
    LEAF_BLIGHT/
    PHOMOPSIS_LEAF_SPOT/
```

Tách dữ liệu:

```bash
python scripts/split_dataset.py --source data/raw --destination data/processed --split 0.7 0.15 0.15
```

## ⚙️ Huấn Luyện & Đánh Giá

```bash
# Train + auto evaluate
python scripts/train_model.py --config configs/config_resnet18.yaml

# Đánh giá lại checkpoint
python scripts/evaluate_model.py \
  --config configs/config_resnet18.yaml \
  --checkpoint models/checkpoints/resnet18/best_model.pth \
  --split test
```

## 🤖 Suy Luận & Grad-CAM

```bash
python scripts/inference.py \
  --config configs/config_resnet18.yaml \
  --checkpoint models/checkpoints/resnet18/best_model.pth \
  --image path/to/leaf.jpg \
  --grad-cam outputs/grad_cam_outputs/sample.png
```

## 📈 Trực Quan Kết Quả

```bash
python scripts/visualize_results.py --config configs/config_resnet18.yaml
```

Artifacts:
- Lịch sử huấn luyện: `results/<model>/metrics/training_history.json`
- TensorBoard: `results/<model>/tensorboard`
- Plot loss/accuracy + confusion matrix: `results/<model>/plots`

## 🧱 Cấu Trúc Chính

- `configs/` – YAML cho từng mô hình
- `src/data` – dataset, tiền xử lý, dataloader
- `src/models` – ResNet wrapper, custom CNN, factory, utils
- `src/training` – trainer, callbacks, scheduler, metrics
- `src/evaluation` – evaluator & metrics
- `src/inference` – predictor & preprocessing
- `src/visualization` – plotting & Grad-CAM
- `scripts/` – entry points (split/train/eval/inference/visualize)
- `tests/` – unit tests cho data, models, evaluation, utils

## 🎯 Mục Tiêu Hiệu Năng

| Model    | Target Accuracy |
|----------|-----------------|
| ResNet18 | 82-88%          |
| ResNet34 | 88-92%          |
| ResNet50 | >92%            |
| CustomCNN| Baseline        |

Chi tiết và kết quả thực tế vui lòng xem `docs/RESULTS.md`.