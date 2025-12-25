# Kế Hoạch Chi Tiết: Xây Dựng EfficientNet-B0 và MobileNetV3 với Fine-Tuning và AutoAugmentation

## Tổng Quan Dự Án

| Thông tin | Chi tiết |
|-----------|----------|
| **Models mới** | EfficientNet-B0, MobileNetV3-Large |
| **Kỹ thuật** | Fine-tuning layers, AutoAugmentation |
| **Số class** | 5 (ALGAL_LEAF_SPOT, ALLOCARIDARA_ATTACK, HEALTHY_LEAF, LEAF_BLIGHT, PHOMOPSIS_LEAF_SPOT) |
| **Framework** | PyTorch + torchvision |
| **Thời gian dự kiến** | 5-7 ngày |

---

## Phase 1: Chuẩn Bị và Nghiên Cứu (1 ngày)

### 1.1 Nghiên cứu kiến trúc model

| Model | Input Size | Params | Top-1 Acc (ImageNet) | Đặc điểm |
|-------|------------|--------|----------------------|----------|
| EfficientNet-B0 | 224×224 | 5.3M | 77.1% | Compound scaling, squeeze-excitation blocks |
| MobileNetV3-Large | 224×224 | 5.4M | 75.2% | Inverted residuals, h-swish activation |

### 1.2 Checklist chuẩn bị

- [ ] Cài đặt thêm package `timm` (PyTorch Image Models) nếu cần
- [ ] Kiểm tra GPU memory để xác định batch size phù hợp
- [ ] Backup code hiện tại trước khi thực hiện thay đổi

---

## Phase 2: Triển Khai AutoAugmentation (1-1.5 ngày)

### 2.1 Tạo module AutoAugment

| File | Đường dẫn | Mô tả |
|------|-----------|-------|
| `autoaugment.py` | `src/data/autoaugment.py` | Chứa các policy AutoAugment |

### 2.2 Các loại AutoAugment cần implement

| Policy | Mô tả | Sử dụng cho |
|--------|-------|-------------|
| **ImageNet Policy** | 25 sub-policies từ paper AutoAugment | Baseline |
| **RandAugment** | Random augmentation với N ops, magnitude M | Đơn giản, hiệu quả |
| **TrivialAugment** | Một augmentation ngẫu nhiên mỗi ảnh | Nhanh, không cần tune |

### 2.3 Cập nhật preprocessing.py

```
Thêm tham số mới vào config:
- augmentation.auto_augment: "imagenet" | "rand" | "trivial" | null
- augmentation.rand_augment_n: 2  (số operations)
- augmentation.rand_augment_m: 9  (magnitude)
```

### 2.4 Tasks cụ thể

| Task | File cần sửa | Độ ưu tiên |
|------|--------------|------------|
| Tạo AutoAugment policies | `src/data/autoaugment.py` | High |
| Tích hợp vào transform pipeline | `src/data/preprocessing.py` | High |
| Thêm config options | `configs/*.yaml` | Medium |
| Unit test AutoAugment | `tests/test_data.py` | Medium |

---

## Phase 3: Triển Khai EfficientNet-B0 (1 ngày)

### 3.1 Tạo file model

| File | Đường dẫn |
|------|-----------|
| `efficientnet.py` | `src/models/efficientnet.py` |

### 3.2 Cấu trúc EfficientNet với Fine-Tuning

```python
# Pseudo-code structure
#class EfficientNetB0:
#    - Load pretrained EfficientNet-B0 từ torchvision
#    - Thay thế classifier head:
#        * Dropout(p=dropout_rate)
#        * Linear(1280 -> num_classes)
#    - Fine-tuning strategy:
#        * freeze_backbone: bool
#        * unfreeze_layers: int (số block cuối cần unfreeze)
```

### 3.3 Fine-Tuning Strategy cho EfficientNet-B0

| Strategy | Layers được train | Use case |
|----------|-------------------|----------|
| **Freeze All** | Chỉ classifier | Dataset nhỏ, tránh overfit |
| **Gradual Unfreeze** | Block 6-7 + classifier | Balanced |
| **Full Fine-tune** | Tất cả layers | Dataset lớn, compute đủ |

### 3.4 Config parameters mới

```yaml
model:
  name: efficientnet_b0
  pretrained: true
  dropout: 0.3
  freeze_backbone: false
  unfreeze_from_block: 5  # Unfreeze từ block 5 trở đi (0-7)
```

---

## Phase 4: Triển Khai MobileNetV3 (1 ngày)

### 4.1 Tạo file model

| File | Đường dẫn |
|------|-----------|
| `mobilenet.py` | `src/models/mobilenet.py` |

### 4.2 Cấu trúc MobileNetV3 với Fine-Tuning

```python
# Pseudo-code structure
# class MobileNetV3:
#    - Load pretrained MobileNetV3-Large từ torchvision
#    - Thay thế classifier:
#        Linear(960 -> 1280)
#        Hardswish()
#        Dropout(p=dropout_rate)
#        Linear(1280 -> num_classes)
#    - Fine-tuning strategy tương tự EfficientNet
```

### 4.3 Fine-Tuning Strategy cho MobileNetV3

| Strategy | Layers được train | Mô tả |
|----------|-------------------|-------|
| **Freeze Features** | Chỉ classifier | Nhanh, ít overfit |
| **Partial Unfreeze** | InvertedResidual cuối + classifier | Recommended |
| **Full Fine-tune** | Tất cả | Cần regularization mạnh |

### 4.4 Config parameters mới

```yaml
model:
  name: mobilenetv3_large
  pretrained: true
  dropout: 0.2
  freeze_backbone: false
  unfreeze_from_layer: 12  # MobileNetV3 có 16 InvertedResidual blocks
```

---

## Phase 5: Cập Nhật Model Factory và Configs (0.5 ngày)

### 5.1 Files cần cập nhật

| File | Thay đổi |
|------|----------|
| `src/models/model_factory.py` | Thêm điều kiện cho efficientnet_b0, mobilenetv3_large |
| `src/models/__init__.py` | Export các model mới (nếu có) |

### 5.2 Tạo config files mới

| File | Dựa trên |
|------|----------|
| `configs/config_efficientnet_b0.yaml` | `config_resnet18.yaml` |
| `configs/config_mobilenetv3.yaml` | `config_resnet18.yaml` |

### 5.3 Hyperparameters đề xuất

| Parameter | EfficientNet-B0 | MobileNetV3-Large |
|-----------|-----------------|-------------------|
| `image_size` | 224 | 224 |
| `batch_size` | 32 | 48 |
| `learning_rate` | 1e-4 (fine-tune) | 1e-4 (fine-tune) |
| `weight_decay` | 1e-5 | 1e-5 |
| `dropout` | 0.3 | 0.2 |
| `optimizer` | AdamW | AdamW |
| `scheduler` | CosineAnnealingWarmRestarts | CosineAnnealing |
| `num_epochs` | 50 | 50 |
| `auto_augment` | "rand" | "rand" |

---

## Phase 6: Training và Fine-Tuning (1-2 ngày)

### 6.0 Kiểm Tra GPU/CUDA Trước Khi Training

```powershell
# Kiểm tra CUDA có sẵn không
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Kiểm tra GPU memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB') if torch.cuda.is_available() else print('No GPU')"
```

### 6.1 Training Pipeline với GPU/CUDA

#### Cách làm 

```powershell
# Train EfficientNet-B0 với GPU
python scripts/train_model.py --config configs/config_efficientnet_b0.yaml HOẶC
python -m scripts.train_model --config configs\config_efficientnet_b0.yaml --device cuda


# Train MobileNetV3-Large với GPU
python scripts/train_model.py --config configs/config_mobilenetv3.yaml HOẶC
python -m scripts.train_model --config configs\config_mobilenetv3.yaml --device cuda
```


### 6.2 Fine-Tuning 2-Stage (Recommended)

#### Stage 1: Train Classifier Only (5 epochs)

```powershell
# Tạo config tạm với freeze_backbone = true
# Hoặc sửa trực tiếp trong config file:
# model.freeze_backbone: true

python scripts/train_model.py --config configs/config_efficientnet_b0.yaml
# Sau 5 epochs, dừng lại (Ctrl+C) hoặc để early stopping
```

#### Stage 2: Unfreeze và Fine-tune Full Model

```powershell
# Sửa config:
# model.freeze_backbone: false
# training.learning_rate: 0.00001 (giảm 10x)
# Thêm --resume để tiếp tục từ checkpoint

python scripts/train_model.py --config configs/config_efficientnet_b0.yaml --resume models/checkpoints/efficientnet_b0/best_model.pth
```

### 6.3 Fine-Tuning Schedule Chi Tiết

| Stage | Epochs | Learning Rate | Layers | Batch Size |
|-------|--------|---------------|--------|------------|
| **Stage 1: Warmup** | 1-5 | 1e-4 | Classifier only (freeze backbone) | 32-48 |
| **Stage 2: Fine-tune** | 6-50 | 1e-5 → 1e-7 | Unfreeze partial backbone | 32 |

### 6.4 Monitoring với TensorBoard

```powershell
# Mở TensorBoard để theo dõi training
tensorboard --logdir results/efficientnet_b0/tensorboard --port 6006

# Hoặc theo dõi tất cả models
tensorboard --logdir results --port 6006
```

Sau đó mở browser tại: http://localhost:6006

### 6.5 Troubleshooting GPU

| Vấn đề | Giải pháp |
|--------|-----------|
| CUDA out of memory | Giảm `batch_size` xuống 16 hoặc 8 |
| GPU không được detect | Kiểm tra CUDA driver, reinstall PyTorch với CUDA |
| Training chậm | Bật `mixed_precision: true`, tăng `num_workers` |
| GPU utilization thấp | Tăng `batch_size`, tăng `num_workers` |

### 6.6 Monitoring Resources

| Metric | Tool | Đường dẫn |
|--------|------|-----------|
| Loss/Accuracy curves | TensorBoard | `results/<model>/tensorboard/` |
| Training history | JSON | `results/<model>/metrics/training_history.json` |
| GPU Usage | nvidia-smi | `nvidia-smi -l 1` (refresh mỗi 1 giây) |

---

## Phase 7: Đánh Giá và So Sánh (0.5-1 ngày)

### 7.1 Evaluation metrics

| Metric | Mô tả |
|--------|-------|
| Accuracy | Tổng quan performance |
| Precision/Recall/F1 | Per-class performance |
| Confusion Matrix | Phân tích lỗi |
| Inference Time | Tốc độ dự đoán |
| Model Size | Kích thước file .pth |

### 7.2 So sánh models

| Model | Accuracy | F1-Score | Inference (ms) | Size (MB) |
|-------|----------|----------|----------------|-----------|
| ResNet18 | TBD | TBD | TBD | TBD |
| ResNet34 | TBD | TBD | TBD | TBD |
| ResNet50 | TBD | TBD | TBD | TBD |
| **EfficientNet-B0** | TBD | TBD | TBD | TBD |
| **MobileNetV3-Large** | TBD | TBD | TBD | TBD |

### 7.3 Visualization

| Output | Script | Đường dẫn |
|--------|--------|-----------|
| Training plots | `scripts/visualize_results.py` | `results/<model>/plots/` |
| Comparison charts | `scripts/visualize_results.py` | `results/comparison/` |
| Grad-CAM | `src/visualization/grad_cam.py` | `outputs/grad_cam_outputs/` |

---

## Phase 8: Tích Hợp Demo và Documentation (0.5 ngày)

### 8.1 Cập nhật Demo App

| Task | File | Mô tả |
|------|------|-------|
| Thêm model selection | `demo/app.py` | Dropdown chọn model |
| Cập nhật UI | `demo/templates/index.html` | Hiển thị model info |

### 8.2 Cập nhật Grad-CAM

| Model | Target Layer |
|-------|--------------|
| EfficientNet-B0 | `model.features[-1]` (MBConv cuối) |
| MobileNetV3-Large | `model.features[-1]` (InvertedResidual cuối) |

### 8.3 Documentation

| File | Cập nhật |
|------|----------|
| `docs/RESULTS.md` | Thêm kết quả EfficientNet, MobileNetV3 |
| `docs/API_DOCUMENTATION.md` | Thêm API cho models mới |
| `README.md` | Cập nhật danh sách models |

---

## Tổng Hợp Files Cần Tạo/Sửa

### Files mới cần tạo

| File | Mô tả |
|------|-------|
| `src/data/autoaugment.py` | AutoAugmentation policies |
| `src/models/efficientnet.py` | EfficientNet-B0 model |
| `src/models/mobilenet.py` | MobileNetV3-Large model |
| `configs/config_efficientnet_b0.yaml` | Config cho EfficientNet |
| `configs/config_mobilenetv3.yaml` | Config cho MobileNetV3 |

### Files cần sửa

| File | Thay đổi |
|------|----------|
| `src/models/model_factory.py` | Đăng ký models mới |
| `src/data/preprocessing.py` | Tích hợp AutoAugment |
| `src/visualization/grad_cam.py` | Thêm target layers cho models mới |
| `tests/test_models.py` | Thêm unit tests |
| `tests/test_data.py` | Test AutoAugment |
| `requirements.txt` | Thêm dependencies nếu cần |
| `docs/RESULTS.md` | Cập nhật kết quả |

---

## Timeline Tổng Quan

```
Ngày 1: Phase 1 + Phase 2 (Chuẩn bị + AutoAugment)
Ngày 2: Phase 3 (EfficientNet-B0)
Ngày 3: Phase 4 (MobileNetV3)
Ngày 4: Phase 5 + Phase 6 bắt đầu (Factory + Training)
Ngày 5-6: Phase 6 tiếp tục (Training hoàn tất)
Ngày 7: Phase 7 + Phase 8 (Evaluation + Documentation)
```

---

## Checklist Hoàn Thành

- [x] **Phase 1**: Nghiên cứu và chuẩn bị
- [x] **Phase 2**: Implement AutoAugmentation
- [x] **Phase 3**: Implement EfficientNet-B0 với fine-tuning
- [x] **Phase 4**: Implement MobileNetV3-Large với fine-tuning
- [x] **Phase 5**: Cập nhật model factory và configs
- [ ] **Phase 6**: Training cả 2 models
- [ ] **Phase 7**: Evaluation và so sánh
- [ ] **Phase 8**: Tích hợp demo và documentation

---

## Ghi Chú Kỹ Thuật

### Fine-Tuning Best Practices

1. **Learning Rate**: Dùng LR thấp hơn 10-100x so với training from scratch
2. **Discriminative LR**: LR thấp cho layers đầu, cao hơn cho layers cuối
3. **Gradual Unfreezing**: Unfreeze dần từ classifier → backbone
4. **Regularization**: Tăng dropout, weight decay khi fine-tune full model

### AutoAugment Tips

1. **RandAugment** thường đủ tốt và dễ tune (chỉ cần N và M)
2. **N=2, M=9** là setting phổ biến cho ImageNet-pretrained models
3. Với dataset nhỏ, AutoAugment giúp giảm overfitting đáng kể

### GPU Memory Considerations

| Model | Batch Size 32 | Batch Size 48 |
|-------|---------------|---------------|
| EfficientNet-B0 | ~4GB VRAM | ~6GB VRAM |
| MobileNetV3-Large | ~3GB VRAM | ~4.5GB VRAM |

