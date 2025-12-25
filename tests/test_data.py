import numpy as np
from PIL import Image

from src.data.dataset import DurianLeafDataset
from src.data.autoaugment import (
    ImageNetPolicy,
    RandAugment,
    TrivialAugment,
    get_auto_augment,
    AutoAugmentTransform,
)


def _create_dummy_images(tmp_path, class_names):
    for cls in class_names:
        class_dir = tmp_path / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(class_dir / f"{cls}_{idx}.png")


def test_dataset_builds_samples(tmp_path):
    class_names = ["ALGAL_LEAF_SPOT", "ALLOCARIDARA_ATTACK"]
    _create_dummy_images(tmp_path, class_names)
    dataset = DurianLeafDataset(str(tmp_path), class_names)
    assert len(dataset) == 4
    first = dataset[0]
    assert "image" in first and "label" in first


# ============================================================================
# AutoAugment Tests
# ============================================================================

def test_imagenet_policy():
    """Test ImageNet AutoAugment policy."""
    policy = ImageNetPolicy()
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    result = policy(img)
    assert isinstance(result, Image.Image)
    assert result.size == img.size


def test_randaugment():
    """Test RandAugment with different N and M values."""
    for n in [1, 2, 3]:
        for m in [5, 9]:
            aug = RandAugment(n=n, m=m)
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            result = aug(img)
            assert isinstance(result, Image.Image)
            assert result.size == img.size


def test_trivialaugment():
    """Test TrivialAugment."""
    aug = TrivialAugment()
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    result = aug(img)
    assert isinstance(result, Image.Image)
    assert result.size == img.size


def test_get_auto_augment_factory():
    """Test get_auto_augment factory function."""
    # Test all policies
    for policy in ["imagenet", "rand", "trivial"]:
        aug = get_auto_augment(policy)
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = aug(img)
        assert isinstance(result, Image.Image)


def test_autoaugment_transform_wrapper():
    """Test AutoAugmentTransform for albumentations compatibility."""
    transform = AutoAugmentTransform(policy="rand", n=2, m=9, p=1.0)
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = transform(image=img_array)
    assert "image" in result
    assert result["image"].shape == img_array.shape


def test_randaugment_consistency():
    """Test that RandAugment produces valid output consistently."""
    aug = RandAugment(n=2, m=9)
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Run multiple times to ensure stability
    for _ in range(10):
        result = aug(img)
        assert isinstance(result, Image.Image)
        assert result.mode in ['RGB', 'L']


