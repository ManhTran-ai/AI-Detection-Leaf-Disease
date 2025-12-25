"""
AutoAugmentation policies for data augmentation.
Includes ImageNet Policy, RandAugment, and TrivialAugment implementations.
"""

from typing import Callable, List, Optional
import random

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================

def _int_parameter(level: int, max_val: int) -> int:
    """Helper to scale integer parameters based on magnitude level (0-10)."""
    return int(level * max_val / 10)


def _float_parameter(level: int, max_val: float) -> float:
    """Helper to scale float parameters based on magnitude level (0-10)."""
    return float(level) * max_val / 10


def _apply_op(img: Image.Image, name: str, magnitude: float) -> Image.Image:
    """Apply a single augmentation operation to an image."""

    if name == "Identity":
        return img

    elif name == "AutoContrast":
        return ImageOps.autocontrast(img)

    elif name == "Equalize":
        return ImageOps.equalize(img)

    elif name == "Invert":
        return ImageOps.invert(img)

    elif name == "Rotate":
        # magnitude: degrees
        angle = magnitude if random.random() > 0.5 else -magnitude
        return img.rotate(angle, fillcolor=(128, 128, 128))

    elif name == "Posterize":
        # magnitude: bits (1-8)
        bits = max(1, int(8 - magnitude))
        return ImageOps.posterize(img, bits)

    elif name == "Solarize":
        # magnitude: threshold (0-255)
        threshold = int(256 - magnitude)
        return ImageOps.solarize(img, threshold)

    elif name == "SolarizeAdd":
        # Add value then solarize
        threshold = 128
        add_val = int(magnitude)
        img_array = np.array(img).astype(np.int32)
        img_array = img_array + add_val
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        return ImageOps.solarize(img, threshold)

    elif name == "Color":
        # magnitude: factor (0-1.8+)
        factor = 1.0 + magnitude if random.random() > 0.5 else 1.0 - magnitude
        factor = max(0.0, factor)
        return ImageEnhance.Color(img).enhance(factor)

    elif name == "Contrast":
        factor = 1.0 + magnitude if random.random() > 0.5 else 1.0 - magnitude
        factor = max(0.0, factor)
        return ImageEnhance.Contrast(img).enhance(factor)

    elif name == "Brightness":
        factor = 1.0 + magnitude if random.random() > 0.5 else 1.0 - magnitude
        factor = max(0.0, factor)
        return ImageEnhance.Brightness(img).enhance(factor)

    elif name == "Sharpness":
        factor = 1.0 + magnitude if random.random() > 0.5 else 1.0 - magnitude
        factor = max(0.0, factor)
        return ImageEnhance.Sharpness(img).enhance(factor)

    elif name == "ShearX":
        # magnitude: shear factor
        shear = magnitude if random.random() > 0.5 else -magnitude
        return img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), fillcolor=(128, 128, 128))

    elif name == "ShearY":
        shear = magnitude if random.random() > 0.5 else -magnitude
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0), fillcolor=(128, 128, 128))

    elif name == "TranslateX":
        # magnitude: pixels or fraction
        pixels = magnitude if random.random() > 0.5 else -magnitude
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=(128, 128, 128))

    elif name == "TranslateY":
        pixels = magnitude if random.random() > 0.5 else -magnitude
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=(128, 128, 128))

    elif name == "Cutout":
        # magnitude: size of cutout region
        h, w = img.size
        size = int(magnitude)
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        x1 = max(0, x - size // 2)
        y1 = max(0, y - size // 2)
        x2 = min(w, x + size // 2)
        y2 = min(h, y + size // 2)
        img_array = np.array(img)
        img_array[y1:y2, x1:x2, :] = 128
        return Image.fromarray(img_array)

    elif name == "Blur":
        return img.filter(ImageFilter.GaussianBlur(radius=magnitude))

    else:
        raise ValueError(f"Unknown operation: {name}")


# ============================================================================
# IMAGENET AUTO AUGMENT POLICY
# ============================================================================

# 25 sub-policies from the AutoAugment paper for ImageNet
IMAGENET_POLICY = [
    [("Posterize", 0.4, 8), ("Rotate", 0.6, 9)],
    [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
    [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
    [("Posterize", 0.6, 7), ("Posterize", 0.6, 6)],
    [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
    [("Equalize", 0.4, 4), ("Rotate", 0.8, 8)],
    [("Solarize", 0.6, 3), ("Equalize", 0.6, 7)],
    [("Posterize", 0.8, 5), ("Equalize", 1.0, 2)],
    [("Rotate", 0.2, 3), ("Solarize", 0.6, 8)],
    [("Equalize", 0.6, 8), ("Posterize", 0.4, 6)],
    [("Rotate", 0.8, 8), ("Color", 0.4, 0)],
    [("Rotate", 0.4, 9), ("Equalize", 0.6, 2)],
    [("Equalize", 0.0, 7), ("Equalize", 0.8, 8)],
    [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
    [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
    [("Rotate", 0.8, 8), ("Color", 1.0, 2)],
    [("Color", 0.8, 8), ("Solarize", 0.8, 7)],
    [("Sharpness", 0.4, 7), ("Invert", 0.6, 8)],
    [("ShearX", 0.6, 5), ("Equalize", 1.0, 9)],
    [("Color", 0.4, 0), ("Equalize", 0.6, 3)],
    [("Equalize", 1.0, 8), ("Solarize", 0.6, 6)],
    [("Solarize", 0.8, 8), ("Equalize", 0.8, 4)],
    [("TranslateY", 0.2, 9), ("TranslateY", 0.6, 9)],
    [("Equalize", 0.6, 5), ("Equalize", 0.8, 8)],
    [("Equalize", 0.6, 8), ("AutoContrast", 0.4, 4)],
]


class ImageNetPolicy:
    """AutoAugment policy from the AutoAugment paper for ImageNet.

    Applies one of 25 sub-policies randomly, each consisting of 2 operations
    with specific probabilities and magnitudes.
    """

    def __init__(self):
        self.policies = IMAGENET_POLICY

    def __call__(self, img: Image.Image) -> Image.Image:
        policy = random.choice(self.policies)

        for op_name, prob, magnitude in policy:
            if random.random() < prob:
                # Scale magnitude based on operation
                mag = self._get_magnitude(op_name, magnitude)
                img = _apply_op(img, op_name, mag)

        return img

    def _get_magnitude(self, op_name: str, level: int) -> float:
        """Get the actual magnitude value for an operation."""
        magnitude_ranges = {
            "Rotate": 30,  # degrees
            "Posterize": 4,  # bits reduction
            "Solarize": 256,  # threshold
            "Color": 0.9,
            "Contrast": 0.9,
            "Brightness": 0.9,
            "Sharpness": 0.9,
            "ShearX": 0.3,
            "ShearY": 0.3,
            "TranslateX": 100,  # pixels
            "TranslateY": 100,
        }
        max_val = magnitude_ranges.get(op_name, 1.0)
        return _float_parameter(level, max_val)


# ============================================================================
# RANDAUGMENT
# ============================================================================

# Available operations for RandAugment
RAND_AUGMENT_OPS = [
    "Identity",
    "AutoContrast",
    "Equalize",
    "Rotate",
    "Solarize",
    "Color",
    "Posterize",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
]


class RandAugment:
    """RandAugment: Practical automated data augmentation.

    Randomly selects N operations from a fixed set and applies them
    with a fixed magnitude M.

    Args:
        n: Number of operations to apply per image (default: 2)
        m: Magnitude for all operations, range 0-10 (default: 9)
        ops: List of operations to choose from (default: RAND_AUGMENT_OPS)
    """

    def __init__(self, n: int = 2, m: int = 9, ops: Optional[List[str]] = None):
        self.n = n
        self.m = m
        self.ops = ops if ops is not None else RAND_AUGMENT_OPS

    def __call__(self, img: Image.Image) -> Image.Image:
        ops = random.choices(self.ops, k=self.n)

        for op_name in ops:
            mag = self._get_magnitude(op_name, self.m)
            img = _apply_op(img, op_name, mag)

        return img

    def _get_magnitude(self, op_name: str, level: int) -> float:
        """Scale magnitude based on operation type."""
        magnitude_ranges = {
            "Rotate": 30,
            "Posterize": 4,
            "Solarize": 256,
            "Color": 0.9,
            "Contrast": 0.9,
            "Brightness": 0.9,
            "Sharpness": 0.9,
            "ShearX": 0.3,
            "ShearY": 0.3,
            "TranslateX": 100,
            "TranslateY": 100,
        }
        max_val = magnitude_ranges.get(op_name, 1.0)
        return _float_parameter(level, max_val)


# ============================================================================
# TRIVIALAUGMENT
# ============================================================================

class TrivialAugment:
    """TrivialAugment: Tuning-free augmentation.

    Applies exactly one randomly selected operation with a random magnitude.
    No tuning required - simpler and often as effective as RandAugment.
    """

    def __init__(self, ops: Optional[List[str]] = None):
        self.ops = ops if ops is not None else RAND_AUGMENT_OPS

    def __call__(self, img: Image.Image) -> Image.Image:
        op_name = random.choice(self.ops)
        magnitude_level = random.randint(0, 10)
        mag = self._get_magnitude(op_name, magnitude_level)
        return _apply_op(img, op_name, mag)

    def _get_magnitude(self, op_name: str, level: int) -> float:
        """Scale magnitude based on operation type."""
        magnitude_ranges = {
            "Rotate": 30,
            "Posterize": 4,
            "Solarize": 256,
            "Color": 0.9,
            "Contrast": 0.9,
            "Brightness": 0.9,
            "Sharpness": 0.9,
            "ShearX": 0.3,
            "ShearY": 0.3,
            "TranslateX": 100,
            "TranslateY": 100,
        }
        max_val = magnitude_ranges.get(op_name, 1.0)
        return _float_parameter(level, max_val)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_auto_augment(policy: str, n: int = 2, m: int = 9) -> Callable:
    """Get an auto augmentation transform based on policy name.

    Args:
        policy: One of "imagenet", "rand", "trivial"
        n: Number of operations for RandAugment (default: 2)
        m: Magnitude for RandAugment (default: 9)

    Returns:
        An augmentation callable that takes and returns PIL Image

    Example:
        >>> aug = get_auto_augment("rand", n=2, m=9)
        >>> augmented_img = aug(img)
    """
    policy = policy.lower()

    if policy == "imagenet":
        return ImageNetPolicy()
    elif policy == "rand":
        return RandAugment(n=n, m=m)
    elif policy == "trivial":
        return TrivialAugment()
    else:
        raise ValueError(f"Unknown auto augment policy: {policy}. "
                        f"Choose from: imagenet, rand, trivial")


# ============================================================================
# ALBUMENTATIONS WRAPPER
# ============================================================================

class AutoAugmentTransform:
    """Albumentations-compatible wrapper for AutoAugment policies.

    This allows integration with the existing albumentations pipeline.

    Args:
        policy: One of "imagenet", "rand", "trivial"
        n: Number of operations for RandAugment
        m: Magnitude for RandAugment
        p: Probability of applying the transform
    """

    def __init__(self, policy: str = "rand", n: int = 2, m: int = 9, p: float = 1.0):
        self.policy = policy
        self.n = n
        self.m = m
        self.p = p
        self.transform = get_auto_augment(policy, n, m)

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        """Apply transform to numpy image array."""
        if random.random() < self.p:
            # Convert numpy to PIL
            pil_img = Image.fromarray(image)
            # Apply auto augment
            pil_img = self.transform(pil_img)
            # Convert back to numpy
            image = np.array(pil_img)

        return {"image": image, **kwargs}

    def __repr__(self):
        return f"AutoAugmentTransform(policy={self.policy}, n={self.n}, m={self.m}, p={self.p})"

