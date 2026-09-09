"""Filename conventions shared by export inspection and attention training."""

import re


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
AUGMENTATION = re.compile(r"_(rot180|hflip|vflip)$")
VIDEO_FRAME = re.compile(r"^(.*sequence_\d+)_frame_(\d+)(?:_|$)")


def contact_key(path):
    """Group sensor-prefixed frames and augmentations into acquisitions."""
    stem = AUGMENTATION.sub("", path.stem)
    match = VIDEO_FRAME.match(stem)
    return "video::" + match[1] if match else "image::" + stem
