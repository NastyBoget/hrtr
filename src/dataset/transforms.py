import albumentations as A
import cv2

transforms = A.Compose([
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.25, always_apply=False),
    A.Rotate(limit=3, border_mode=cv2.BORDER_CONSTANT, p=1.0),
    A.Cutout(num_holes=10, p=0.75),
    A.GridDistortion(distort_limit=0.15, border_mode=cv2.BORDER_CONSTANT, p=0.75),
    A.Blur(blur_limit=3, p=0.5),
    A.JpegCompression(quality_lower=75, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.75)
])
