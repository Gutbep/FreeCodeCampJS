#!/usr/bin/env python3
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

KEYPOINT_INDEXES = [33, 263, 1, 61, 291]


@dataclass
class FaceData:
    landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]
    keypoints: np.ndarray


@dataclass
class ImageFaces:
    path: Path
    image_bgr: np.ndarray
    faces: List[FaceData]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swap faces between images using MediaPipe Face Mesh and OpenCV."
    )
    parser.add_argument("--target", required=True, help="Path to target image (receives the new face).")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="One or more source images (provide faces to swap onto target).",
    )
    parser.add_argument(
        "--output",
        default="face_swap_output.png",
        help="Output path for the swapped image.",
    )
    parser.add_argument(
        "--target-face",
        type=int,
        default=None,
        help="Face index on target image to replace (0-based).",
    )
    parser.add_argument(
        "--source-image-index",
        type=int,
        default=None,
        help="Which source image to use when multiple are provided (0-based).",
    )
    parser.add_argument(
        "--source-face",
        type=int,
        default=None,
        help="Face index on source image to use (0-based).",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def detect_faces(image_bgr: np.ndarray) -> List[FaceData]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    results = face_mesh.process(image_rgb)
    face_mesh.close()

    faces: List[FaceData] = []
    if not results.multi_face_landmarks:
        return faces

    for face_landmarks in results.multi_face_landmarks:
        coords = []
        for landmark in face_landmarks.landmark:
            x = min(max(int(landmark.x * width), 0), width - 1)
            y = min(max(int(landmark.y * height), 0), height - 1)
            coords.append([x, y])
        landmarks = np.array(coords, dtype=np.int32)
        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)
        bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
        keypoints = landmarks[KEYPOINT_INDEXES]
        faces.append(FaceData(landmarks=landmarks, bbox=bbox, keypoints=keypoints))

    return faces


def summarize_faces(image_faces: ImageFaces) -> str:
    lines = [f"Detected {len(image_faces.faces)} face(s) in {image_faces.path}"]
    for idx, face in enumerate(image_faces.faces):
        x_min, y_min, x_max, y_max = face.bbox
        lines.append(
            f"  [{idx}] bbox=({x_min}, {y_min})-({x_max}, {y_max})"
        )
    return "\n".join(lines)


def choose_index(prompt: str, maximum: int, default: Optional[int]) -> int:
    if maximum <= 1:
        return 0
    if default is not None:
        if 0 <= default < maximum:
            return default
        raise ValueError(f"Index {default} is out of range (0..{maximum - 1}).")

    while True:
        response = input(f"{prompt} [0-{maximum - 1}]: ")
        try:
            value = int(response)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if 0 <= value < maximum:
            return value
        print(f"Please choose a number between 0 and {maximum - 1}.")


def pick_source_image(sources: List[ImageFaces], index: Optional[int]) -> ImageFaces:
    if len(sources) == 1:
        return sources[0]
    if index is not None:
        if 0 <= index < len(sources):
            return sources[index]
        raise ValueError(f"Source image index {index} is out of range.")

    print("Available source images:")
    for idx, source in enumerate(sources):
        print(f"  [{idx}] {source.path} ({len(source.faces)} face(s))")

    chosen = choose_index("Select source image", len(sources), None)
    return sources[chosen]


def compute_affine_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.estimateAffinePartial2D(source_points, target_points, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError("Unable to compute affine transform between faces.")
    return matrix


def warp_face(
    source_image: np.ndarray,
    source_face: FaceData,
    target_shape: Tuple[int, int],
    matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = target_shape
    warped = cv2.warpAffine(source_image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    transformed_landmarks = cv2.transform(source_face.landmarks[None, :, :].astype(np.float32), matrix)[
        0
    ].astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(transformed_landmarks)
    cv2.fillConvexPoly(mask, hull, 255)
    return warped, mask


def blend_faces(
    target_image: np.ndarray,
    warped_source: np.ndarray,
    mask: np.ndarray,
    target_face: FaceData,
) -> np.ndarray:
    x_min, y_min, x_max, y_max = target_face.bbox
    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)
    return cv2.seamlessClone(warped_source, target_image, mask, (center_x, center_y), cv2.NORMAL_CLONE)


def main() -> int:
    args = parse_args()
    target_path = Path(args.target)
    source_paths = [Path(path) for path in args.sources]

    try:
        target_image = load_image(target_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    target_faces = detect_faces(target_image)
    target_info = ImageFaces(path=target_path, image_bgr=target_image, faces=target_faces)
    print(summarize_faces(target_info))

    if not target_faces:
        print("No faces detected in target image.", file=sys.stderr)
        return 1

    source_images: List[ImageFaces] = []
    for path in source_paths:
        try:
            image_bgr = load_image(path)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        faces = detect_faces(image_bgr)
        source_images.append(ImageFaces(path=path, image_bgr=image_bgr, faces=faces))
        print(summarize_faces(source_images[-1]))

    chosen_source = pick_source_image(source_images, args.source_image_index)
    if not chosen_source.faces:
        print("No faces detected in chosen source image.", file=sys.stderr)
        return 1

    try:
        target_index = choose_index(
            "Select target face index", len(target_faces), args.target_face
        )
        source_index = choose_index(
            "Select source face index", len(chosen_source.faces), args.source_face
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    target_face = target_faces[target_index]
    source_face = chosen_source.faces[source_index]

    transform = compute_affine_transform(source_face.keypoints.astype(np.float32), target_face.keypoints.astype(np.float32))
    warped_source, mask = warp_face(
        chosen_source.image_bgr,
        source_face,
        target_image.shape[:2],
        transform,
    )
    blended = blend_faces(target_image, warped_source, mask, target_face)

    output_path = Path(args.output)
    cv2.imwrite(str(output_path), blended)
    print(f"Saved swapped image to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
