# Face Swap Agent

This repository provides a simple, local face swapping agent that:

- Detects every face in a target image and lets you pick which one to replace.
- Detects faces across one or more source images so you can choose who should provide the new face.
- Swaps the face using MediaPipe Face Mesh for landmarks and OpenCV for alignment + blending.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python face_swapper.py \
  --target path/to/target.jpg \
  --sources path/to/source1.jpg path/to/source2.jpg \
  --output swapped.png
```

You can skip prompts by providing indices:

```bash
python face_swapper.py \
  --target path/to/target.jpg \
  --sources path/to/source1.jpg path/to/source2.jpg \
  --source-image-index 1 \
  --target-face 0 \
  --source-face 0 \
  --output swapped.png
```

## Notes

- The script works best with front-facing photos and good lighting.
- If no face is detected, check the image resolution or choose a different photo.
- The agent is fully local and does not require OpenAI API calls, but you can integrate additional API logic if desired.
- On some installations, MediaPipe may print a oneDNN/TensorFlow warning about numerical differences; this is informational and can be ignored.
