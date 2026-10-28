import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import mediapipe as mp
import timm
import torch

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Torch:", torch.__version__, "runtime CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    torch.ones(1, device="cuda").add_(1)
    print("CUDA tensor operation: OK")
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__, "Hands API:", hasattr(mp, "solutions"))
print("timm:", timm.__version__, "FastViT-MA36 registered:", "fastvit_ma36" in timm.list_models())
