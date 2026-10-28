from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from models.simplehand import SimpleHand


class SimpleHandPredictor:
    def __init__(self, checkpoint, mano_right, device="cuda", amp=False):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Check NVIDIA driver and CUDA PyTorch wheel, or use --device cpu.")
        checkpoint, mano_right = Path(checkpoint), Path(mano_right)
        for path in (checkpoint, mano_right):
            if not path.is_file():
                raise FileNotFoundError(f"Required asset missing: {path}. See README.md.")
        self.model = SimpleHand(mano_right)
        # Trusted official checkpoint only. No fallback to random initialization.
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise ValueError("Expected a state dictionary or checkpoint dictionary.")
        state = loaded.get("state_dict", loaded.get("model_state_dict", loaded))
        state = {k.removeprefix("module."): v for k, v in state.items()}
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError("Checkpoint does not match official FastViT-MA36 SimpleHand. "
                               "Use epoch_200_rerun1 or rerun2 from the author's README. "
                               "Do not use ImageNet-only FastViT weights.\n" + str(exc)) from exc
        self.model.to(self.device).eval()
        self.faces = self.model.faces.cpu().numpy().copy()
        self.amp = amp and self.device.type == "cuda"
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.warmup()

    @torch.inference_mode()
    def predict(self, crop_bgr):
        array = np.ascontiguousarray(crop_bgr.transpose(2, 0, 1))
        image = torch.from_numpy(array).unsqueeze(0).to(self.device)
        context = torch.autocast("cuda", dtype=torch.float16) if self.amp else nullcontext()
        with context:
            output = self.model(image)
        # CPU transfer synchronizes CUDA, so measured wall time includes computation.
        output = {key: value[0].float().cpu().numpy() for key, value in output.items()}
        if not all(np.isfinite(value).all() for value in output.values()):
            raise RuntimeError("Non-finite output. If --amp is enabled, retry without it.")
        return output

    def warmup(self):
        for _ in range(3):
            self.predict(np.zeros((224, 224, 3), dtype=np.uint8))
