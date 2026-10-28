"""Inference-only wrapper preserving all official learnable parameter names.

MeshHead and position encoding are copied unchanged from upstream, see NOTICE.md.
MANO assets remain in a separate, user-downloaded official repository.
"""
from pathlib import Path
import pickle
import numpy as np
import torch
from torch import nn
import timm
from .upstream.modules import MeshHead, AttentionBlock


class SimpleHand(nn.Module):
    def __init__(self, mano_right: Path):
        super().__init__()
        self.backbone = timm.create_model("fastvit_ma36", pretrained=False,
                                          drop_path_rate=0.2)
        self.avg_pool = nn.AvgPool2d((7, 7), 1)
        self.keypoints_2d_head = nn.Linear(1216, 42)
        self.mesh_head = MeshHead(
            in_channels=1216, depths=[1, 1, 1], token_nums=[21, 84, 336],
            dims=[256, 128, 64], dropout=0.1,
            first_prenorms=[True, True, True],
            block_types=[AttentionBlock] * 3,
        )
        # Use only the official trusted asset. Pickle can execute code.
        with Path(mano_right).open("rb") as stream:
            data = pickle.load(stream, encoding="latin1")
        regressor = data["J_regressor"]
        if hasattr(regressor, "toarray"):
            regressor = regressor.toarray()
        regressor = np.asarray(regressor, dtype=np.float32)
        if regressor.shape != (16, 778):
            raise ValueError(f"Expected MANO regressor (16,778), got {regressor.shape}")
        self.register_buffer("joint_regressor", torch.from_numpy(regressor), persistent=False)
        faces = np.asarray(data["f"], dtype=np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= 778:
            raise ValueError("Invalid MANO triangle topology")
        self.register_buffer("faces", torch.from_numpy(faces), persistent=False)
        self.register_buffer("tips", torch.tensor([744, 320, 443, 555, 672]), persistent=False)
        self.register_buffer("joint_order", torch.tensor(
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
             10, 11, 12, 19, 7, 8, 9, 20]), persistent=False)

    def forward(self, image):
        # Official dataset reads cv2 BGR; official HandNet.forward normalizes here.
        image = image.float() / 255.0 - 0.5
        features = self.backbone.forward_features(image)
        global_feature = self.avg_pool(features).flatten(1)
        uv = self.keypoints_2d_head(global_feature).reshape(-1, 21, 2)
        vertices = self.mesh_head(features, uv)
        # 16 anatomical joints plus 5 fingertip vertices, matching official code.
        joints16 = torch.einsum("jv,bvc->bjc", self.joint_regressor, vertices.float())
        joints = torch.cat((joints16, vertices[:, self.tips].float()), dim=1)
        joints = joints[:, self.joint_order]
        return {"uv": uv, "vertices": vertices, "joints": joints}
