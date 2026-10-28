# Attribution

This application builds on SimpleHand by Zhishan Zhou et al., CVPR 2024.

- Source: https://github.com/patienceFromZhou/simpleHand
- Pinned commit: `c3e6abc55512d712c4ae9b2695e529607f73b00f`
- `models/upstream/modules.py` and `position_embedding.py` are unmodified copies.
- Their original MIT license is retained in `models/upstream/LICENSE`.
- Position encoding retains its additional upstream attribution in the file header.

The application implements webcam ROI handling, an inference-only wrapper,
interactive mesh rendering, export, recording, runtime statistics and evaluation
tools. It uses pretrained weights from the original authors. It does not claim
the SimpleHand architecture or its published benchmark results as original work.

Checkpoint files, MANO assets and datasets are excluded. Obtain these resources
from their official sources and follow the applicable terms. The setup script
retrieves the official source at the pinned revision instead of bundling it.
