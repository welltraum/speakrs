# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy", "pyannote.audio==4.0.7"]
# ///
"""Generate pyannote VBxClustering hard clusters for speaker count constraints.

Input: the `test.wav` pipeline fixtures and the PLDA arrays in `fixtures/models`.
Output: `fixtures/pipeline_hard_clusters_<case>.npy`, compared in `src/pipeline/tests.rs`.

uv run scripts/generate_speaker_count_fixtures.py
"""

import tempfile
from pathlib import Path

import numpy as np
from pyannote.audio.core.plda import PLDA
from pyannote.audio.pipelines.clustering import VBxClustering
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.audio.pipelines.utils.diarization import set_num_speakers
from pyannote.core import SlidingWindow, SlidingWindowFeature

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

# case name -> (num_speakers, min_speakers, max_speakers); VBx finds 2 speakers
CASES = {
    "exact_3": (3, None, None),
    "exact_1": (1, None, None),
    "min_4": (None, 4, None),
    "max_1": (None, None, 1),
}


def load_plda(models: Path, tmp: Path) -> PLDA:
    arrays = {name: np.load(models / f"plda_{name}.npy") for name in ("mean1", "mean2", "lda", "mu", "tr", "psi")}
    np.savez(tmp / "xvec_transform.npz", mean1=arrays["mean1"], mean2=arrays["mean2"], lda=arrays["lda"])
    np.savez(tmp / "plda.npz", mu=arrays["mu"], tr=arrays["tr"], psi=arrays["psi"])
    return PLDA(tmp / "xvec_transform.npz", tmp / "plda.npz")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plda = load_plda(FIXTURES / "models", Path(tmp))
    clustering = VBxClustering(plda)
    clustering.instantiate(SpeakerDiarization.default_parameters(None)["clustering"])

    segmentations = SlidingWindowFeature(
        np.load(FIXTURES / "pipeline_segmentation_data.npy"),
        SlidingWindow(start=0.0, duration=10.0, step=1.0),
    )
    embeddings = np.load(FIXTURES / "pipeline_embeddings_data.npy")
    inactive = np.sum(segmentations.data, axis=1) == 0

    for name, (num, low, high) in CASES.items():
        num_speakers, min_speakers, max_speakers = set_num_speakers(num, low, high)
        hard_clusters, _, _ = clustering(
            embeddings=embeddings.copy(),
            segmentations=segmentations,
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
        )
        hard_clusters[inactive] = -2
        np.save(FIXTURES / f"pipeline_hard_clusters_{name}.npy", hard_clusters.astype(np.int8))
        print(name, np.unique(hard_clusters[hard_clusters >= 0]))


if __name__ == "__main__":
    main()
