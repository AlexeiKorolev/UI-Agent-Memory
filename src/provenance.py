"""Write results/provenance.json: exact revisions of models, data, reference code and software."""
import json
import platform
import subprocess

from src.data import PROJ, SPLIT
from src.remote_zip import REPO as DATA_REPO, REV as DATA_REV
from src.vlm import MODELS


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"ERR {e}"


def main():
    import torch, transformers, vllm, PIL, pandas, rapidfuzz, statsmodels
    prov = {
        "models": {role: {"repo": r, "revision": rev} for role, (r, rev) in MODELS.items()},
        "dataset": {"repo": DATA_REPO, "revision": DATA_REV, "split": f"{SPLIT} / test",
                    "superseded_repo": {"repo": "OpenGVLab/GUI-Odyssey", "revision": "71e0e7e2d169c642e7a99c264d82ffecadf67889",
                                        "note": "v1; README points to the v2 repo above, which we use"}},
        "reference_code": {
            "OpenGVLab/GUI-Odyssey": sh(f"git -C {PROJ}/third_party/GUI-Odyssey rev-parse HEAD"),
            "inclusionAI/UI-Venus": sh(f"git -C {PROJ}/third_party/UI-Venus rev-parse HEAD"),
            "ui_venus_prompt_source": "UI-Venus-1.5 Technical Report, arXiv:2602.09082v2, Appendix A.3 'Mobile Prompt' (verbatim)",
        },
        "this_repo_commit": sh(f"git -C {PROJ} rev-parse HEAD"),
        "software": {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                     "transformers": transformers.__version__, "vllm": vllm.__version__, "pillow": PIL.__version__,
                     "pandas": pandas.__version__, "rapidfuzz": rapidfuzz.__version__, "statsmodels": statsmodels.__version__,
                     "tesseract": sh("tesseract --version 2>&1 | head -1")},
        "decoding": {"temperature": 0.0, "seed": 0, "backbone_max_tokens": 768, "controller_max_tokens": 256},
        "sampling": {"seed": 0, "method": "stratified by task category (src/data.py::stratified_order)"},
    }
    json.dump(prov, open(PROJ / "results" / "provenance.json", "w"), indent=2)
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
