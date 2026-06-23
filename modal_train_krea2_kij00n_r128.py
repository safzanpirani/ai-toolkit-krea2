r"""
Krea2 (K2) rank-128 character LoRA training for kij00n on Modal H100.

Differs from the Ideogram4 envy drivers in three ways:
  1. K2 weights load from a LOCAL file, not an HF repo download -> mounted from
     the `krea2-weights` Modal volume at /root/krea2-weights/raw.safetensors
     (upload once:  modal volume put krea2-weights <local raw.safetensors> /raw.safetensors).
  2. Captions are PLAIN natural-language text (not Ideogram JSON) -> plain-text preflight.
  3. arch: krea2 (Krea2Model), via the safzanpirani/ai-toolkit-krea2 fork.

The TE (Qwen3-VL-4B) and VAE (Qwen-Image) download from HF (public) and cache on
the hf-hub-cache volume.

Run (PowerShell, UTF-8 forced) from a checkout of safzanpirani/comfyui-modal:
    $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8
    modal run training\modal-scripts\modal_ai_toolkit_krea2_kij00n_r128.py

Override AI_TOOLKIT_LOCAL_PATH / DATASET_LOCAL_PATH via env to launch from a
different machine (e.g. the Mac, where the fork clone + NL dataset live).
"""
import os
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
import sys
import threading
import modal
from dotenv import load_dotenv
load_dotenv()
os.environ["DISABLE_TELEMETRY"] = "YES"

# Checkout of safzanpirani/ai-toolkit-krea2 (the fork carrying the krea2 adapter
# + config). On the box: git clone https://github.com/safzanpirani/ai-toolkit-krea2 F:\ai-toolkit-krea2
AI_TOOLKIT_LOCAL_PATH = os.environ.get("AI_TOOLKIT_LOCAL_PATH", "F:\\ai-toolkit-krea2")

CHAR = "kij00n"
# Natural-language recaptioned kij00n dataset (push from kij00n_crop_nl before running).
DATASET_LOCAL_PATH = os.environ.get("DATASET_LOCAL_PATH", "D:\\lora\\kij00n_krea2_nl")
_DATASET_NAME = "kij00n_krea2_r128"
DATASET_REMOTE_PATH = f"/root/ai-toolkit/datasets/{_DATASET_NAME}"
DEFAULT_CONFIG = f"/root/ai-toolkit/config/krea2_{CHAR}_r128_lora.yaml"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def preflight_validate_dataset(path: str, trigger: str) -> None:
    """Fail fast (before image build / GPU spend) on bad datasets.

    Krea2 uses plain natural-language .txt captions (NOT Ideogram JSON), each
    starting with the trigger word.
    """
    images, captions = set(), {}
    for f in os.listdir(path):
        base, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXTS:
            images.add(base)
        elif ext.lower() == ".txt":
            captions[base] = os.path.join(path, f)
    uncaptioned = images - set(captions)
    orphans = set(captions) - images
    if uncaptioned:
        raise SystemExit(f"PREFLIGHT: {len(uncaptioned)} images without captions: {sorted(uncaptioned)[:5]}")
    if orphans:
        raise SystemExit(f"PREFLIGHT: {len(orphans)} captions without images: {sorted(orphans)[:5]}")
    empty, bad_trigger = [], []
    for base, cap_path in captions.items():
        text = open(cap_path, encoding="utf-8").read().strip()
        if not text:
            empty.append(base)
            continue
        if not text.startswith(trigger):
            bad_trigger.append(base)
    if empty:
        raise SystemExit(f"PREFLIGHT: {len(empty)} empty captions: {sorted(empty)[:5]}")
    if bad_trigger:
        raise SystemExit(f"PREFLIGHT: {len(bad_trigger)} captions missing trigger '{trigger}' prefix: {sorted(bad_trigger)[:5]}")
    print(f"PREFLIGHT OK: {len(images)} image/caption pairs, all plain-text, all start with '{trigger}'")


if modal.is_local() and "--help" not in sys.argv:
    preflight_validate_dataset(DATASET_LOCAL_PATH, CHAR)

model_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
# 26.6 GB bf16 raw.safetensors lives here (upload once via `modal volume put`).
krea2_weights_volume = modal.Volume.from_name("krea2-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
MOUNT_DIR = "/root/ai-toolkit/modal_output"
WEIGHTS_DIR = "/root/krea2-weights"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .uv_pip_install(
        "hf_transfer",
        "torch",
        "torchvision",
        "torchaudio",
        "ftfy",
        "torchao==0.10.0",
        "safetensors",
        "git+https://github.com/huggingface/diffusers.git@dc8d9032171c83741fd37ed2b12bc9d8274464f3",
        "transformers==5.5.3",
        "lycoris-lora==1.8.3",
        "flatten_json",
        "pyyaml",
        "oyaml",
        "tensorboard",
        "kornia",
        "invisible-watermark",
        "einops",
        "accelerate",
        "toml",
        "albumentations==1.4.15",
        "albucore==0.0.16",
        "pydantic",
        "omegaconf",
        "k-diffusion",
        "open_clip_torch",
        "timm==1.0.22",
        "prodigyopt",
        "controlnet_aux==0.0.10",
        "python-dotenv",
        "bitsandbytes",
        "hf_transfer",
        "lpips",
        "pytorch_fid",
        "optimum-quanto==0.2.4",
        "sentencepiece",
        "huggingface_hub==1.10.1",
        "peft==0.18.1",
        "gradio",
        "python-slugify",
        "opencv-python",
        "pytorch-wavelets==1.3.0",
        "matplotlib==3.10.1",
        "setuptools==69.5.1",
        "av==16.0.1",
        "torchcodec==0.9.1",
        "librosa==0.11.0",
        "mutagen==1.47.0",
        "scipy==1.12.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(AI_TOOLKIT_LOCAL_PATH, remote_path="/root/ai-toolkit")
    .add_local_dir(DATASET_LOCAL_PATH, remote_path=DATASET_REMOTE_PATH)
)

app = modal.App(
    name=f"krea2-lora-{CHAR}-r128",
    image=image,
    volumes={
        MOUNT_DIR: model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
        WEIGHTS_DIR: krea2_weights_volume,
    },
)


def print_end_message(jobs_completed: int, jobs_failed: int) -> None:
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    print()
    print("=" * 40)
    print("Result:")
    print(f" - {completed_string}")
    if failure_string:
        print(f" - {failure_string}")
    print("=" * 40)


@app.function(
    gpu=os.environ.get("MODAL_GPU", "H100"),  # set MODAL_GPU=B200 to switch
    timeout=86400,
    secrets=[hf_secret],
)
def train(
    config_file_list_str: str,
    recover: bool = False,
    name: str = None,
):
    config_file_list = [c.strip() for c in config_file_list_str.split(",") if c.strip()]
    print(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")
    # Sanity: weights present on the mounted volume before any GPU spend.
    weights_path = os.path.join(WEIGHTS_DIR, "raw.safetensors")
    if not os.path.exists(weights_path):
        raise SystemExit(
            f"krea2 weights not found at {weights_path}. Upload once with:\n"
            f"  modal volume put krea2-weights <local raw.safetensors> /raw.safetensors"
        )
    sys.path.insert(0, "/root/ai-toolkit")
    from toolkit.job import get_job
    jobs_completed = 0
    jobs_failed = 0
    stop_committer = threading.Event()
    commit_interval_s = 120

    def _periodic_commit():
        while not stop_committer.wait(commit_interval_s):
            try:
                model_volume.commit()
            except Exception as e:
                print(f"periodic volume commit failed (will retry): {e}")

    committer_thread = threading.Thread(target=_periodic_commit, daemon=True)
    committer_thread.start()
    try:
        for config_file in config_file_list:
            job = None
            try:
                os.makedirs(MOUNT_DIR, exist_ok=True)
                job = get_job(config_file, name)
                process_training_folder = job.config.get("process", [{}])[0].get("training_folder")
                if process_training_folder != MOUNT_DIR:
                    raise RuntimeError(
                        f"YAML training_folder must be set to {MOUNT_DIR!r} "
                        f"(the persistent volume mount), got {process_training_folder!r}."
                    )
                print(f"Training outputs will be saved to: {MOUNT_DIR}")
                job.run()
                model_volume.commit()
                jobs_completed += 1
            except Exception as e:
                print(f"Error running job: {e}")
                jobs_failed += 1
                if not recover:
                    print_end_message(jobs_completed, jobs_failed)
                    raise
            finally:
                if job is not None:
                    try:
                        job.cleanup()
                    except Exception as cleanup_error:
                        print(f"Warning: job cleanup failed: {cleanup_error}")
    finally:
        stop_committer.set()
        committer_thread.join(timeout=5)
        try:
            model_volume.commit()
        except Exception as e:
            print(f"final volume commit failed: {e}")
    print_end_message(jobs_completed, jobs_failed)


@app.local_entrypoint()
def cli(
    config_file_list_str: str = DEFAULT_CONFIG,
    recover: bool = False,
    name: str = None,
):
    train.remote(
        config_file_list_str=config_file_list_str,
        recover=recover,
        name=name,
    )
