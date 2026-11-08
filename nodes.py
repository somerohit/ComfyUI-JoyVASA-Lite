"""
ComfyUI-JoyVASA-Lite
Minimal ComfyUI wrapper around the official jdh-algo/JoyVASA inference script.
JoyVASA runs in its own isolated venv (torch 2.2.2) so it never touches ComfyUI's torch.
"""
import os
import sys
import shutil
import subprocess
import tempfile
import time
import uuid

import numpy as np
import torch

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
JOY_DIR = os.path.join(NODE_DIR, "JoyVASA")
VENV_DIR = os.path.join(NODE_DIR, "venv")
VENV_PY = os.path.join(VENV_DIR, "bin", "python")
WEIGHTS = os.path.join(JOY_DIR, "pretrained_weights")
REQS = os.path.join(NODE_DIR, "requirements-joyvasa.txt")
JOY_REPO = "https://github.com/jdh-algo/JoyVASA.git"
LOG_FILE = os.path.join(NODE_DIR, "setup.log")


def _run(cmd, cwd=None, log=None, env=None, timeout=None):
    """Run a command, stream output to log list + setup.log, raise on failure."""
    line = f"\n$ {' '.join(cmd)}  (cwd={cwd or os.getcwd()})\n"
    if log is not None:
        log.append(line)
    with open(LOG_FILE, "a") as lf:
        lf.write(line)
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    with open(LOG_FILE, "a") as lf:
        lf.write(out)
    if log is not None:
        log.append(out[-4000:])
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out[-6000:]}")
    return out


def _uv_cmd():
    uv = shutil.which("uv")
    if uv:
        return [uv]
    uv_local = os.path.join(NODE_DIR, "uv")
    if os.path.exists(uv_local):
        return [uv_local]
    # try as python module
    try:
        subprocess.run([sys.executable, "-m", "uv", "--version"], check=True, capture_output=True)
        return [sys.executable, "-m", "uv"]
    except Exception:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    return [sys.executable, "-m", "uv"]


def ensure_setup(log=None, force=False):
    log = log if log is not None else []
    # 1. clone JoyVASA
    if not os.path.exists(os.path.join(JOY_DIR, "inference.py")):
        _run(["git", "clone", "--depth", "1", JOY_REPO, JOY_DIR], log=log)
    # 2. venv via uv
    uv = _uv_cmd()
    if not os.path.exists(VENV_PY):
        _run(uv + ["venv", "--python", "3.10", VENV_DIR], log=log)
    marker = os.path.join(VENV_DIR, ".joyvasa_deps_ok")
    if force or not os.path.exists(marker):
        _run(uv + ["pip", "install", "--python", VENV_PY, "-r", REQS], log=log, timeout=3600)
        open(marker, "w").write("ok")
    # 3. weights
    wmarker = os.path.join(WEIGHTS, ".joyvasa_weights_ok")
    if force or not os.path.exists(wmarker):
        os.makedirs(WEIGHTS, exist_ok=True)
        script = f"""
from huggingface_hub import snapshot_download
snapshot_download("KwaiVGI/LivePortrait", local_dir={WEIGHTS!r},
                  allow_patterns=["insightface/*", "insightface/**", "liveportrait/*", "liveportrait/**"])
snapshot_download("jdh-algo/JoyVASA", local_dir={os.path.join(WEIGHTS, 'JoyVASA')!r})
snapshot_download("TencentGameMate/chinese-hubert-base", local_dir={os.path.join(WEIGHTS, 'chinese-hubert-base')!r})
print("weights ok")
"""
        _run([VENV_PY, "-c", script], log=log, timeout=3600)
        open(wmarker, "w").write("ok")
    log.append("\nJoyVASA setup OK\n")
    return "".join(log)


def _save_audio_wav(audio, path):
    """ComfyUI AUDIO dict -> 16 kHz mono wav"""
    wav = audio["waveform"]  # (B, C, T)
    sr = int(audio["sample_rate"])
    w = wav[0]
    if w.shape[0] > 1:
        w = w.mean(dim=0, keepdim=True)
    w = w.squeeze(0).float().cpu().numpy()
    try:
        import torchaudio
        torchaudio.save(path, torch.from_numpy(w).unsqueeze(0), sr)
        return
    except Exception:
        pass
    import wave
    pcm = (np.clip(w, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def _load_video_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    arr = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


class JoyVASA_Setup:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"force_reinstall": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "run"
    CATEGORY = "JoyVASA"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")

    def run(self, force_reinstall):
        log = []
        try:
            out = ensure_setup(log, force=force_reinstall)
        except Exception as e:
            out = "".join(log) + f"\nERROR: {e}"
        return {"ui": {"text": [out[-8000:]]}, "result": (out,)}


class JoyVASA_Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "animation_mode": (["human", "animal"], {"default": "human"}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "flag_do_crop": ("BOOLEAN", {"default": True}),
                "flag_pasteback": ("BOOLEAN", {"default": True}),
                "flag_stitching": ("BOOLEAN", {"default": True}),
                "source_max_dim": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 64}),
                "filename_prefix": ("STRING", {"default": "JoyVASA"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "video_path", "log")
    FUNCTION = "run"
    CATEGORY = "JoyVASA"
    OUTPUT_NODE = True

    def run(self, image, audio, animation_mode, cfg_scale, flag_do_crop, flag_pasteback,
            flag_stitching, source_max_dim, filename_prefix):
        log = []
        ensure_setup(log)

        work = tempfile.mkdtemp(prefix="joyvasa_")
        uid = uuid.uuid4().hex[:8]
        img_path = os.path.join(work, f"ref_{uid}.png")
        wav_path = os.path.join(work, f"audio_{uid}.wav")
        out_dir = os.path.join(work, "out")
        os.makedirs(out_dir, exist_ok=True)

        from PIL import Image
        img = (image[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(img_path)
        _save_audio_wav(audio, wav_path)

        def bflag(name, val):
            return [f"--{name}" if val else f"--no-{name}"]

        cmd = [VENV_PY, "inference.py", "-r", img_path, "-a", wav_path,
               "--animation_mode", animation_mode, "--cfg_scale", str(cfg_scale),
               "-o", out_dir, "--source_max_dim", str(source_max_dim)]
        cmd += bflag("flag_do_crop", flag_do_crop)
        cmd += bflag("flag_pasteback", flag_pasteback)
        cmd += bflag("flag_stitching", flag_stitching)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONUNBUFFERED"] = "1"
        t0 = time.time()
        _run(cmd, cwd=JOY_DIR, log=log, env=env, timeout=7200)
        log.append(f"\nJoyVASA inference took {time.time()-t0:.1f}s\n")

        vids = [f for f in os.listdir(out_dir) if f.endswith(".mp4") and not f.endswith("_temp.mp4")]
        if not vids:
            vids = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
        if not vids:
            raise RuntimeError("JoyVASA produced no mp4.\n" + "".join(log)[-6000:])
        src = os.path.join(out_dir, vids[0])

        try:
            import folder_paths
            out_root = folder_paths.get_output_directory()
        except Exception:
            out_root = os.path.join(NODE_DIR, "output")
        os.makedirs(out_root, exist_ok=True)
        final = os.path.join(out_root, f"{filename_prefix}_{uid}.mp4")
        shutil.copy(src, final)

        frames = _load_video_frames(final)
        logtxt = "".join(log)
        return {
            "ui": {"text": [f"saved: {final}\nframes: {frames.shape[0]}"],
                   "videos": [{"filename": os.path.basename(final), "subfolder": "", "type": "output"}]},
            "result": (frames, audio, final, logtxt),
        }


NODE_CLASS_MAPPINGS = {
    "JoyVASA_Setup": JoyVASA_Setup,
    "JoyVASA_Generate": JoyVASA_Generate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyVASA_Setup": "JoyVASA Setup (install venv + weights)",
    "JoyVASA_Generate": "JoyVASA Generate (image + audio -> video)",
}
