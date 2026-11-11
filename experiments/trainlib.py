"""
trainlib – PyTorch-Teil der Experimente: Daten (Camelyon17-WILDS), Modell, Training,
Evaluation und Zeitmessung. Die Farbfunktionen liegen in stainlib.py.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

import stainlib as sl

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
VARIANTS = ("baseline", "macenko", "hed", "macenko+hed")


# --------------------------------------------------------------------------------------
# Gerät und Daten
# --------------------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_camelyon(root: str = "data", download: bool = True):
    """Lädt Camelyon17-WILDS (ca. 450'000 Patches 96x96 aus 5 Spitälern, 50 Slides).

    Rückgabe: (WILDS-Datensatz, meta) mit meta = {'hospital', 'slide', 'y'} als NumPy-Arrays.
    """
    from wilds import get_dataset

    ds = get_dataset(dataset="camelyon17", root_dir=root, download=download)
    md = ds.metadata_array.numpy()
    fields = list(ds.metadata_fields)
    meta = {
        "hospital": md[:, fields.index("hospital")].astype(int),
        "slide": md[:, fields.index("slide")].astype(int),
        "y": np.asarray(ds.y_array).astype(int),
    }
    return ds, meta


def load_image(ds, i: int) -> np.ndarray:
    return np.asarray(ds.get_input(int(i)).convert("RGB"))


def _balanced(idx: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Bis zu n Indizes, je zur Hälfte Tumor und Normalgewebe."""
    parts = []
    for label in (0, 1):
        pool = idx[y[idx] == label]
        parts.append(rng.choice(pool, size=min(n // 2, len(pool)), replace=False))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def make_split(meta: dict, train_hosp, n_train_per_hosp: int = 10_000, n_val_slides_per_hosp: int = 2,
               n_eval_per_hosp: int = 4_000, seed: int = 0) -> dict:
    """Split nach Spital und Slide.

    train   : Patches aus den Trainingsspitälern (ohne die Validierungs-Slides)
    id_val  : andere Slides derselben Spitäler -> Leistung ohne Domain Shift
    ood[h]  : jedes nicht trainierte Spital einzeln -> Leistung mit Domain Shift
    """
    rng = np.random.default_rng(seed)
    hosp, slide, y = meta["hospital"], meta["slide"], meta["y"]
    split = {"train_hosp": [int(h) for h in train_hosp], "train": [], "id_val": [], "ood": {}}
    for h in sorted(np.unique(hosp)):
        idx_h = np.where(hosp == h)[0]
        if h in train_hosp:
            slides = np.unique(slide[idx_h])
            val_slides = rng.choice(slides, size=n_val_slides_per_hosp, replace=False)
            is_val = np.isin(slide[idx_h], val_slides)
            split["train"] += _balanced(idx_h[~is_val], y, n_train_per_hosp, rng).tolist()
            split["id_val"] += _balanced(idx_h[is_val], y, n_eval_per_hosp, rng).tolist()
        else:
            split["ood"][int(h)] = _balanced(idx_h, y, n_eval_per_hosp, rng).tolist()
    split["train"] = [int(i) for i in split["train"]]
    split["id_val"] = [int(i) for i in split["id_val"]]
    split["ood"] = {h: [int(i) for i in v] for h, v in split["ood"].items()}
    return split


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))


def load_json(path):
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, dict) and "ood" in obj:
        obj["ood"] = {int(k): v for k, v in obj["ood"].items()}
    return obj


# --------------------------------------------------------------------------------------
# Macenko-Parameter pro Slide
# --------------------------------------------------------------------------------------

def fit_reference(ds, indices, k: int = 256, seed: int = 0, dark_thresh=None) -> sl.StainParams:
    """Referenz = Macenko-Schätzung auf einem Mosaik zufälliger Trainingspatches."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(np.asarray(indices), size=min(k, len(indices)), replace=False)
    return sl.macenko_fit(sl.mosaic([load_image(ds, i) for i in sel]), dark_thresh=dark_thresh)


def fit_slide_params(ds, meta: dict, slides=None, k: int = 32, seed: int = 0, perturb=None,
                     verbose: bool = True, dark_thresh=None) -> dict:
    """Schätzt Stain-Matrix und Intensität einmal pro Slide aus k zufälligen Patches.

    Nutzt keine Labels – ist daher auch auf Testdaten zulässig.
    perturb: optionale Testzeit-Verschiebung (für Notebook 2), wird vor dem Fit angewandt.
    Slides, bei denen die Schätzung scheitert, fehlen im Ergebnis (-> Patches bleiben unverändert).
    """
    rng = np.random.default_rng(seed)
    slides = np.unique(meta["slide"]) if slides is None else np.asarray(slides)
    params, failed = {}, []
    for s in slides:
        idx = np.where(meta["slide"] == s)[0]
        sel = rng.choice(idx, size=min(k, len(idx)), replace=False)
        imgs = []
        for i in sel:
            img = load_image(ds, i)
            if perturb is not None:
                img = perturb(img, int(i), int(s))
            imgs.append(img)
        try:
            params[int(s)] = sl.macenko_fit(sl.mosaic(imgs), dark_thresh=dark_thresh)
        except (ValueError, np.linalg.LinAlgError):
            failed.append(int(s))
    if verbose:
        print(f"Slide-Parameter: {len(params)} geschätzt, {len(failed)} fehlgeschlagen {failed or ''}")
    return params


def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

class PatchDataset(Dataset):
    """Liefert (Tensor, Label, Spital). Reihenfolge der Bildschritte:

    1. Testzeit-Verschiebung (perturb, nur Notebook 2)
    2. Macenko mit den Parametern des Slides (mode enthält 'macenko')
    3. nur Training: HED-Jitter (mode enthält 'hed'), Drehung/Spiegelung (geo_aug=True)
    """

    def __init__(self, wds, indices, meta: dict, mode: str = "baseline", slide_params: dict | None = None,
                 ref: sl.StainParams | None = None, train: bool = False, hed_sigma: float = 0.1, perturb=None,
                 geo_aug: bool = True):
        assert mode in VARIANTS, f"mode muss eines von {VARIANTS} sein"
        if "macenko" in mode:
            assert slide_params is not None and ref is not None, "macenko braucht slide_params und ref"
        self.wds, self.indices = wds, np.asarray(indices, dtype=np.int64)
        self.y, self.hosp, self.slide = meta["y"], meta["hospital"], meta["slide"]
        self.mode, self.slide_params, self.ref = mode, slide_params or {}, ref
        self.train, self.hed_sigma, self.perturb, self.geo_aug = train, hed_sigma, perturb, geo_aug
        self._rng = None

    def __len__(self):
        return len(self.indices)

    def _rng_get(self) -> np.random.Generator:
        if self._rng is None:  # pro Worker eigener Zufallsstrom
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        return self._rng

    def image(self, k: int) -> np.ndarray:
        """Vorverarbeitetes Bild als uint8-Array (für Abbildungen)."""
        i = int(self.indices[k])
        s = int(self.slide[i])
        img = load_image(self.wds, i)
        if self.perturb is not None:
            img = self.perturb(img, i, s)
        if "macenko" in self.mode:
            p = self.slide_params.get(s)
            if p is not None:
                img = sl.macenko_apply(img, p, self.ref)
        if self.train:
            rng = self._rng_get()
            if "hed" in self.mode:
                img = sl.hed_jitter(img, self.hed_sigma, rng)
            if self.geo_aug:
                img = np.rot90(img, k=int(rng.integers(4)))
                if rng.random() < 0.5:
                    img = img[:, ::-1]
        return np.ascontiguousarray(img)

    def __getitem__(self, k: int):
        i = int(self.indices[k])
        x = torch.from_numpy(self.image(k)).permute(2, 0, 1).float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, int(self.y[i]), int(self.hosp[i])


def benchmark_transform(ds: PatchDataset, n: int = 500) -> float:
    """Millisekunden pro Patch für Laden + Vorverarbeitung (ein CPU-Kern)."""
    n = min(n, len(ds))
    t0 = time.perf_counter()
    for k in range(n):
        ds.image(k)
    return (time.perf_counter() - t0) / n * 1000


# --------------------------------------------------------------------------------------
# Modell, Training, Evaluation
# --------------------------------------------------------------------------------------

def build_model(arch: str = "resnet18", pretrained: bool = True) -> nn.Module:
    import torchvision

    if arch == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    elif arch == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    else:
        raise ValueError(arch)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def _loader(ds, batch_size, shuffle, workers, device, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == "cuda", drop_last=drop_last,
                      persistent_workers=workers > 0)


def train_model(model: nn.Module, train_ds: Dataset, epochs: int = 3, batch_size: int = 128,
                lr: float = 3e-4, workers: int = 4, device: torch.device | None = None,
                log=print) -> dict:
    """Trainiert und misst die Zeit. Rückgabe: Zeitmessungen (Sekunden, Bilder/s)."""
    device = device or get_device()
    model.to(device)
    loader = _loader(train_ds, batch_size, True, workers, device, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    t = {"epoch_s": [], "data_wait_s": 0.0, "images": 0}
    for ep in range(epochs):
        model.train()
        t0 = tw = time.perf_counter()
        loss_sum, nb = 0.0, 0
        for x, y, _ in loader:
            t["data_wait_s"] += time.perf_counter() - tw
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum, nb = loss_sum + loss.item(), nb + 1
            t["images"] += x.shape[0]
            tw = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t["epoch_s"].append(time.perf_counter() - t0)
        log(f"  Epoche {ep + 1}/{epochs}: Loss {loss_sum / max(nb, 1):.4f}, {t['epoch_s'][-1]:.0f} s")
    t["train_s"] = float(sum(t["epoch_s"]))
    t["images_per_s"] = t["images"] / max(t["train_s"], 1e-9)
    return t


@torch.no_grad()
def predict(model: nn.Module, ds: Dataset, batch_size: int = 256, workers: int = 4,
            device: torch.device | None = None) -> dict:
    device = device or get_device()
    model.to(device).eval()
    probs, ys, hs = [], [], []
    t0 = time.perf_counter()
    for x, y, h in _loader(ds, batch_size, False, workers, device):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
        hs.append(h.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    out = {"prob": np.concatenate(probs), "y": np.concatenate(ys), "hosp": np.concatenate(hs)}
    out["seconds"], out["images_per_s"] = sec, len(out["y"]) / max(sec, 1e-9)
    return out


def metrics(pred: dict) -> dict:
    y, p = pred["y"], pred["prob"]
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return {"acc": float(((p >= 0.5) == y).mean()), "auc": float(auc), "n": int(len(y))}


@torch.no_grad()
def extract_features(model: nn.Module, ds: Dataset, batch_size: int = 256, workers: int = 4,
                     device: torch.device | None = None) -> np.ndarray:
    """Merkmale der vorletzten Schicht (ResNet: globales Pooling vor fc)."""
    device = device or get_device()
    backbone = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    feats = []
    for x, _, _ in _loader(ds, batch_size, False, workers, device):
        feats.append(backbone(x.to(device)).flatten(1).float().cpu().numpy())
    return np.concatenate(feats)


# --------------------------------------------------------------------------------------
# Ein komplettes Experiment
# --------------------------------------------------------------------------------------

def run_suffix(cfg: dict) -> str:
    """Namenszusatz für Dateien: leer bei der Standardbedingung (vortrainiert + Drehung/Spiegelung,
    3 Epochen, hed_sigma 0.1, Original-Macenko), sonst z. B. '_scratch_nogeo' oder
    '_scratch_nogeo_ep10_hed0.2_dark'. So bleiben die Standard-Dateinamen für Notebook 2 und 3 gleich
    und frühere Ergebnisse werden nie überschrieben."""
    pretrained = cfg.get("pretrained", True)
    parts = []
    if not pretrained:
        parts.append("scratch")
    if not cfg.get("geo_aug", True):
        parts.append("nogeo")
    default_epochs = 3 if pretrained else 5
    if cfg.get("epochs", default_epochs) != default_epochs:
        parts.append(f"ep{cfg['epochs']}")
    if cfg.get("hed_sigma", 0.1) != 0.1:
        parts.append(f"hed{cfg['hed_sigma']}")
    if cfg.get("macenko_dark") is not None:
        parts.append("dark" if cfg["macenko_dark"] == 30 else f"dark{cfg['macenko_dark']}")
    return "".join("_" + p for p in parts)


def run_experiment(wds, meta, split, variants, seeds, slide_params, ref, cfg: dict,
                   model_dir: str | None = "models", log=print) -> tuple[list, dict]:
    """Trainiert jede Variante mit jedem Seed und wertet ID- und OOD-Leistung aus.

    cfg: epochs, batch_size, lr, workers, hed_sigma, arch,
         pretrained (Standard True: ImageNet-Gewichte), geo_aug (Standard True: Drehung/Spiegelung)
    Rückgabe: (Ergebniszeilen, Zeitmessungen je Variante/Seed)
    """
    pretrained, geo_aug = cfg.get("pretrained", True), cfg.get("geo_aug", True)
    cond = run_suffix(cfg)
    device = get_device()
    rows, timings = [], {}
    for seed in seeds:
        for v in variants:
            log(f"\n=== {v} | Seed {seed} | Trainingsspitäler {split['train_hosp']} | "
                f"pretrained={pretrained} geo_aug={geo_aug} ===")
            torch.manual_seed(seed)
            np.random.seed(seed)
            common = dict(meta=meta, mode=v, slide_params=slide_params, ref=ref)
            train_ds = PatchDataset(wds, split["train"], train=True, hed_sigma=cfg["hed_sigma"],
                                    geo_aug=geo_aug, **common)
            model = build_model(cfg.get("arch", "resnet18"), pretrained=pretrained)
            t = train_model(model, train_ds, cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["workers"],
                            device, log)
            evals = {"id_val": split["id_val"], **{f"ood_h{h}": idx for h, idx in split["ood"].items()}}
            infer_s, infer_n = 0.0, 0
            for name, idx in evals.items():
                pred = predict(model, PatchDataset(wds, idx, **common), workers=cfg["workers"], device=device)
                m = metrics(pred)
                infer_s, infer_n = infer_s + pred["seconds"], infer_n + m["n"]
                rows.append({"variant": v, "seed": seed, "train_hosp": str(split["train_hosp"]),
                             "pretrained": pretrained, "geo_aug": geo_aug, "epochs": cfg["epochs"],
                             "eval": name, **m})
                log(f"  {name:8s} Acc {m['acc']:.3f}  AUC {m['auc']:.3f}  (n={m['n']})")
            t["infer_images_per_s"] = infer_n / max(infer_s, 1e-9)
            timings[f"{v}|{seed}|{split['train_hosp']}{cond}"] = t
            if model_dir is not None:
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                tag = "-".join(map(str, split["train_hosp"]))
                torch.save(model.state_dict(), Path(model_dir) / f"{v}_seed{seed}_train{tag}{cond}.pt")
    return rows, timings
