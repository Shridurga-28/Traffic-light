# LISA -> YOLO (handles semicolon CSVs + your headers)
import os, json, shutil, random
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ==== EDIT ONLY THESE ====
BASE = Path(r"E:\Abaja\traffic")          # your root (as in your screenshot)
OUT_ROOT = BASE / "dataset_yolo_tl"       # output folder
TRAIN, VAL = 0.8, 0.1                     # test = 0.1
random.seed(123)

# label mapping
RAW2CANON = {
    "go":"green",
    "stop":"red",
    "warning":"yellow", "amber":"yellow",
    "off":"off",
    "unknown":"unknown",
    # arrows (optional)
    "goleft":"green","goright":"green","goforward":"green",
    "stopleft":"red","stopright":"red",
    "warningleft":"yellow","warningright":"yellow",
}
CLASSES = ["red","yellow","green","off","unknown"]
CLS2ID = {c:i for i,c in enumerate(CLASSES)}

# exact column names from your sample
COL_FILENAME = "Filename"
COL_TAG      = "Annotation tag"
COL_X1       = "Upper left corner X"
COL_Y1       = "Upper left corner Y"
COL_X2       = "Lower right corner X"
COL_Y2       = "Lower right corner Y"

def read_csv_any(p: Path):
    # prefer semicolon; fall back to auto
    try:
        return pd.read_csv(p, sep=";", engine="python")
    except Exception:
        return pd.read_csv(p, sep=None, engine="python")

def safe_label(tag):
    t = "unknown" if pd.isna(tag) else str(tag).strip().lower()
    return RAW2CANON.get(t, "unknown")

def to_cxcywh(x1,y1,x2,y2,W,H):
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    x1, x2 = min(x1,x2), max(x1,x2)
    y1, y2 = min(y1,y2), max(y1,y2)
    w = max(0.0, x2-x1); h = max(0.0, y2-y1)
    cx = x1 + w/2.0; cy = y1 + h/2.0
    return cx/W, cy/H, w/W, h/H

def find_all_csv(base: Path):
    return list(base.rglob("frameAnnotationsBOX.csv"))

def infer_image_path(csv_path: Path, filename_value: str):
    """
    Resolve the real image file for a CSV row.
    Handles cases like 'dayTest/daySequence1--00001.jpg' by:
      - trying BASE/that_path
      - stripping 'dayTest/' or 'nightTest/' prefix
      - searching common sequence dirs' frames/ subfolders
      - final fallback: basename search under BASE
    """
    name_in_csv = str(filename_value).replace("\\", "/").strip()
    basename = Path(name_in_csv).name

    # 1) direct path under BASE (if CSV stored a relative path that truly exists)
    cand = (BASE / name_in_csv)
    if cand.exists():
        return cand

    # 2) strip 'dayTest/' or 'nightTest/' prefix and try under BASE directly
    norm = name_in_csv
    for pref in ("dayTest/", "nightTest/"):
        if norm.startswith(pref):
            norm = norm[len(pref):]
            break
    cand2 = BASE / norm
    if cand2.exists():
        return cand2

    # 3) common LISA layout: look in <root>/<sequence or clip>/frames/<basename>
    likely_roots = [
        BASE / "daySequence1",
        BASE / "daySequence2",
        BASE / "dayTrain",
        BASE / "nightSequence1",
        BASE / "nightSequence2",
        BASE / "nightTrain",
        BASE / "sample-dayClip6",
        BASE / "sample-nightClip1",
    ]
    for root in likely_roots:
        frames_dir = root / "frames"
        if frames_dir.is_dir():
            p = frames_dir / basename
            if p.exists():
                return p

        # sometimes nested as <root>/<same-name>/frames/…
        nested = root / root.name / "frames" / basename
        if nested.exists():
            return nested

    # 4) sibling 'frames' next to the CSV (works for many LISA drops)
    csv_frames = csv_path.parent / "frames" / basename
    if csv_frames.exists():
        return csv_frames

    # 5) last resort: search by basename anywhere under BASE
    for p in BASE.rglob(basename):
        return p

    return None

def main():
    csv_files = find_all_csv(BASE)
    if not csv_files:
        print("No frameAnnotationsBOX.csv under", BASE); return

    for sub in ["images/train","images/val","images/test","labels/train","labels/val","labels/test"]:
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    # collect rows
    by_img = {}
    for csvf in csv_files:
        try:
            df = read_csv_any(csvf)
        except Exception as e:
            print(f"[WARN] could not read {csvf}: {e}"); continue

        # check required columns
        miss = [c for c in [COL_FILENAME,COL_X1,COL_Y1,COL_X2,COL_Y2] if c not in df.columns]
        if miss:
            print(f"[WARN] {csvf} missing {miss}; skipping.")
            continue

        sdf = df.dropna(subset=[COL_FILENAME, COL_X1, COL_Y1, COL_X2, COL_Y2]).copy()
        for _, r in sdf.iterrows():
            fname = str(r[COL_FILENAME]).strip()
            if not fname: continue
            img_path = infer_image_path(csvf, fname)
            if img_path is None or not img_path.exists(): continue

            label = safe_label(r[COL_TAG]) if COL_TAG in df.columns else "unknown"
            item = {"img": img_path, "label": label,
                    "x1": r[COL_X1], "y1": r[COL_Y1], "x2": r[COL_X2], "y2": r[COL_Y2]}
            by_img.setdefault(img_path, []).append(item)

    if not by_img:
        print("No valid rows found. Check paths/columns."); return

    # deterministic split by image
    imgs = sorted(by_img.keys(), key=lambda p: str(p).lower())
    n = len(imgs); n_train = int(n*TRAIN); n_val = int(n*VAL)
    splits = {"train": set(imgs[:n_train]),
              "val":   set(imgs[n_train:n_train+n_val]),
              "test":  set(imgs[n_train+n_val:])}

    # convert
    for img_path in tqdm(imgs, desc="Writing YOLO"):
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            continue

        split = "train" if img_path in splits["train"] else ("val" if img_path in splits["val"] else "test")
        dst_img = OUT_ROOT / f"images/{split}" / img_path.name
        if not dst_img.exists():
            try: shutil.copy2(img_path, dst_img)
            except Exception: continue

        lines = []
        for it in by_img[img_path]:
            cls_id = CLS2ID[it["label"]]
            cx, cy, w, h = to_cxcywh(it["x1"], it["y1"], it["x2"], it["y2"], W, H)
            if w<=0 or h<=0: continue
            if (w*W)<2 or (h*H)<2: continue
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (OUT_ROOT / f"labels/{split}").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / f"labels/{split}" / (img_path.stem + ".txt")).write_text("\n".join(lines))

    data_yaml = {
        "path": str(OUT_ROOT.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": {i:c for i,c in enumerate(CLASSES)}
    }
    (OUT_ROOT/"data.yaml").write_text(json.dumps(data_yaml, indent=2))
    print("\nWrote:", OUT_ROOT.resolve())
    print("Images:", len(imgs))
    print("Classes:", CLASSES)

if __name__ == "__main__":
    main()
