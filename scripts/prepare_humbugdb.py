from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.preprocess import _label_for_path  # noqa: E402  (reuse labeling rule)

SPECIES_COLUMN_CANDIDATES = ("species", "label", "class")
ID_COLUMN_CANDIDATES = ("id", "name", "filename", "fname", "file", "sample", "record_id")
SOUND_TYPE_CANDIDATES = ("sound_type", "type", "category")
UNLABELED_FOLDER = "background"  # rows with no species -> useful negatives (label 0)


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:           # exact (case-insensitive) match first
        if cand in lower:
            return lower[cand]
    for col_lower, original in lower.items():  # then substring match
        if any(cand in col_lower for cand in candidates):
            return original
    return None


def _sanitize(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _index_audio(audio_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in audio_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in config.AUDIO_EXTENSIONS:
            index.setdefault(path.stem, path)
            index.setdefault(path.name, path)
    return index


def _resolve_audio(row_id: str, index: dict[str, Path]) -> Path | None:
    candidates = (row_id, row_id.lstrip("0") or row_id)
    for key in candidates:
        if key in index:
            return index[key]
        for ext in config.AUDIO_EXTENSIONS:
            if key + ext in index:
                return index[key + ext]
    return None


def _place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "link":
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)  # Windows without privilege -> fall back to copy
    else:
        shutil.copy2(src, dst)


def prepare(src: Path, metadata: Path | None, audio_root: Path | None,
            out: Path, mode: str) -> None:
    if metadata is None:
        csvs = sorted(src.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found under {src}; pass --metadata explicitly")
        metadata = csvs[0]
    audio_root = audio_root or src

    df = pd.read_csv(metadata)
    species_col = _pick_column(df, SPECIES_COLUMN_CANDIDATES)
    id_col = _pick_column(df, ID_COLUMN_CANDIDATES)
    sound_col = _pick_column(df, SOUND_TYPE_CANDIDATES)
    if id_col is None:
        raise ValueError(f"Could not detect an id/filename column in {list(df.columns)}")
    print(f"Metadata : {metadata}  ({len(df)} rows)")
    print(f"Columns  : id='{id_col}'  species='{species_col}'  sound_type='{sound_col}'")

    index = _index_audio(audio_root)
    print(f"Audio    : {len(index)} files indexed under {audio_root}\n")

    matched = unmatched = 0
    species_counts: dict[str, int] = {}
    for row in df.itertuples(index=False):
        row_id = str(getattr(row, id_col))
        audio_path = _resolve_audio(row_id, index)
        if audio_path is None:
            unmatched += 1
            continue

        species = getattr(row, species_col) if species_col else None
        folder = _sanitize(species) if pd.notna(species) and str(species).strip() else ""
        if not folder and sound_col:
            sval = getattr(row, sound_col)
            folder = _sanitize(sval) if pd.notna(sval) else ""
        folder = folder or UNLABELED_FOLDER

        _place(audio_path, out / folder / audio_path.name, mode)
        species_counts[folder] = species_counts.get(folder, 0) + 1
        matched += 1

    print(f"Placed {matched} files into {out}  (unmatched rows: {unmatched})")
    print("\nPer-species counts and resulting binary label:")
    pos = neg = 0
    for folder, count in sorted(species_counts.items(), key=lambda kv: -kv[1]):
        label = _label_for_path((folder,))
        tag = "AEDES (1)" if label == config.LABEL_AEDES else "non    (0)"
        if label == config.LABEL_AEDES:
            pos += count
        else:
            neg += count
        print(f"  {folder:<28} {count:>6}   -> {tag}")
    print(f"\nBinary totals: positives(aegypti)={pos}  negatives={neg}")
    if pos == 0:
        print("[warn] no Aedes aegypti files matched — check the species column values")


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize a metadata-CSV dataset by species")
    parser.add_argument("--src", default=str(config.RAW_DIR / "_humbugdb_download"),
                        help="folder where the dataset was unzipped (CSV + audio)")
    parser.add_argument("--metadata", default=None, help="explicit CSV path (else auto-find under --src)")
    parser.add_argument("--audio-root", default=None, help="audio folder (default: --src)")
    parser.add_argument("--out", default=str(config.RAW_DIR / "humbugdb"),
                        help="destination dataset folder under data/raw/")
    parser.add_argument("--mode", choices=("copy", "link", "move"), default="copy",
                        help="copy (safe default), link (symlink), or move the files")
    args = parser.parse_args()

    prepare(
        src=Path(args.src),
        metadata=Path(args.metadata) if args.metadata else None,
        audio_root=Path(args.audio_root) if args.audio_root else None,
        out=Path(args.out),
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
