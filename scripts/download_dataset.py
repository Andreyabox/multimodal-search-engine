from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "hltcoe/MultiVENT2.0"
REPO_TYPE = "dataset"
DEFAULT_TARGET = Path(__file__).resolve().parents[1] / "data" / "multi_vent_2"


def _is_within_directory(base_dir: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base_dir.resolve())
        return True
    except Exception:
        return False


def safe_extract_tar(tar_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            member_path = dest_dir / member.name
            if not _is_within_directory(dest_dir, member_path):
                raise RuntimeError(
                    f"Unsafe path in tar (path traversal): {tar_path} -> {member.name}"
                )
        tf.extractall(path=dest_dir)


def extract_archives_under(root: Path, delete_archives: bool) -> int:
    archives = sorted(root.rglob("*.tar*"))
    if not archives:
        print("Архивы .tar* не найдены — распаковывать нечего.")
        return 0

    print(f"Найдено архивов: {len(archives)}. Распаковываем в {root} ...")
    for p in archives:
        print(f"- extract: {p.relative_to(root)}")
        safe_extract_tar(p, root)
        if delete_archives:
            p.unlink(missing_ok=True)
    print("Распаковка завершена.")
    return len(archives)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MultiVENT 2.0 from HF Hub.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--full",
        action="store_true",
        help="Скачать полный датасет (~1.93 TB).",
    )
    group.add_argument(
        "--subset",
        nargs="+",
        metavar="SHARD",
        help="Скачать только указанную часть train/<id>, например: 000724",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Куда сохранять (по умолчанию: {DEFAULT_TARGET}).",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="После скачивания распаковать все .tar* архивы внутри target.",
    )
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="Удалить .tar* архивы после успешной распаковки (требует --extract).",
    )
    return parser.parse_args()


def build_allow_patterns(args: argparse.Namespace) -> list[str] | None:
    if args.full:
        return None
    patterns: list[str] = []
    for shard in args.subset:
        patterns.append(f"train/{shard}/*")
        patterns.append(f"videos/{shard}*")
        patterns.append(f"audios/{shard}*")
    patterns.append("*.md")
    patterns.append("*.json")
    patterns.append("*.jsonl")
    return patterns


def main() -> int:
    args = parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN не задан. Получите токен на "
            "https://huggingface.co/settings/tokens и положите в .env.",
            file=sys.stderr,
        )
        return 1

    args.target.mkdir(parents=True, exist_ok=True)

    allow_patterns = build_allow_patterns(args)
    print(f"Скачиваем {REPO_ID} в {args.target}")
    if allow_patterns is None:
        print("Режим: ПОЛНЫЙ датасет (~1.93 TB).")
    else:
        print(f"Режим: подвыборка из ({len(args.subset)} частей): {args.subset}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(args.target),
        allow_patterns=allow_patterns,
        token=token,
        max_workers=4,
    )

    if args.delete_archives and not args.extract:
        print(
            "ERROR: --delete-archives можно использовать только вместе с --extract.",
            file=sys.stderr,
        )
        return 1
    if args.extract:
        extract_archives_under(args.target, delete_archives=args.delete_archives)
    print("Готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
