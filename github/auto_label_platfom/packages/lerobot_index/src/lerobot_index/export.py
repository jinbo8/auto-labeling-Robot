from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_sidecar_episode(
    path: Path,
    *,
    episode_index: int,
    task: str | None,
    fps: float,
    items: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_index": episode_index,
        "task": task,
        "fps": fps,
        "items": items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def annotations_to_coco(
    annotations: list[dict[str, Any]],
    *,
    categories: list[str],
    image_size: tuple[int, int] = (640, 480),
) -> dict[str, Any]:
    """Build a minimal COCO detection dict from platform annotation dicts.

    Expects items with frame_index, camera_key, label, geometry.bbox [x,y,w,h].
    """
    w, h = image_size
    cat_ids = {name: i + 1 for i, name in enumerate(categories)}
    images = []
    anns = []
    image_key_to_id: dict[tuple[str, int], int] = {}
    ann_id = 1
    for item in annotations:
        geom = item.get("geometry") or {}
        if geom.get("type") != "bbox" and not item.get("bbox"):
            continue
        bbox = geom.get("bbox") or item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        cam = item.get("camera_key") or "default"
        fi = int(item["frame_index"])
        key = (cam, fi)
        if key not in image_key_to_id:
            image_id = len(images) + 1
            image_key_to_id[key] = image_id
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{cam}/frame_{fi:06d}.jpg",
                    "width": w,
                    "height": h,
                    "camera_key": cam,
                    "frame_index": fi,
                }
            )
        label = item.get("label") or "object"
        if label not in cat_ids:
            cat_ids[label] = len(cat_ids) + 1
        anns.append(
            {
                "id": ann_id,
                "image_id": image_key_to_id[key],
                "category_id": cat_ids[label],
                "bbox": [float(x) for x in bbox],
                "area": float(bbox[2]) * float(bbox[3]),
                "iscrowd": 0,
            }
        )
        ann_id += 1
    cats = [{"id": i, "name": n} for n, i in sorted(cat_ids.items(), key=lambda x: x[1])]
    return {"images": images, "annotations": anns, "categories": cats}


def build_export_zip(
    out_zip: Path,
    *,
    manifest: dict[str, Any],
    sidecar_by_episode: dict[int, dict[str, Any]],
    coco: dict[str, Any] | None = None,
) -> Path:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for ei, payload in sorted(sidecar_by_episode.items()):
            zf.writestr(
                f"annotations/episode_{ei:05d}.json",
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
        if coco is not None:
            zf.writestr("coco/instances.json", json.dumps(coco, ensure_ascii=False, indent=2))
    return out_zip


def group_items_by_episode(items: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        grouped[int(it["episode_index"])].append(it)
    return grouped
