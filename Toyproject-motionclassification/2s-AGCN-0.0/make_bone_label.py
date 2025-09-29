# make_bone_sidecars_from_joint_labels_dict.py
# 목적: 파일명 라벨 파싱 없이(sidecar-only) BONE npy 옆에 dict 라벨 .pkl 생성
# 1) label_root(/train/label/*.pkl)에서 라벨 객체를 읽어 base_key 매칭으로 복사(원형 dict 보존)
# 2) 없으면 joint_root(/train/joint/**/*.pkl)에서 같은 base_key의 pkl로 폴백(원형 dict 보존)
# 3) 원본이 int/float/문자열/넘파이 스칼라면 {wrap_key: int} 형태로 감싸 dict로 저장
# ※ base_key = 파일명에서 첫 '_y숫자' 이전의 prefix (파일명 라벨 정보는 사용하지 않음 → 누수 방지)

from pathlib import Path
import argparse, pickle, re, sys
from typing import Any, Tuple

def strip_at_y(name: str) -> str:
    """stem에서 첫 `_y<digits>` 직전까지의 prefix 반환."""
    m = re.search(r"_y\d+", name, flags=re.IGNORECASE)
    return name[:m.start()] if m else name

def to_int_if_possible(obj: Any) -> Tuple[bool, int | None]:
    """정수형으로 안전 변환 시도. (ok, 값) 반환."""
    if isinstance(obj, bool):
        return True, int(obj)
    if isinstance(obj, int):
        return True, obj
    if isinstance(obj, float):
        r = round(obj)
        return (abs(obj - r) < 1e-6, int(r))
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return True, int(obj)
        if isinstance(obj, (np.floating,)):
            fv = float(obj); r = round(fv)
            return (abs(fv - r) < 1e-6, int(r))
    except Exception:
        pass
    if isinstance(obj, str):
        s = obj.strip()
        if s.isdigit():
            return True, int(s)
        try:
            fv = float(s); r = round(fv)
            if abs(fv - r) < 1e-6:
                return True, int(r)
        except Exception:
            pass
    if isinstance(obj, (list, tuple)) and len(obj) == 1:
        return to_int_if_possible(obj[0])
    return False, None

def load_label_obj(pkl_path: Path) -> Any:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def build_label_map(label_root: Path) -> dict[str, Any]:
    """label_root의 .pkl들을 base_key -> 라벨객체로 매핑(원형 보존)."""
    mp: dict[str, Any] = {}
    bad = 0
    for pkl in label_root.rglob("*.pkl"):
        stem = pkl.stem
        base_key = strip_at_y(stem)
        try:
            obj = load_label_obj(pkl)
            mp[base_key] = obj
        except Exception as e:
            bad += 1
            print(f"[WARN] failed to read {pkl}: {e}", file=sys.stderr)
    if bad:
        print(f"[WARN] unreadable label files: {bad}", file=sys.stderr)
    return mp

def find_joint_sidecar_obj(joint_root: Path, bone_base_key: str) -> Any | None:
    """JOINT .pkl 중 base_key가 같은 라벨 객체 찾기(원형 보존)."""
    for p in joint_root.rglob("*.pkl"):
        if strip_at_y(p.stem) == bone_base_key:
            try:
                return load_label_obj(p)
            except Exception:
                pass
    return None

def main():
    ap = argparse.ArgumentParser(description="Create BONE sidecar dict labels by mirroring from /train/label (no filename parsing).")
    ap.add_argument("--label_root", required=True, help="정답 라벨 PKLs (예: /content/.../train/label)")
    ap.add_argument("--bone_root",  required=True, help="BONE NPY 루트 (예: /content/.../train/bone)")
    ap.add_argument("--joint_root", default=None, help="(선택) JOINT 루트(.pkl 폴백)")
    ap.add_argument("--wrap_key",   default="label", help="원본이 정수형일 때 감쌀 dict 키(기본 'label')")
    ap.add_argument("--overwrite",  type=int, default=0, help="기존 bone .pkl 덮어쓰기 (0/1, 기본 0)")
    ap.add_argument("--dry_run",    type=int, default=0, help="미리보기 (0/1, 1이면 실제 파일 쓰지 않음)")
    args = ap.parse_args()

    label_root = Path(args.label_root).resolve()
    bone_root  = Path(args.bone_root).resolve()
    joint_root = Path(args.joint_root).resolve() if args.joint_root else None

    if not label_root.is_dir():
        print(f"[ERR] label_root not found or not a dir: {label_root}", file=sys.stderr); sys.exit(1)
    if not bone_root.is_dir():
        print(f"[ERR] bone_root not found or not a dir: {bone_root}", file=sys.stderr); sys.exit(1)
    if joint_root and not joint_root.exists():
        print(f"[WARN] joint_root not found: {joint_root} (fallback disabled)")
        joint_root = None

    label_map = build_label_map(label_root)
    print(f"[INFO] loaded labels: {len(label_map)} base keys from {label_root}")

    total = written = skipped_exist = missing = failed = 0

    for bnpy in bone_root.rglob("*.npy"):
        total += 1
        bpkl = bnpy.with_suffix(".pkl")
        if bpkl.exists() and not args.overwrite:
            skipped_exist += 1
            continue

        base_key = strip_at_y(bnpy.stem)

        # 1) label_root 우선
        obj = label_map.get(base_key, None)

        # 2) joint 폴백
        if obj is None and joint_root is not None:
            obj = find_joint_sidecar_obj(joint_root, base_key)

        if obj is None:
            missing += 1
            continue

        # 출력은 항상 dict
        if isinstance(obj, dict):
            out_obj = obj  # 원형 보존
        else:
            ok, intval = to_int_if_possible(obj)
            out_obj = {args.wrap_key: int(intval)} if ok else {args.wrap_key: obj}

        try:
            if not args.dry_run:
                with open(bpkl, "wb") as f:
                    pickle.dump(out_obj, f)
            written += 1
        except Exception as e:
            failed += 1
            print(f"[ERR] write failed for {bpkl}: {e}", file=sys.stderr)

    print(f"[DONE] bone files scanned: {total}")
    print(f"       written: {written}")
    print(f"       skipped (already exists): {skipped_exist}")
    print(f"       missing (no source label found): {missing}")
    print(f"       failed writes: {failed}")

if __name__ == "__main__":
    main()
