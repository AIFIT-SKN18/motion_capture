
import os, sys, argparse, json, glob, subprocess, shlex, pathlib

HERE = pathlib.Path(__file__).resolve().parent
AUG_SCRIPT = str(HERE / "augment_variants.py")

def infer_dataset(path: str, fallback: str = "ntu/xsub") -> str:
    p = path.replace("\\", "/").lower()
    if "/xsub/" in p or p.endswith("/xsub") or p.endswith("\\xsub"):
        return "ntu/xsub"
    if "/xview/" in p or p.endswith("/xview") or p.endswith("\\xview"):
        return "ntu/xview"
    if "kinetics" in p:
        return "kinetics"
    return fallback

def map_label(joint_path: str) -> str | None:
    # Expect same dir, replace suffix
    if joint_path.endswith("_data_joint.npy"):
        cand = joint_path.replace("_data_joint.npy", "_label.pkl")
        if os.path.isfile(cand):
            return cand
    # fallback: any *_label*.pkl in the same folder (choose the only one)
    d = os.path.dirname(joint_path)
    cands = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pkl") and "_label" in f]
    if len(cands) == 1:
        return cands[0]
    return None

def relpath_under(root: str, path: str) -> str:
    try:
        return os.path.relpath(path, root)
    except Exception:
        return os.path.basename(path)

def build_out_dir(out_root: str, root: str, joint_path: str) -> str:
    rel = os.path.dirname(relpath_under(root, joint_path))
    out_dir = os.path.join(out_root, rel)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def already_done(out_dir: str, base: str, variant: str) -> bool:
    # Check if both joint & bone variant files exist
    j = os.path.join(out_dir, f"{base}_data_joint_aug_{variant}.npy")
    b = os.path.join(out_dir, f"{base}_data_bone_aug_{variant}.npy")
    return os.path.isfile(j) and os.path.isfile(b)

def main():
    ap = argparse.ArgumentParser(description="Auto-map joint/label pairs and run augment_variants in batch.")
    ap.add_argument("--root", required=True, help="Root directory to scan recursively")
    ap.add_argument("--out_root", required=True, help="Where to write augmented files (mirrors structure)")
    ap.add_argument("--pattern", default="**/*_data_joint.npy", help="Glob to find joint files (recursive)")
    ap.add_argument("--dataset", default="auto", choices=["auto","ntu/xsub","ntu/xview","kinetics"], help="Override dataset; 'auto' infers from path")
    ap.add_argument("--variants", default="flip,ud,lrud", help="Variants to generate or 'all'")
    ap.add_argument("--copies", type=int, default=1, help="Augmented copies per variant per sample")
    ap.add_argument("--T", type=int, default=300, help="Target frames")
    ap.add_argument("--yaw_deg", type=float, default=15.0)
    ap.add_argument("--pitch_deg", type=float, default=8.0)
    ap.add_argument("--roll_deg", type=float, default=8.0)
    ap.add_argument("--require_label", action="store_true", help="Skip files without labels if set")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if both joint/bone outputs already exist")
    ap.add_argument("--dry_run", action="store_true", help="Print planned commands without running")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    joints = sorted(glob.glob(os.path.join(root, args.pattern), recursive=True))
    if not joints:
        print(f"[WARN] No joint files found under {root} with pattern {args.pattern}")
        return 0

    plan = []
    for jpath in joints:
        jpath = os.path.abspath(jpath)
        base = os.path.splitext(os.path.basename(jpath))[0].replace("_data_joint","")
        lpath = map_label(jpath)
        if args.require_label and not lpath:
            print(f"[SKIP] No label for {jpath}")
            continue

        ds = args.dataset if args.dataset != "auto" else infer_dataset(jpath)
        out_dir = build_out_dir(out_root, root, jpath)

        # skip-existing check per variant
        if args.skip_existing:
            # expand variants list
            variants = "all" if args.variants.strip().lower() == "all" else [v.strip() for v in args.variants.split(",")]
            if variants != "all":
                all_exist = all(already_done(out_dir, base, v) for v in variants)
                if all_exist:
                    print(f"[SKIP] All variants already exist for {jpath}")
                    continue

        cmd = [sys.executable, AUG_SCRIPT,
               "--in_joint", jpath,
               "--out_dir", out_dir,
               "--dataset", ds,
               "--variants", args.variants,
               "--copies", str(args.copies),
               "--T", str(args.T),
               "--yaw_deg", str(args.yaw_deg),
               "--pitch_deg", str(args.pitch_deg),
               "--roll_deg", str(args.roll_deg)]
        if lpath:
            cmd += ["--in_label", lpath]

        plan.append(dict(joint=jpath, label=lpath, dataset=ds, out_dir=out_dir, cmd=cmd))

    print(f"[INFO] Planned jobs: {len(plan)}")
    if args.dry_run:
        for job in plan:
            print(">", " ".join(shlex.quote(c) for c in job["cmd"]))
        return 0

    # Execute
    for i, job in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {job['joint']}  dataset={job['dataset']}  out={job['out_dir']}")
        if job["label"]:
            print(f"   label: {job['label']}")
        print("   running:", " ".join(shlex.quote(c) for c in job["cmd"]))
        ret = subprocess.call(job["cmd"])
        if ret != 0:
            print(f"[ERROR] Command failed with code {ret}")
            return ret
    print("\n[OK] All done.")
    # Save a summary
    summary = [{"joint": p["joint"], "label": p["label"], "dataset": p["dataset"], "out_dir": p["out_dir"]} for p in plan]
    with open(os.path.join(out_root, "_batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Summary saved at {os.path.join(out_root, '_batch_summary.json')}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
