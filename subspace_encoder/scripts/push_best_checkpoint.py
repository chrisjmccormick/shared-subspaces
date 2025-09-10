#!/usr/bin/env python3
"""
Push the 'best_checkpoint' folder (from your config) to the Hugging Face Hub.

- Reads <config>.json
- Uses pre_train.best_checkpoint as the local folder to upload
- Uses pre_train.hub_model_id as the target repo
- Relies on cached auth from `huggingface-cli login`
"""

import argparse
import json
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

def parse_args():
    p = argparse.ArgumentParser(description="Push best checkpoint (from config) to Hugging Face Hub")
    p.add_argument("config", type=Path, help="Path to your training config JSON")
    p.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"],
                   help="Hugging Face repo type (default: model)")
    p.add_argument("--branch", default="main", help="Target branch/revision on the Hub (default: main)")
    p.add_argument("--path-in-repo", default=".", help="Subdirectory inside the repo (default: repo root)")
    p.add_argument("--private", action="store_true", help="Create repo as private (no effect if it exists)")
    p.add_argument("--message", default="Upload best checkpoint from config", help="Commit message")
    return p.parse_args()

def main():
    args = parse_args()

    if not args.config.exists():
        print(f"❌ Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r") as f:
        cfg = json.load(f)

    try:
        pre = cfg["pre_train"]
    except KeyError:
        print("❌ Config missing 'pre_train' section.", file=sys.stderr)
        sys.exit(1)

    best_ckpt = Path(pre.get("best_checkpoint", ""))
    repo_id = pre.get("hub_model_id", "")

    if not best_ckpt or not best_ckpt.exists():
        print(f"❌ best_checkpoint path invalid or not found: {best_ckpt}", file=sys.stderr)
        sys.exit(1)

    if not repo_id:
        print("❌ 'hub_model_id' is missing in config['pre_train'].", file=sys.stderr)
        sys.exit(1)

    print(f"📦 Local folder to upload: {best_ckpt.resolve()}")
    print(f"🏷️  Target repo: {repo_id} (type={args.repo_type})   branch: {args.branch}")
    print(f"📁 Path in repo: {args.path_in_repo}")

    api = HfApi()

    # Ensure the repo exists (no-op if it already does)
    create_repo(repo_id=repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)

    # Upload the checkpoint folder
    upload_folder(
        repo_id=repo_id,
        repo_type=args.repo_type,
        folder_path=str(best_ckpt),
        path_in_repo=args.path_in_repo,
        commit_message=args.message,
        revision=args.branch,
        ignore_patterns=[
            ".git*", "*.tmp", "*~", "__pycache__/*", ".ipynb_checkpoints/*", ".DS_Store",
        ],
    )

    base = f"https://huggingface.co/{repo_id}"
    if args.repo_type != "model":
        base += f"?type={args.repo_type}"
    print("\n✅ Done!")
    print(f"   Repo URL: {base}")
    print(f"   Uploaded: {best_ckpt}")

if __name__ == "__main__":
    main()
