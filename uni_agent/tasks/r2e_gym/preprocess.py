"""Preprocess R2E-Gym into the uni-agent task format."""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset


def build_r2e_gym(
    data_source: str = "R2E-Gym/R2E-Gym-Subset",
    split: str = "train",
    image_prefix: str = "",
    max_instances: int | None = None,
):
    def process(example):
        repo_name = example.get("repo_name", "")
        commit_hash = example.get("commit_hash", "")
        instance_id = example.get("instance_id") or f"{repo_name}__{commit_hash[:10]}"
        raw_image = example.get("docker_image") or instance_id
        image = f"{image_prefix.rstrip('/')}/{raw_image}" if image_prefix and "/" not in raw_image else raw_image

        metadata = {
            "instance_id": instance_id,
            "repo": repo_name,
            "commit_hash": commit_hash,
            "problem_statement": example["problem_statement"],
            "expected_output_json": example["expected_output_json"],
            "docker_image": image,
        }
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": example["problem_statement"]}],
            "extra_info": {
                "tools_kwargs": {
                    "task": {
                        "name": "r2e_gym",
                        "sandbox": {"image": image},
                        "metadata": metadata,
                    }
                }
            },
        }

    dataset = load_dataset(data_source, split=split)
    if max_instances is not None and max_instances >= 0:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
    return dataset.map(process, remove_columns=dataset.column_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", default="R2E-Gym/R2E-Gym-Subset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-prefix", default=os.getenv("IMAGE_PREFIX", ""))
    parser.add_argument("--local-save-dir", default="~/data/r2e_gym")
    parser.add_argument("--output-name", default="train.parquet")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    ds = build_r2e_gym(
        data_source=args.data_source,
        split=args.split,
        image_prefix=args.image_prefix,
        max_instances=args.max_instances,
    )
    out_path = os.path.join(save_dir, args.output_name)
    ds.to_parquet(out_path)
    print(f"Wrote {len(ds)} instances to {out_path}", flush=True)
