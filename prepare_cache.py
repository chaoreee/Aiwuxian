from __future__ import annotations

import argparse

from src.data import build_angle_delay_cache, build_map_feature_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Round2_Map")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--delay-bins", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = build_angle_delay_cache(args.data_dir, args.cache_dir, args.delay_bins, args.overwrite)
    print("cache ready:", *paths)
    map_paths = build_map_feature_cache(args.data_dir, args.cache_dir, args.overwrite)
    print("map features ready:", *map_paths)


if __name__ == "__main__":
    main()
