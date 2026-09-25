"""Download the pinned, prequantized MLX bridge model snapshot."""
import argparse

from huggingface_hub import snapshot_download

MODEL_REPOSITORY = 'mlx-community/Qwen2.5-7B-Instruct-4bit'
MODEL_REVISION = 'c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', help='optional Hugging Face cache directory')
    args = parser.parse_args()
    path = snapshot_download(
        repo_id=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        cache_dir=args.cache_dir,
    )
    print(path)


if __name__ == '__main__':
    main()
