"""Print the installed versions used by the local bridge environment."""
import importlib.metadata
import json

PACKAGES = ('mlx', 'mlx-lm', 'transformers', 'numpy', 'huggingface-hub')


def environment():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


if __name__ == '__main__':
    print(json.dumps(environment(), indent=2, sort_keys=True))
