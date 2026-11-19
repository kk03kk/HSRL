"""Install the four pinned author network files and license under artifacts."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.references import reference_directory


def restore():
    item = json.loads((ROOT/"config/reference-assets.json").read_bytes())["hsbg-ppo"]
    target = ROOT/item["directory"]
    if target.exists():
        return reference_directory("hsbg-ppo")
    target.parent.mkdir(parents=True, exist_ok=True)
    repo = item["source_repository"].removeprefix("https://github.com/")
    with TemporaryDirectory(dir=target.parent) as staging:
        source = Path(staging)/"source"
        for name, expected in item["files"].items():
            url = f"https://raw.githubusercontent.com/{repo}/{item['source_revision']}/{name}"
            with urlopen(url, timeout=45) as response:
                data = response.read()
            if sha256(data).hexdigest() != expected:
                raise ValueError(f"source hash mismatch for {name}")
            path = source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        source.rename(target)
    return reference_directory("hsbg-ppo")


if __name__ == "__main__":
    print(restore())
