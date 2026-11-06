"""验证固定 AutoTavern 卡表与历史 checkpoint 的有序词表兼容。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = "1c2086a997afde46d68c6c5992cab194ddcfce88"
EXPECTED = "7a02d10794cf80c56fa77b1dd235d49b892c8d3e31c2cf56887f696e2cf922a6"
VOCAB = "5041e83d810e846b3cbc7d7ff056b84e301472d6ff021ffd1971a18a248e05fb"


def main():
    raw = (ROOT / "engine/crates/hsbg-engine/data/cards.json").read_bytes().replace(b"\r\n", b"\n")
    db = json.loads(raw)
    ids = [card["id"] for card in db["cards"]]
    if (hashlib.sha256(raw).hexdigest() != EXPECTED or len(ids) != 3983
            or hashlib.sha256(json.dumps(ids).encode()).hexdigest() != VOCAB
            or db["meta"]["client_version"] != "37.0.2.256589"
            or db["meta"]["server_patch"] != "37.0.2"):
        raise SystemExit("37.0.2 卡表摘要、顺序或规则版本不符")
    print(f"37.0.2.256589 upstream={UPSTREAM} cards={len(ids)} vocab={VOCAB}")


if __name__ == "__main__":
    main()
