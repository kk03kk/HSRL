"""Frozen native opponent fields for warband experiments."""
import hashlib
import json
from pathlib import Path

from taverngym._core import BcArena, WarbandBank


class Fields:
    def __init__(self, plan):
        self.spec = json.loads(Path(plan).read_text(encoding="utf-8"))
        if not self.spec:
            raise ValueError("An arena plan must define at least one field")
        self.banks = {}
        self.identity = {}
        for name, definition in self.spec.items():
            if set(definition) - {"mixed", "bank", "seats", "start", "level", "mode"}:
                raise ValueError(f"Unknown options for field {name}")
            self.identity[name] = definition.copy()
            if path := definition.get("bank"):
                if path not in self.banks:
                    with Path(path).open("rb") as file:
                        digest = hashlib.file_digest(file, "sha256").hexdigest()
                    self.banks[path] = WarbandBank(path), digest
                self.identity[name]["bank_sha256"] = self.banks[path][1]

    def __call__(self, seeds, name, workers):
        spec = self.spec[name]
        mixed = spec.get("mixed", False)
        if "bank" not in spec:
            return BcArena(seeds, mixed, workers)
        return BcArena.with_warbands(seeds, mixed, workers, self.banks[spec["bank"]][0],
                                     spec["seats"], spec["start"], spec["level"], spec["mode"])
