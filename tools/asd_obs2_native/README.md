# ASD 0750 evaluation observer

This crate uses the current HSAI engine's types to encode a snapshot. The actual game is advanced by the unchanged, pinned HSAI engine. The observer checks the loaded state and uses the individual seat view.

`src/obs2.rs` is from [asd-HS commit 9b96cf0](https://github.com/asd-HS/hearthstone-bg-agent/commit/9b96cf0c8788d8f3f6caf65b1408b11f8fbee3b3)'s `autotavern-observe2.patch`. The imports point to HSAI's engine and the unused entry point that honors a privileged flag is omitted. Inference uses the same author's `model.py` and `recurrent.py` under `workflows/asd0750`. Its runtime import is wired to the current layout and the SHA-verified static table exported from patched AutoTavern `fb61201a557ddd36a411cf44aac51ce4d9b606ce`. See the accompanying upstream licenses.

For NUS, create a frozen source release with the Player150 CPU-capable environment and run `deploy/slurm/asd0750_paired_eval.sbatch`. The job builds the observer in its own output area. For other environments, build this crate, expose its shared library as the Python extension `_asd_obs2`, and run `python workflows/asd0750_paired_eval.py --help`.
