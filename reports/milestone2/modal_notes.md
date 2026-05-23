# Modal NABirds Training Notes

## Dataset Assumptions

- The Modal app mounts one Volume at `/data`. By default that Volume is named `nabirds-data`; override locally with `NABIRDS_MODAL_VOLUME`.
- Jobs expect either an unpacked dataset at `/data/nabirds` or an archive at `/data/nabirds.tar.gz` that expands to a top-level `nabirds/` directory.
- If only `/data/nabirds.tar.gz` exists, the scaffold extracts it into `/data`. It refuses path-traversal and link members and does not delete or overwrite an existing incomplete `/data/nabirds` directory.
- Manifests, Hugging Face cache, Torch cache, checkpoints, and eval outputs are written under `/data/nabirds_runs`.

## Example Commands

Upload data once:

```bash
modal volume create nabirds-data
modal volume put nabirds-data nabirds.tar.gz /nabirds.tar.gz
```

Run the supported project scripts in Modal:

```bash
modal run scripts/modal_train_nabirds.py --task build-manifests
modal run scripts/modal_train_nabirds.py --task vlm-smoke
modal run scripts/modal_train_nabirds.py --task vlm-full --input-mode full
modal run scripts/modal_train_nabirds.py --task train-visual --input-mode full --epochs 10 --batch-size 64
modal run scripts/modal_train_nabirds.py --task train-visual --input-mode bbox --epochs 10 --batch-size 64
modal run scripts/modal_train_nabirds.py --task train-fused --branch-mode shared --epochs 10 --batch-size 32
modal run scripts/modal_train_nabirds.py --task train-vlm-adapter --epochs 5 --batch-size 32
```

For quick training smoke tests:

```bash
modal run scripts/modal_train_nabirds.py --task train-visual --model resnet18 --epochs 1 --batch-size 8 --limit 32 --no-pretrained
modal run scripts/modal_train_nabirds.py --task train-fused --model resnet18 --epochs 1 --batch-size 4 --limit 32 --no-pretrained
modal run scripts/modal_train_nabirds.py --task train-vlm-adapter --epochs 1 --batch-size 4 --limit 32
```

Set `NABIRDS_MODAL_GPU=A100` before `modal run` if a different Modal GPU class is needed.
