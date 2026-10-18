from pathlib import Path
from collections import Counter
import hashlib
import random

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class RopeImages(Dataset):
    def __init__(self, root, rows, size):
        self.root = root
        self.rows = rows
        self.size = size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            # 縦横比を維持して余白を追加し、画像端の傷を切り落とさない。
            image = ImageOps.pad(image, (self.size, self.size),
                                 method=Image.Resampling.BILINEAR, color=(124, 116, 104))
        return self.transform(image)


def split_dataset(
        data_root,
        classes,
        seed,
        validation_fraction,
        test_fraction,
        smoke_test=False,
):
    rows = []
    groups = {}
    seen = {}
    # configに指定したクラスだけを読む。extraやritec_dataを自動で取り込まない。
    for label, name in enumerate(classes):
        folder = (data_root / name).resolve()
        if "ritec_data" in folder.parts:
            raise ValueError("ritec_data is reserved for external validation")
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png")
        if smoke_test:
            paths = paths[:6]
        class_groups = {}
        for path in paths:
            # 同一ファイルを別の分割へ入れない。似た画像の判定は行わない。
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != label:
                raise ValueError(f"Conflicting labels for identical image: {path}")
            seen[digest] = label
            index = len(rows)
            rows.append({"path": str(path.relative_to(data_root)),
                         "label": label, "sha256": digest})
            class_groups.setdefault(digest, []).append(index)
        if len(class_groups) < 3:
            raise ValueError(f"At least three distinct images are required: {folder}")
        groups[label] = list(class_groups.values())

    splits = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for class_groups in groups.values():
        rng.shuffle(class_groups)
        count = len(class_groups)
        nv = max(1, round(count * validation_fraction))
        nt = max(1, round(count * test_fraction))
        if nv + nt >= count:
            raise ValueError("Not enough images for train/validation/test splits")
        chunks = {"validation": class_groups[:nv], "test": class_groups[nv:nv + nt],
                  "train": class_groups[nv + nt:]}
        for name, chunk in chunks.items():
            splits[name].extend(i for group in chunk for i in group)

    counts = {name: dict(Counter(classes[rows[i]["label"]] for i in indices))
              for name, indices in splits.items()}
    split_info = {"root": str(data_root), "classes": classes,
                  "rows": rows, "indices": splits, "counts": counts}
    return split_info


def extract_features(
        model,
        image_data,
        bs,
        device,
        num_workers=0,
):
    # 固定した特徴抽出部を一度だけ実行し、各画像の576次元特徴をCPUメモリに保持する。
    image_loader = DataLoader(
        dataset=image_data,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()
    features = []
    with torch.no_grad():
        for batch, image in enumerate(image_loader, 1):
            feature = model.avgpool(model.features(image.to(device))).flatten(1)
            if not torch.isfinite(feature).all():
                raise RuntimeError("Nonfinite image features")
            features.append(feature.cpu())
            if batch == 1 or batch % 10 == 0 or batch == len(image_loader):
                print(f"features {batch}/{len(image_loader)} batches", flush=True)
    labels = torch.tensor([row["label"] for row in image_data.rows])
    return TensorDataset(torch.cat(features), labels)
