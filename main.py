import argparse
import csv
import hashlib
import json
import math
import random
import time
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import MEAN, STD, RopeImages, extract_features, split_dataset
from models import network


EXPERIMENT_DIR = Path(__file__).resolve().parent
CLASSES = ["normal", "Strand Coreout"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_device(name="auto"):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if name not in ("cuda", "mps", "cpu"):
        raise ValueError(f"Unknown device: {name}")
    return name


def make_dataloader(
        train_data,
        test_data,
        bs,
        seed,
):
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=bs,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=bs,
        shuffle=False,
    )
    return train_loader, test_loader


def train_model(
        model,
        train_loader,
        device,
        optimizer,
        loss_func,
):
    model.train()

    for x, t in train_loader:
        x = x.to(device)
        t = t.to(device)

        optimizer.zero_grad(set_to_none=True)

        y = model(x)
        loss = loss_func(y, t).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")

        loss.backward()
        optimizer.step()

    # 既存実験と評価条件を揃え、学習指標は更新後・Dropoutなし・クラス重みなしで求める。
    return test_model(
        model,
        train_loader,
        device,
        torch.nn.CrossEntropyLoss(),
    )


@torch.no_grad()
def test_model(
        model,
        test_loader,
        device,
        loss_func,
        return_matrix=False,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_num = 0
    matrix = torch.zeros(2, 2, dtype=torch.long)

    for x, t in test_loader:
        x = x.to(device)
        t = t.to(device)

        y = model(x)
        loss = loss_func(y, t)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite evaluation loss")

        batch_size = t.size(0)
        pred = y.argmax(dim=1)
        total_loss += loss.item() * batch_size
        total_correct += pred.eq(t).sum().item()
        total_num += batch_size
        matrix += torch.bincount((t * 2 + pred).cpu(), minlength=4).reshape(2, 2)

    avg_loss = total_loss / total_num
    avg_acc = total_correct / total_num

    if return_matrix:
        return avg_loss, avg_acc, matrix.tolist()
    return avg_loss, avg_acc


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(history, path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_plots(history, matrix, classes, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = output / "img"
    images.mkdir()
    for key, filename in [("loss", "train_loss.png"), ("accuracy", "test_acc.png")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for split in ("train", "validation"):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{key}"] for r in history], label=split)
        ax.set(xlabel="Epoch", ylabel=key)
        ax.legend()
        fig.tight_layout()
        fig.savefig(images / filename, dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    values = np.array(matrix)
    ax.imshow(values, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=classes, yticklabels=classes,
           xlabel="Predicted", ylabel="Actual", title="Held-out test")
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > values.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(images / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def resolve_path(path, base=EXPERIMENT_DIR):
    # 相対パスは、起動した場所ではなくmain.py（または設定ファイル）の場所を基準にする。
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def config_defaults(path):
    # 以前の --config config.toml も使えるようにする。通常はCLI引数で指定する。
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    data, model, training = (config[k] for k in ("data", "model", "training"))
    if data["classes"] != CLASSES or model["architecture"] != "mobilenet_v3_small":
        raise ValueError("This experiment requires normal/Coreout and mobilenet_v3_small")
    return {
        "seed": training["seed"], "lr": training["learning_rate"],
        "bs": training["batch_size"], "epochs": training["epochs"],
        "device": training["device"], "weight_decay": training["weight_decay"],
        "num_workers": training["num_workers"], "num_threads": training["num_threads"],
        "image_size": model["image_size"],
        "data_root": resolve_path(data["root"], path.parent),
        "weights": resolve_path(model["pretrained_weights"], path.parent),
        "results_root": resolve_path(training["output_root"], path.parent),
        "validation_fraction": data["validation_fraction"],
        "test_fraction": data["test_fraction"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bs", "--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Relative to main.py")
    parser.add_argument("--weights", type=Path, default=Path("weights/mobilenet_v3_small-047dcff4.pth"), help="Relative to main.py")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Relative to main.py")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Relative to main.py")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--config", type=Path, help="Relative to main.py; paths inside are relative to the config file")

    known, _ = parser.parse_known_args()
    if known.config:
        parser.set_defaults(**config_defaults(resolve_path(known.config)))
    args = parser.parse_args()

    seed = args.seed
    lr = args.lr
    bs = args.bs
    epochs = 1 if args.smoke_test else args.epochs
    image_size = args.image_size
    weight_decay = args.weight_decay
    data_dir = resolve_path(args.data_root)
    weights_path = resolve_path(args.weights)
    output_dim = len(CLASSES)
    training_started = datetime.now()
    training_date = training_started.strftime("%Y%m%d")
    timestamp = training_started.strftime("%Y%m%d_%H%M%S_%f")
    results_dir = resolve_path(args.output) if args.output else (
        resolve_path(args.results_root) / f"lr={lr}_bs={bs}" / f"seed={seed}" /
        (("smoke_" if args.smoke_test else "run_") + timestamp)
    )
    model_file = f"mobilenet_v3_small_rope_finetuned_{training_date}.pt"
    model_path = results_dir / model_file
    if not math.isfinite(lr) or lr <= 0 or bs < 1 or epochs < 1 or image_size < 32:
        parser.error("learning rate, batch size, epochs and image size must be positive")
    if args.num_workers < 0 or args.num_threads < 1 or not math.isfinite(weight_decay) or weight_decay < 0:
        parser.error("invalid workers, threads or weight decay")
    if not (0 < args.validation_fraction < 1 and 0 < args.test_fraction < 1 and args.validation_fraction + args.test_fraction < 1):
        parser.error("invalid validation/test fractions")
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if results_dir.exists():
        raise FileExistsError(results_dir)

    set_seed(seed)
    torch.set_num_threads(args.num_threads)
    device = set_device(args.device)
    started = time.monotonic()
    print(f"device: {device}", flush=True)
    print(f"seed: {seed} lr: {lr} bs: {bs} epochs: {epochs}", flush=True)

    split_info = split_dataset(
        data_dir, CLASSES, seed,
        args.validation_fraction, args.test_fraction, args.smoke_test,
    )
    print(f"splits: {split_info['counts']}", flush=True)
    image_data = RopeImages(data_dir, split_info["rows"], image_size)
    model = network(weights_path, output_dim).to(device)
    # 画像を一度だけ特徴に変換するため、以下のloaderのxは画像ではなく576次元特徴になる。
    feature_data = extract_features(model, image_data, bs, device, args.num_workers)
    train_data = Subset(feature_data, split_info["indices"]["train"])
    test_data = Subset(feature_data, split_info["indices"]["validation"])
    final_data = Subset(feature_data, split_info["indices"]["test"])

    train_loader, test_loader = make_dataloader(
        train_data=train_data,
        test_data=test_data,
        bs=bs,
        seed=seed,
    )
    final_loader = DataLoader(dataset=final_data, batch_size=bs, shuffle=False)
    # test_loaderはモデル選択用の検証データ。最終テストはfinal_loaderに保留する。
    train_labels = feature_data.tensors[1][split_info["indices"]["train"]]
    counts = torch.bincount(train_labels, minlength=output_dim)
    class_weights = (len(train_labels) / (output_dim * counts.float())).to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    test_loss_func = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    results_dir.mkdir(parents=True)
    save_json(results_dir / "split.json", split_info)
    save_json(results_dir / "config.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "effective_epochs": epochs, "classes": CLASSES, "device": device,
        "training_date": training_date, "model_file": model_file,
        "torch": torch.__version__, "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
    })

    history = []
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_model(
            model.classifier,
            train_loader,
            device,
            optimizer,
            loss_func,
        )
        test_loss, test_acc = test_model(
            model.classifier,
            test_loader,
            device,
            test_loss_func,
        )
        print(
            f"epoch: {epoch}/{epochs} "
            f"train loss: {train_loss:.6f} "
            f"train acc: {train_acc:.6f} "
            f"validation loss: {test_loss:.6f} "
            f"validation acc: {test_acc:.6f}",
            flush=True,
        )
        history.append({
            "optimizer": "AdamW", "seed": seed, "epoch": epoch,
            "train_loss": train_loss, "train_accuracy": train_acc,
            "validation_loss": test_loss, "validation_accuracy": test_acc,
            "test_loss": "", "test_accuracy": "",
        })
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                "architecture": "mobilenet_v3_small", "classes": CLASSES,
                "image_size": image_size, "epoch": epoch, "training_date": training_date,
                "preprocessing": {"resize": "PIL.ImageOps.pad", "interpolation": "bilinear",
                                  "fill": [124, 116, 104], "mean": MEAN, "std": STD},
                "validation": {"loss": test_loss, "accuracy": test_acc},
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }, model_path)
        save_csv(history, results_dir / "results.csv")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    final_loss, final_acc, matrix = test_model(
        model.classifier, final_loader, device, test_loss_func, return_matrix=True,
    )
    history[checkpoint["epoch"] - 1].update(test_loss=final_loss, test_accuracy=final_acc)
    save_csv(history, results_dir / "results.csv")
    values = np.array(matrix)
    per_class = {name: {
        "precision": float(values[i, i] / max(1, values[:, i].sum())),
        "recall": float(values[i, i] / max(1, values[i].sum())),
        "support": int(values[i].sum()),
    } for i, name in enumerate(CLASSES)}
    save_json(results_dir / "metrics.json", {
        "device": device, "epochs": epochs, "best_epoch": checkpoint["epoch"],
        "classes": CLASSES, "counts": split_info["counts"],
        "best_validation": checkpoint["validation"],
        "held_out_test": {"loss": final_loss, "accuracy": final_acc,
                          "confusion_matrix": matrix, "per_class": per_class},
        "elapsed_seconds": time.monotonic() - started,
    })
    save_plots(history, matrix, CLASSES, results_dir)

    restored = network(weights_path, output_dim).to(device)
    restored.load_state_dict(checkpoint["state_dict"])
    restored.eval()
    first = split_info["indices"]["test"][0]
    image = RopeImages(data_dir, [split_info["rows"][first]], image_size)[0].unsqueeze(0).to(device)
    feature = feature_data.tensors[0][first:first + 1].to(device)
    with torch.no_grad():
        direct = restored(image)
        cached = restored.classifier(feature)
    if not torch.isfinite(direct).all() or list(direct.shape) != [1, 2] or not torch.allclose(direct, cached, rtol=1e-3, atol=1e-4):
        raise RuntimeError("Saved model inference verification failed")
    save_json(results_dir / "verification.json", {
        "checkpoint_reload": True, "full_model_matches_cached_features": True,
        "input_shape": list(image.shape), "output_shape": list(direct.shape),
    })
    print(f"COMPLETE: best epoch={checkpoint['epoch']} final test acc={final_acc:.6f} output={results_dir}", flush=True)





if __name__ == "__main__":
    main()
