from course import *
import torch.nn.functional as F


def l1_and_pca(out):
    from sklearn.datasets import make_moons
    from sklearn.decomposition import PCA

    X, y = make_moons(1000, noise=0.25, random_state=11)
    train, val, test = split3(X, y)
    rows = []
    curves = {}
    for alpha in [0.0, 1e-5, 1e-4, 1e-3]:
        seed_all()
        m = MLP(2, 2, (32, 32))
        m, h, t = fit(m, train, val, epochs=80, lr=0.01, l1=alpha)
        metrics, _, _ = classify(m, val)
        weights = torch.cat([p.detach().flatten() for p in m.parameters()])
        rows.append(
            {
                "L1": alpha,
                **metrics,
                "weight_L1": weights.abs().sum().item(),
                "fraction_abs_below_1e-3": (weights.abs() < 1e-3).float().mean().item(),
            }
        )
        curves[str(alpha)] = h
    table(rows, out, "l1_regularization")
    histories(curves, out, "l1_curves")
    rng = np.random.default_rng(11)
    x = rng.multivariate_normal(
        [0, 0, 0], [[4, 2, 0.2], [2, 3, 0.3], [0.2, 0.3, 1]], size=500
    )
    centered = x - x.mean(0)
    cov = centered.T @ centered / (len(x) - 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    projection = centered @ vectors[:, :2]
    pca = PCA(2).fit(x)
    assert np.allclose(values[:2], pca.explained_variance_)
    assert np.allclose(
        np.abs(vectors[:, :2].T @ pca.components_.T), np.eye(2), atol=1e-10
    )
    table(
        {
            "component": [1, 2, 3],
            "eigenvalue": values,
            "explained_variance_fraction": values / values.sum(),
        },
        out,
        "manual_pca",
    )
    plt.scatter(projection[:, 0], projection[:, 1], s=10, alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA вручную через ковариацию и eigh")
    showfig("manual_pca", out)
    note(
        "L1 приближает веса к нулю. Долю почти нулевых весов считаю с допуском 1e−3. В PCA собственные векторы ковариации задают направления, собственные значения — дисперсии. При сравнении учитываю произвольный знак векторов."
    )


def fashion_data():
    p = DATA / "fashion_subset.npz"
    if not p.exists():
        import gzip, urllib.request

        folder = DATA / "fashion_raw"
        folder.mkdir(exist_ok=True)
        arrays = {}
        for kind, name, skip in [
            ("images", "train-images-idx3-ubyte.gz", 16),
            ("labels", "train-labels-idx1-ubyte.gz", 8),
        ]:
            path = folder / name
            if not path.exists():
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
                    + name,
                    path,
                )
            with gzip.open(path, "rb") as f:
                arrays[kind] = np.frombuffer(f.read(), dtype=np.uint8)[skip:]
        X = arrays["images"].reshape(-1, 1, 28, 28).astype("float32") / 255.0
        y = arrays["labels"]
        ids, _ = train_test_split(
            np.arange(len(y)), train_size=1200, stratify=y, random_state=11
        )
        np.savez_compressed(p, X=X[ids], y=y[ids])
    z = np.load(p)
    return split3(z["X"], z["y"])


class SmallAlexNet(nn.Module):
    def __init__(self, simplified=False, native=False):
        super().__init__()
        self.size = 28 if native else 96
        if simplified:
            self.features = nn.Sequential(
                nn.Conv2d(1, 12, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(12, 24, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 24, 3, padding=1),
                nn.ReLU(),
            )
            channels = 24
        else:
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 11, stride=4, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2),
                nn.Conv2d(32, 64, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2),
                nn.Conv2d(64, 96, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(96, 96, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(96, 64, 3, padding=1),
                nn.ReLU(),
            )
            channels = 64
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(channels * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.head(
            self.features(
                F.interpolate(
                    x, size=(self.size, self.size), mode="bilinear", align_corners=False
                )
            )
        )


class NiN(nn.Module):
    def __init__(self, pointwise=2):
        super().__init__()
        layers = [nn.Conv2d(1, 16, 5, padding=2), nn.ReLU()]
        for _ in range(pointwise):
            layers += [nn.Conv2d(16, 16, 1), nn.ReLU()]
        layers += [
            nn.MaxPool2d(2),
            nn.Conv2d(16, 10, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def model_cost(model, shape=(1, 1, 28, 28)):
    multiply = 0
    activation_elements = 0
    hooks = []

    def hook(layer, inp, out):
        nonlocal multiply, activation_elements
        activation_elements += out.numel()
        if isinstance(layer, nn.Conv2d):
            multiply += (
                out.numel()
                * (layer.in_channels // layer.groups)
                * np.prod(layer.kernel_size)
            )
        elif isinstance(layer, nn.Linear):
            multiply += out.numel() * layer.in_features

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(shape))
    for h in hooks:
        h.remove()
    return {
        "parameters": count_params(model),
        "multiplications_one_image": int(multiply),
        "parameter_MB_fp32": count_params(model) * 4 / 1024**2,
        "sum_layer_output_MB_fp32": activation_elements * 4 / 1024**2,
    }


def alex_nin(out):
    train, val, test = fashion_data()
    rows = []
    curves = {}
    configs = [
        ("AlexNet_like_96", False, False, 64),
        ("simplified_96", True, False, 64),
        ("simplified_native28", True, True, 64),
        ("native28_batch16", True, True, 16),
        ("native28_batch128", True, True, 128),
    ]
    for name, simple, native, batch in configs:
        seed_all()
        m = SmallAlexNet(simple, native)
        m, h, t = fit(m, train, val, epochs=10, lr=0.003, batch=batch)
        metrics, _, _ = classify(m, val)
        rows.append(
            {"model": name, **metrics, **model_cost(m), "seconds": t, "batch": batch}
        )
        curves[name] = h
    table(rows, out, "alexnet_simplification")
    histories(curves, out, "alexnet_curves")
    rows = []
    for pointwise in [1, 2, 3]:
        seed_all()
        m = NiN(pointwise)
        m, h, t = fit(m, train, val, epochs=10, lr=0.003)
        metrics, _, _ = classify(m, val)
        rows.append(
            {"pointwise_layers": pointwise, **metrics, **model_cost(m), "seconds": t}
        )
    table(rows, out, "nin_pointwise")
    note(
        "Fashion-MNIST: 1200 объектов. AlexNet уменьшен для CPU и небольших изображений. В NiN слои 1×1 смешивают каналы. Умножения считаю для Conv2d и Linear, память — как сумму размеров выходов слоёв. Измерений GPU-памяти здесь нет."
    )


def name_rnn(out):
    import urllib.request, zipfile, unicodedata, string

    path = DATA / "names_data.zip"
    if not path.exists():
        urllib.request.urlretrieve(
            "https://download.pytorch.org/tutorial/data.zip", path
        )
    samples = []
    labels = []
    categories = ["English", "Russian", "Spanish"]
    rng = np.random.default_rng(11)
    with zipfile.ZipFile(path) as z:
        for label, category in enumerate(categories):
            lines = z.read("data/names/" + category + ".txt").decode().splitlines()
            lines = sorted(set(lines))
            ids = rng.choice(len(lines), min(250, len(lines)), replace=False)
            for i in ids:
                name = "".join(
                    c
                    for c in unicodedata.normalize("NFD", lines[i])
                    if unicodedata.category(c) != "Mn"
                ).lower()
                samples.append(name)
                labels.append(label)
    alphabet = {c: i + 2 for i, c in enumerate(string.ascii_lowercase + " -'")}
    xx = torch.tensor(
        [
            [alphabet.get(c, 1) for c in s[:24]] + [0] * (24 - len(s[:24]))
            for s in samples
        ]
    )
    yy = torch.tensor(labels)
    ids = np.arange(len(yy))
    tr, te = train_test_split(ids, test_size=0.2, stratify=yy, random_state=11)
    tr, va = train_test_split(tr, test_size=0.25, stratify=yy[tr], random_state=11)
    from language_music import TextRNN

    rows = []
    curves = {}
    for kind in ["RNN", "LSTM", "GRU"]:
        seed_all()
        m = TextRNN(len(alphabet) + 2, kind)
        m.head = nn.Linear(24, 3)
        m, h, t = fit(m, (xx[tr], yy[tr]), (xx[va], yy[va]), epochs=30, lr=0.005)
        metrics, pred, p = classify(m, (xx[te], yy[te]))
        rows.append({"kind": kind, **metrics, "seconds": t})
        curves[kind] = h
    table(rows, out, "character_names")
    histories(curves, out, "character_names_curves")
    ConfusionMatrixDisplay.from_predictions(yy[te], pred, display_labels=categories)
    plt.title("Символьная GRU: категория фамилии")
    showfig("name_confusion", out)
    errors = np.flatnonzero(pred != yy[te].numpy())[:10]
    table(
        [
            {
                "name": samples[te[j]],
                "true": categories[yy[te[j]]],
                "predicted": categories[pred[j]],
            }
            for j in errors
        ],
        out,
        "name_errors",
    )
    note(
        "Данные — фамилии из PyTorch tutorial. Одна фамилия может встречаться в нескольких языковых группах. Packing исключает PAD из скрытого состояния."
    )


def yolo_resolution(out):
    import os

    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO
    from vision import penn_data, match_boxes

    images, targets = penn_data()
    images = images[50:60]
    targets = targets[50:60]
    weights = DATA / "pretrained" / "yolo11n.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    rows = []
    for size in [192, 256, 320, 384, 448]:
        samples = []
        times = []
        for image, target in zip(images, targets):
            source = (
                (image.permute(1, 2, 0).numpy() * 255)
                .astype("uint8")[:, :, ::-1]
                .copy()
            )
            start = time.perf_counter()
            r = model.predict(
                source,
                imgsz=size,
                rect=False,
                device="cpu",
                conf=0.05,
                iou=0.55,
                verbose=False,
            )[0]
            times.append(time.perf_counter() - start)
            keep = r.boxes.cls.cpu() == 0
            pred = {
                "boxes": r.boxes.xyxy.cpu()[keep],
                "scores": r.boxes.conf.cpu()[keep],
                "labels": torch.ones(int(keep.sum()), dtype=torch.long),
            }
            samples.append(match_boxes(pred, target["boxes"], threshold=0.4))
        total = {k: sum(s[k] for s in samples) for k in ["TP", "FP", "FN"]}
        rows.append(
            {
                "size": size,
                **total,
                "precision": total["TP"] / max(1, total["TP"] + total["FP"]),
                "recall": total["TP"] / max(1, total["TP"] + total["FN"]),
                "median_ms": np.median(times[1:]) * 1000,
            }
        )
    df = table(rows, out, "yolo_resolution")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df["size"], df.recall, "o-", label="Recall")
    ax[0].plot(df["size"], df.precision, "s--", label="Precision")
    ax[0].legend()
    ax[0].set(xlabel="Размер квадратного входа, px", ylabel="Метрика", ylim=(0, 1))
    ax[1].plot(df["size"], df.median_ms, "o-")
    ax[1].set(xlabel="Размер входа, px", ylabel="Медианное время, мс")
    showfig("yolo_resolution_quality", out)
    note(
        "YOLO11n с весами COCO. Вход BGR, rect=False, person=0 переведён в person=1. Первый проход пропускаю при измерении времени. Для всех размеров использую одни десять фотографий и порог 0.4."
    )


def mask_overlap(out):
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn,
        MaskRCNN_ResNet50_FPN_Weights,
    )
    from vision import penn_data
    from PIL import Image

    images, targets = penn_data()
    image = images[0]
    target = targets[0]
    instance_path = (
        DATA / "PennFudanPed" / "PedMasks" / (Path(target["name"]).stem + "_mask.png")
    )
    instance_ids = np.asarray(
        Image.open(instance_path).resize(
            (image.shape[2], image.shape[1]), Image.Resampling.NEAREST
        )
    )
    instance_id = np.unique(instance_ids[instance_ids > 0])[0]
    mask = tensor(instance_ids == instance_id).bool()
    ys, xs = torch.where(mask)
    left, right, top, bottom = (
        int(xs.min()),
        int(xs.max()) + 1,
        int(ys.min()),
        int(ys.max()) + 1,
    )
    crop = image[:, top:bottom, left:right]
    cut = mask[top:bottom, left:right]
    h, w = crop.shape[-2:]
    scale = min(180 / h, 100 / w)
    hh, ww = round(h * scale), round(w * scale)
    person = F.interpolate(crop[None], (hh, ww), mode="bilinear", align_corners=False)[
        0
    ]
    silhouette = F.interpolate(cut.float()[None, None], (hh, ww), mode="nearest")[
        0, 0
    ].bool()
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, min_size=320, max_size=400
    ).eval()
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for j, overlap in enumerate([0.0, 0.25, 0.5, 0.75]):
        canvas = torch.full((3, 260, 320), 0.6)
        masks = []
        for x in [40, 40 + round(ww * (1 - overlap))]:
            m = torch.zeros(260, 320, dtype=torch.bool)
            m[40 : 40 + hh, x : x + ww] = silhouette
            masks.append(m)
            region = canvas[:, 40 : 40 + hh, x : x + ww]
            region[:, silhouette] = person[:, silhouette]
        gt = masks[0] | masks[1]
        with torch.inference_mode():
            p = model([canvas])[0]
        keep = (p["labels"] == 1) & (p["scores"] >= 0.5)
        pred = (
            (p["masks"][keep, 0] > 0.5).any(0) if keep.any() else torch.zeros_like(gt)
        )
        inter = (pred & gt).sum().item()
        dice = 2 * inter / max(1, pred.sum().item() + gt.sum().item())
        rows.append(
            {
                "horizontal_overlap_fraction": overlap,
                "detected_people": int(keep.sum()),
                "union_Dice": dice,
                "union_IoU": inter / max(1, (pred | gt).sum().item()),
            }
        )
        axes[0, j].imshow(canvas.permute(1, 2, 0))
        axes[0, j].set_title(f"Перекрытие {overlap:.0%}")
        axes[1, j].imshow(gt, cmap="gray")
        axes[1, j].contour(pred.numpy(), levels=[0.5], colors=["#2463a2"])
        axes[1, j].set_title(f"Найдено: {int(keep.sum())}, Dice={dice:.2f}")
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    showfig("maskrcnn_overlap", out)
    table(rows, out, "maskrcnn_overlap")
    note(
        "Фигуру вырезаю по исходной маске и дважды размещаю на одном фоне. Белым показана объединённая разметка, голубым — граница Mask R-CNN. Dice объединённой маски может быть высоким, даже если два человека найдены как один объект."
    )
