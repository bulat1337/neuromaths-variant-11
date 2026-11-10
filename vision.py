from course import *
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms.functional import gaussian_blur
from torchvision.ops import box_iou
from PIL import Image

torch.hub.set_dir(str(DATA / "pretrained"))


class TransferNet(nn.Module):
    def __init__(self, pretrained=True, tune=False):
        super().__init__()
        base = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        self.backbone = base.features
        self.head = nn.Linear(576, 10)
        self.tune = tune
        for p in self.backbone.parameters():
            p.requires_grad = False
        if tune:
            for p in self.backbone[-1].parameters():
                p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self.tune:
            self.backbone[-1].train(mode)
        return self

    def embedding(self, x):
        x = F.interpolate(
            x.repeat(1, 3, 1, 1), size=(224, 224), mode="bilinear", align_corners=False
        )
        mean = tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return self.backbone((x - mean) / std).mean((2, 3))

    def forward(self, x):
        return self.head(self.embedding(x))


def _batch_predict(model, data):
    model.eval()
    parts = []
    with torch.no_grad():
        for x in data[0].split(32):
            parts.append(model(x))
    logits = torch.cat(parts)
    y = data[1].numpy()
    p = logits.softmax(1).numpy()
    pred = p.argmax(1)
    return (
        {
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        },
        pred,
        p,
    )


def transfer_experiment(out):
    alltrain, val, test = mnist()
    train = tuple(x[:400] for x in alltrain)
    val = tuple(x[:200] for x in val)
    test = tuple(x[:200] for x in test)
    print(
        "MNIST subset:",
        len(train[1]),
        len(val[1]),
        len(test[1]),
        "shape",
        tuple(train[0].shape),
    )
    # Проверка одинаковых изображений в разных выборках.
    import hashlib

    hashes = [
        {hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in pair[0]}
        for pair in [train, val, test]
    ]
    print(
        "Точные совпадения train/val/test:",
        len(hashes[0] & hashes[1]),
        len(hashes[0] & hashes[2]),
    )
    print("Числа классов train:", np.bincount(train[1].numpy(), minlength=10))
    cache = DATA / "transfer_runs.pt"
    models = {}
    curves = {}
    rows = []
    if cache.exists():
        state = torch.load(cache, weights_only=False)
        for name, entry in state.items():
            m = (
                CNN()
                if name == "scratch_CNN"
                else TransferNet(pretrained=False, tune=name == "fine_tuning")
            )
            m.load_state_dict(entry["state"])
            models[name] = m
            curves[name] = pd.DataFrame(entry["history"])
            rows.append(entry["metrics"])
        note(
            "Загружены веса из data/transfer_runs.pt. Для повторного обучения удалите этот файл."
        )
    else:
        for name in ["scratch_CNN", "frozen_backbone", "fine_tuning"]:
            seed_all()
            m = (
                CNN()
                if name == "scratch_CNN"
                else TransferNet(tune=name == "fine_tuning")
            )
            m, h, t = fit(
                m,
                train,
                val,
                epochs=6 if name != "scratch_CNN" else 12,
                batch=32,
                lr=0.003 if name == "scratch_CNN" else 0.001,
            )
            metrics, _, _ = _batch_predict(m, val)
            rows.append(
                {
                    "model": name,
                    **metrics,
                    "trainable_parameters": count_params(m),
                    "seconds": t,
                }
            )
            models[name] = m
            curves[name] = h
        torch.save(
            {
                k: {
                    "state": m.state_dict(),
                    "history": curves[k].to_dict("records"),
                    "metrics": next(x for x in rows if x["model"] == k),
                }
                for k, m in models.items()
            },
            cache,
        )
    table(rows, out, "transfer_strategies")
    histories(curves, out, "transfer_curves")
    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    for i, a in enumerate(ax.ravel()):
        a.imshow(train[0][i, 0], cmap="gray")
        a.set_title(str(train[1][i].item()))
        a.axis("off")
    showfig("source_examples", out)
    return {"models": models, "train": train, "val": val, "test": test, "rows": rows}


def domain_evaluation(result, out):
    seed_all()
    x, y = result["test"]
    conditions = {
        "clean": x,
        "dark": x * 0.35,
        "bright": torch.clamp(x * 1.5 + 0.1, 0, 1),
        "gray_background": x * 0.65 + 0.35,
        "blur": gaussian_blur(x, [5, 5], 1.2),
        "noise": torch.clamp(x + 0.18 * torch.randn_like(x), 0, 1),
    }
    rows = []
    for name, m in result["models"].items():
        clean = _batch_predict(m, (x, y))[0]["accuracy"]
        for condition, xx in conditions.items():
            metrics, _, _ = _batch_predict(m, (xx, y))
            rows.append(
                {
                    "model": name,
                    "condition": condition,
                    **metrics,
                    "accuracy_drop": clean - metrics["accuracy"],
                }
            )
    df = table(rows, out, "domain_shift")
    df.pivot(index="condition", columns="model", values="accuracy").plot.bar(
        figsize=(11, 4)
    )
    plt.ylabel("Test accuracy")
    plt.xlabel("Условие")
    plt.ylim(0, 1)
    plt.title("Устойчивость к заранее заданным сдвигам")
    showfig("domain_quality", out)
    fig, ax = plt.subplots(1, 6, figsize=(12, 3))
    for a, (k, xx) in zip(ax, conditions.items()):
        a.imshow(xx[0, 0], cmap="gray", vmin=0, vmax=1)
        a.set_title(k, fontsize=9)
        a.axis("off")
    showfig("domain_examples", out)
    worst = df.loc[df.accuracy_drop.idxmax()]
    note(
        f'Наибольшее измеренное падение: {worst["model"]}, условие {worst["condition"]}, Δaccuracy={worst.accuracy_drop:.3f}. Изменение яркости/фона меняет статистику входа, сохраняя смысловую метку цифры. Синтетические сдвиги представляют контролируемую проверку; реальные камеры дополнительно меняют шум, цвет и резкость.'
    )


def embedding_analysis(result, out):
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity

    m = result["models"]["frozen_backbone"]
    m.eval()
    with torch.no_grad():
        et = torch.cat([m.embedding(x) for x in result["train"][0].split(32)]).numpy()
        ee = torch.cat([m.embedding(x) for x in result["test"][0].split(32)]).numpy()
    similarities = cosine_similarity(ee, et)
    idx = similarities.argmax(1)
    pred = result["train"][1].numpy()[idx]
    y = result["test"][1].numpy()
    print(
        "Nearest-train retrieval accuracy:",
        accuracy_score(y, pred),
        "cosine examples:",
        similarities[:3, :3],
    )
    pca = PCA(n_components=2, random_state=11)
    projection = pca.fit_transform(et)
    plt.scatter(
        projection[:, 0], projection[:, 1], c=result["train"][1], cmap="tab10", s=13
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA замороженных MobileNet-эмбеддингов")
    showfig("embedding_pca", out)
    chosen = max(result["rows"], key=lambda row: row["macro_f1"])["model"]
    metrics, pred, p = _batch_predict(result["models"][chosen], result["test"])
    table(
        pd.DataFrame(
            classification_report(y, pred, output_dict=True, zero_division=0)
        ).T.reset_index(names="class"),
        out,
        "per_class_metrics",
    )
    ids = np.flatnonzero(pred != y)[:10]
    table(
        [
            {
                "index": int(i),
                "true": int(y[i]),
                "predicted": int(pred[i]),
                "confidence": float(p[i].max()),
            }
            for i in ids
        ],
        out,
        "errors",
    )
    if len(ids):
        fig, ax = plt.subplots(2, 5, figsize=(11, 5))
        for a, i in zip(ax.ravel(), ids):
            a.imshow(result["test"][0][i, 0], cmap="gray")
            a.set_title(f"{y[i]} → {pred[i]}")
            a.axis("off")
        for a in ax.ravel()[len(ids) :]:
            a.axis("off")
        showfig("error_gallery", out)
    note(
        "PCA может искажать соседство, поэтому дополнительно считаю точность поиска ближайшего изображения. Похожие цифры, неполные штрихи и разная толщина линий дают ошибки."
    )


def penn_data():
    folder = DATA / "PennFudanPed"
    if not folder.exists():
        import urllib.request, zipfile

        archive = DATA / "PennFudanPed.zip"
        if not archive.exists():
            urllib.request.urlretrieve(
                "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip", archive
            )
        with zipfile.ZipFile(archive) as z:
            z.extractall(DATA)
    images = []
    targets = []
    for path in sorted((folder / "PNGImages").glob("*.png"))[:60]:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        scale = 512 / max(w, h)
        size = (round(w * scale), round(h * scale))
        im = im.resize(size)
        mask = np.asarray(
            Image.open(folder / "PedMasks" / (path.stem + "_mask.png")).resize(
                size, Image.Resampling.NEAREST
            )
        )
        boxes = []
        for label in np.unique(mask):
            if label == 0:
                continue
            ys, xs = np.where(mask == label)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        images.append(tensor(np.asarray(im).copy().transpose(2, 0, 1) / 255.0))
        targets.append(
            {
                "name": path.name,
                "boxes": tensor(boxes).reshape(-1, 4),
                "mask": tensor(mask > 0),
            }
        )
    return images, targets


def match_boxes(pred, gt, threshold=0.4):
    keep = (pred["labels"] == 1) & (pred["scores"] >= threshold)
    boxes = pred["boxes"][keep]
    scores = pred["scores"][keep]
    used = set()
    ious = []
    tp = 0
    if len(boxes) and len(gt):
        matrix = box_iou(boxes, gt)
        for idx in torch.argsort(scores, descending=True):
            values, order = matrix[idx].sort(descending=True)
            for iou, j in zip(values.tolist(), order.tolist()):
                if j not in used and iou >= 0.5:
                    used.add(j)
                    tp += 1
                    ious.append(iou)
                    break
    return {
        "TP": tp,
        "FP": len(boxes) - tp,
        "FN": len(gt) - tp,
        "matched_IoU": float(np.mean(ious)) if ious else 0.0,
    }


def detection_experiment(out):
    import psutil, threading
    from torchvision.models.detection import (
        ssdlite320_mobilenet_v3_large,
        SSDLite320_MobileNet_V3_Large_Weights,
    )

    images, targets = penn_data()
    images = images[50:60]
    targets = targets[50:60]
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT, score_thresh=0.01
    ).eval()
    rows = []
    predictions = {}
    process = psutil.Process()
    for size in [192, 256, 320, 384, 448]:
        model.transform.fixed_size = (size, size)
        with torch.inference_mode():
            model([images[0]])
        observed = []
        stop = threading.Event()

        def sample_rss():
            while not stop.wait(0.005):
                observed.append(process.memory_info().rss / 1024**2)

        th = threading.Thread(target=sample_rss, daemon=True)
        th.start()
        individual = []
        outputs = []
        times = []
        for image, target in zip(images, targets):
            start = time.perf_counter()
            with torch.inference_mode():
                pred = model([image])[0]
            times.append(time.perf_counter() - start)
            outputs.append(pred)
            individual.append(match_boxes(pred, target["boxes"]))
        stop.set()
        th.join()
        totals = {k: sum(x[k] for x in individual) for k in ["TP", "FP", "FN"]}
        precision = totals["TP"] / max(1, totals["TP"] + totals["FP"])
        recall = totals["TP"] / max(1, totals["TP"] + totals["FN"])
        rows.append(
            {
                "input_size": size,
                **totals,
                "precision": precision,
                "recall": recall,
                "mean_matched_IoU": np.mean([x["matched_IoU"] for x in individual]),
                "median_ms": np.median(times) * 1000,
                "observed_peak_RSS_MB": (
                    max(observed) if observed else process.memory_info().rss / 1024**2
                ),
            }
        )
        predictions[size] = outputs
    df = table(rows, out)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df.input_size, df.recall, "o-", label="Recall")
    ax[0].plot(df.input_size, df.precision, "s--", label="Precision")
    ax[0].set(xlabel="Размер входа, px", ylabel="Метрика", ylim=(0, 1))
    ax[0].legend()
    ax[1].plot(df.input_size, df.median_ms, "o-")
    ax[1].set(xlabel="Размер входа, px", ylabel="Медианное время, мс")
    showfig("resolution_quality_speed", out)
    fig, axes = plt.subplots(5, 2, figsize=(12, 18))
    import matplotlib.patches as patches

    for a, image, target, pred in zip(axes.ravel(), images, targets, predictions[320]):
        a.imshow(image.permute(1, 2, 0))
        a.set_title(target["name"], fontsize=10)
        a.axis("off")
        for box in target["boxes"]:
            x1, y1, x2, y2 = box.tolist()
            a.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="gold",
                    lw=1.5,
                    linestyle="--",
                )
            )
        keep = (pred["labels"] == 1) & (pred["scores"] >= 0.4)
        for box, score in zip(pred["boxes"][keep], pred["scores"][keep]):
            x1, y1, x2, y2 = box.tolist()
            a.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="deepskyblue",
                    lw=1.5,
                )
            )
            a.text(
                x1,
                y1,
                f"{score:.2f}",
                color="black",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 1},
            )
    showfig("ten_detection_predictions", out)
    coord = []
    for target, pred in zip(targets, predictions[320]):
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            if label == 1 and score >= 0.4:
                coord.append(
                    {
                        "image": target["name"],
                        "confidence": float(score),
                        **dict(zip(["x1", "y1", "x2", "y2"], box.tolist())),
                    }
                )
    table(coord, out, "coordinates_confidence")
    note(
        "Жёлтый пунктир — разметка, голубые рамки — SSD. FP — лишние рамки, FN — пропущенные люди, порог совпадения IoU=0.5. Mean matched IoU считается только по совпавшим рамкам. RSS включает весь процесс Python. SSDlite обучен на размере 320; при смене размера меняется также геометрия признаков и anchors. Ниже проверяю YOLO."
    )
    return {
        "model": model,
        "images": images,
        "targets": targets,
        "predictions": predictions,
        "rows": rows,
    }


def detection_thresholds(result, out):
    rows = []
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        metrics = [
            match_boxes(pred, t["boxes"], threshold)
            for pred, t in zip(result["predictions"][320], result["targets"])
        ]
        rows.append(
            {
                "threshold": threshold,
                **{k: sum(v[k] for v in metrics) for k in ["TP", "FP", "FN"]},
            }
        )
    table(rows, out, "confidence_thresholds")
    model = result["model"]
    model.transform.fixed_size = (320, 320)
    rows = []
    for threshold in [0.3, 0.5, 0.7]:
        model.nms_thresh = threshold
        with torch.inference_mode():
            pred = model([result["images"][0]])[0]
        rows.append(
            {
                "NMS_IoU_threshold": threshold,
                **match_boxes(pred, result["targets"][0]["boxes"]),
            }
        )
    table(rows, out, "nms_thresholds")
    model.nms_thresh = 0.55
    table(
        [
            {"image": t["name"], **match_boxes(p, t["boxes"])}
            for t, p in zip(result["targets"], result["predictions"][320])
        ],
        out,
        "errors_by_image",
    )
    note(
        "Высокий confidence-порог убирает слабые рамки и увеличивает риск пропусков. Сильное NMS может удалить рамки соседних людей. Ошибки зависят от перекрытия и размера объектов."
    )


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(24, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1)
        )

    def forward(self, x):
        skip = self.enc(x)
        low = self.mid(F.max_pool2d(skip, 2))
        up = F.interpolate(
            low, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.dec(torch.cat([skip, up], 1))


def segmentation_experiment(result, out):
    images, targets = penn_data()
    x = torch.stack(
        [
            F.interpolate(im[None], (96, 96), mode="bilinear", align_corners=False)[0]
            for im in images
        ]
    )
    y = torch.stack(
        [
            F.interpolate(t["mask"][None, None], (96, 96), mode="nearest")[0]
            for t in targets
        ]
    )
    train = (x[:40], y[:40])
    val = (x[40:50], y[40:50])
    test = (x[50:60], y[50:60])
    seed_all()
    m = TinyUNet()
    m, h, t = fit(
        m, train, val, epochs=60, batch=8, lr=0.003, task="regression", patience=12
    )
    # Для бинарной маски используем порог 0.5.
    m.eval()
    with torch.no_grad():
        pred = m(test[0]) > 0.5
    true = test[1] > 0.5
    rows = []
    for i in range(len(true)):
        inter = (pred[i] & true[i]).sum().item()
        union = (pred[i] | true[i]).sum().item()
        dice = 2 * inter / max(1, pred[i].sum().item() + true[i].sum().item())
        rows.append(
            {
                "image": targets[50 + i]["name"],
                "IoU": inter / max(1, union),
                "Dice": dice,
            }
        )
    df = table(rows, out, "segmentation_metrics")
    histories({"U-Net": h}, out, "segmentation_curves")
    fig, ax = plt.subplots(4, 3, figsize=(10, 11))
    for i in range(4):
        ax[i, 0].imshow(test[0][i].permute(1, 2, 0))
        ax[i, 1].imshow(true[i, 0], cmap="gray")
        ax[i, 2].imshow(pred[i, 0], cmap="gray")
        for a in ax[i]:
            a.axis("off")
    for a, title in zip(ax[0], ["Фото", "Эталон person", "U-Net, порог 0.5"]):
        a.set_title(title)
    showfig("segmentation_comparison", out)
    # Отдельно оцениваем границы маски.
    from scipy.ndimage import binary_erosion, binary_dilation

    boundary_rows = []
    for i in range(len(true)):
        gt = true[i, 0].numpy()
        pp = pred[i, 0].numpy()
        gb = gt ^ binary_erosion(gt)
        pb = pp ^ binary_erosion(pp)
        pr = (pb & binary_dilation(gb, iterations=2)).sum() / max(1, pb.sum())
        re = (gb & binary_dilation(pb, iterations=2)).sum() / max(1, gb.sum())
        boundary_rows.append(
            {
                "image": i,
                "boundary_precision_tolerance_2px": pr,
                "boundary_recall_tolerance_2px": re,
            }
        )
    table(boundary_rows, out, "boundary_metrics")
    torch.save(m.state_dict(), Path(out) / "unet.pt")
    note(
        f"Средние IoU={df.IoU.mean():.3f}, Dice={df.Dice.mean():.3f} на десяти test-изображениях. На разрешении 96×96 тонкие конечности и мелкие объекты теряются. Tiny U-Net использует skip-связь и обучается с MSE по бинарной маске; BCE/Dice-loss, увеличение данных и разрешения являются направлениями улучшения. Здесь оценивается семантическая маска класса person, экземпляры людей объединены."
    )
