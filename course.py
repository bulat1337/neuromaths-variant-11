from pathlib import Path
import copy, json, math, os, random, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Audio, Image as NotebookImage
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
DEVICE = torch.device("cpu")
torch.set_num_threads(4)
COLORS = ["#2463a2", "#bc6c25", "#8a7a24", "#965482", "#4b5563"]
plt.rcParams.update(
    {
        "figure.figsize": (8.5, 4.2),
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=COLORS),
        "figure.dpi": 115,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.18,
    }
)


def seed_all(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor(x, labels=False):
    return torch.as_tensor(np.asarray(x), dtype=torch.long if labels else torch.float32)


def showfig(name, out):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    fig = plt.gcf()
    fig.tight_layout()
    path = out / (name + ".png")
    fig.savefig(path)
    display(NotebookImage(filename=str(path)))
    plt.close(fig)


def table(rows, out=None, name="experiments"):
    df = pd.DataFrame(rows)
    if out is not None:
        Path(out).mkdir(exist_ok=True, parents=True)
        df.to_csv(Path(out) / (name + ".csv"), index=False)
    display(df.round(5))
    return df


def note(s):
    display(Markdown(s))


def split3(X, y, seed=11):
    ids = np.arange(len(y))
    tr, te = train_test_split(ids, test_size=0.2, stratify=y, random_state=seed)
    tr, va = train_test_split(
        tr, test_size=0.25, stratify=np.asarray(y)[tr], random_state=seed
    )
    return tuple(
        (tensor(np.asarray(X)[i]), tensor(np.asarray(y)[i], True)) for i in (tr, va, te)
    )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        widths=(32, 32),
        activation="relu",
        dropout=0.0,
        bn=False,
    ):
        super().__init__()
        layers = []
        d = input_dim
        acts = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
        }
        for w in widths:
            layers.append(nn.Linear(d, w))
            if bn:
                layers.append(nn.BatchNorm1d(w))
            layers.append(acts[activation]())
            if dropout:
                layers.append(nn.Dropout(dropout))
            d = w
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


class CNN(nn.Module):
    def __init__(
        self, bn=False, dropout=0.0, channels=8, padding=1, stride=1, pool="max"
    ):
        super().__init__()
        layers = []
        ci = 1
        for co in (channels, channels * 2):
            layers.append(nn.Conv2d(ci, co, 3, padding=padding, stride=stride))
            if bn:
                layers.append(nn.BatchNorm2d(co))
            layers += [nn.ReLU(), nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2)]
            ci = co
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(ci * 9, 10),
        )

    def forward(self, x):
        return self.head(self.features(x))


def fit(
    model,
    train,
    val,
    epochs=50,
    batch=64,
    lr=0.01,
    optimizer="Adam",
    weight_decay=0.0,
    seed=11,
    task="classification",
    patience=None,
    l1=0.0,
    augment=None,
):
    seed_all(seed)
    model.to(DEVICE)
    opts = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD,
        "Momentum": lambda p, **kw: torch.optim.SGD(p, momentum=0.9, **kw),
        "RMSprop": torch.optim.RMSprop,
    }
    opt = opts[optimizer](
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    lossfn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    loader = DataLoader(
        TensorDataset(*train),
        batch_size=batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    hist = []
    best = float("inf")
    state = copy.deepcopy(model.state_dict())
    stale = 0
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            if augment is not None:
                xb = augment(xb)
            opt.zero_grad()
            loss = lossfn(model(xb), yb)
            if l1:
                loss = loss + l1 * sum(p.abs().sum() for p in model.parameters())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            trp = model(train[0])
            vp = model(val[0])
            tl = lossfn(trp, train[1]).item()
            vl = lossfn(vp, val[1]).item()
            ta = (
                (trp.argmax(1) == train[1]).float().mean().item()
                if task == "classification"
                else np.nan
            )
            va = (
                (vp.argmax(1) == val[1]).float().mean().item()
                if task == "classification"
                else np.nan
            )
        hist.append(
            {
                "epoch": epoch + 1,
                "train_loss": tl,
                "val_loss": vl,
                "train_accuracy": ta,
                "val_accuracy": va,
            }
        )
        if vl < best - 1e-7:
            best = vl
            state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if patience and stale >= patience:
            break
    model.load_state_dict(state)
    model.eval()
    return model, pd.DataFrame(hist), time.perf_counter() - start


def classify(model, data):
    model.eval()
    with torch.no_grad():
        p = model(data[0]).softmax(-1).numpy()
    y = data[1].numpy()
    yh = p.argmax(1)
    return (
        {
            "accuracy": accuracy_score(y, yh),
            "macro_f1": f1_score(y, yh, average="macro", zero_division=0),
        },
        yh,
        p,
    )


def histories(curves, out, name="learning_curves", metric="loss"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, (label, h) in enumerate(curves.items()):
        ax[0].plot(
            h.epoch,
            h["train_" + metric],
            label=label,
            color=COLORS[i % 5],
            ls=["-", "--", ":"][i // 5 % 3],
        )
        ax[1].plot(
            h.epoch,
            h["val_" + metric],
            label=label,
            color=COLORS[i % 5],
            ls=["-", "--", ":"][i // 5 % 3],
        )
    for a, title in zip(ax, ["Обучение", "Валидация"]):
        a.set(xlabel="Эпоха", ylabel=metric, title=title)
        a.legend(fontsize=8)
    showfig(name, out)


def errors(model, data, out, n=10):
    metrics, pred, proba = classify(model, data)
    y = data[1].numpy()
    ids = np.flatnonzero(pred != y)[:n]
    if len(ids) < n:
        ids = np.argsort(proba.max(1))[:n]
        mode = "Наименее уверенные объекты (ошибок меньше заданного числа)"
    else:
        mode = "Ошибки классификации"
    sorted_p = np.sort(proba[ids], axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    comments = [
        (
            "Верный ответ с малым отрывом от альтернативы"
            if pred[i] == y[i]
            else (
                "Ошибка при близких вероятностях классов"
                if ma < 0.25
                else "Уверенная ошибка: проверить неоднозначность признаков и калибровку"
            )
        )
        for i, ma in zip(ids, margin)
    ]
    df = pd.DataFrame(
        {
            "index": ids,
            "true": y[ids],
            "predicted": pred[ids],
            "confidence": proba[ids].max(1),
            "probability_margin": margin,
            "analysis": comments,
        }
    )
    table(df, out, "errors")
    note(
        mode
        + ". Большая уверенность ошибочного ответа указывает на ограничение калибровки; близкие признаки классов усложняют границу."
    )
    return ids


def mnist():
    p = DATA / "mnist_subset.npz"
    if not p.exists():
        from torchvision.datasets import MNIST

        ds = MNIST(str(DATA / "MNIST"), train=True, download=True)
        X = ds.data.numpy().astype("float32")[:, None] / 255.0
        y = ds.targets.numpy()
        ids, _ = train_test_split(
            np.arange(len(y)), train_size=3000, stratify=y, random_state=11
        )
        ts = MNIST(str(DATA / "MNIST"), train=False, download=True)
        ti, _ = train_test_split(
            np.arange(len(ts)),
            train_size=600,
            stratify=ts.targets.numpy(),
            random_state=11,
        )
        tr, va = train_test_split(ids, test_size=0.2, stratify=y[ids], random_state=11)
        np.savez_compressed(
            p,
            Xtr=X[tr],
            ytr=y[tr],
            Xva=X[va],
            yva=y[va],
            Xte=ts.data.numpy()[ti, None].astype("float32") / 255.0,
            yte=ts.targets.numpy()[ti],
        )
    z = np.load(p)
    return tuple(
        (tensor(z["X" + s]), tensor(z["y" + s], True)) for s in ["tr", "va", "te"]
    )


def boundary(model, X, y, out, name="decision_boundary"):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 140),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 140),
    )
    model.eval()
    with torch.no_grad():
        zz = (
            model(tensor(np.c_[xx.ravel(), yy.ravel()]))
            .argmax(1)
            .numpy()
            .reshape(xx.shape)
        )
    plt.contourf(xx, yy, zz, alpha=0.2, cmap="cividis")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="cividis", s=9)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Граница решения")
    showfig(name, out)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
