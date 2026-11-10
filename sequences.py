from course import *
from sklearn.linear_model import Ridge


class SequenceNet(nn.Module):
    def __init__(self, kind="RNN", features=1, hidden=24, outputs=1):
        super().__init__()
        self.kind = kind
        self.core = getattr(nn, kind)(features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        sequence, hidden = self.core(x)
        return self.head(sequence[:, -1])

    def hidden_states(self, x):
        return self.core(x)[0]


def synthetic_series(n=2400):
    t = np.arange(n)
    rng = np.random.default_rng(11)
    return (
        np.sin(0.04 * t) + 0.35 * np.sin(0.13 * t) + 0.06 * rng.normal(size=n)
    ).astype("float32")


def imputed_windows(series, missing, window=30, method="mean"):
    n = len(series)
    cut1 = int(n * 0.6)
    cut2 = int(n * 0.8)
    observed = series[:cut1][~missing[:cut1]]
    mean = observed.mean()
    std = observed.std()
    X = []
    Y = []
    ends = []
    for end in range(window, n):
        values = series[end - window : end].copy()
        mask = missing[end - window : end]
        known = np.flatnonzero(~mask)
        if method in ["linear", "linear_mask"]:
            values = (
                np.interp(np.arange(window), known, values[known])
                if len(known)
                else np.full(window, mean)
            )
        elif method == "ffill":
            previous = mean
            for i in range(window):
                if mask[i]:
                    values[i] = previous
                else:
                    previous = values[i]
        else:
            values[mask] = mean
        values = (values - mean) / std
        xx = (
            np.c_[values, mask.astype("float32")]
            if method in ["mean_mask", "linear_mask"]
            else values[:, None]
        )
        X.append(xx)
        Y.append([(series[end] - mean) / std])
        ends.append(end)
    X = tensor(X)
    Y = tensor(Y)
    ends = np.array(ends)
    pairs = tuple(
        (X[ids], Y[ids])
        for ids in [ends < cut1, (ends >= cut1) & (ends < cut2), ends >= cut2]
    )
    return pairs, {
        "mean": float(mean),
        "std": float(std),
        "window": window,
        "cut1": cut1,
        "cut2": cut2,
        "test_ends": ends[ends >= cut2],
    }


def regression_metrics(model, pair, std=1.0):
    model.eval()
    with torch.no_grad():
        pred = model(pair[0]).numpy().ravel() * std
    truth = pair[1].numpy().ravel() * std
    return (
        {
            "MAE": mean_absolute_error(truth, pred),
            "RMSE": math.sqrt(mean_squared_error(truth, pred)),
        },
        pred,
        truth,
    )


def missing_experiment(out):
    series = synthetic_series()
    missing = np.random.default_rng(11).random(len(series)) < 0.3
    rows = []
    curves = {}
    models = {}
    pairs = {}
    meta = None
    for method in ["mean", "ffill", "linear", "mean_mask", "linear_mask"]:
        (tr, va, te), meta = imputed_windows(series, missing, method=method)
        seed_all()
        m = SequenceNet(features=tr[0].shape[-1])
        m, h, t = fit(m, tr, va, epochs=20, batch=64, lr=0.003, task="regression")
        metrics, _, _ = regression_metrics(m, va, meta["std"])
        rows.append(
            {"method": method, **metrics, "seconds": t, "parameters": count_params(m)}
        )
        curves[method] = h
        models[method] = m
        pairs[method] = (tr, va, te)
    df = table(rows, out)
    histories(curves, out, "missing_learning")
    best = df.sort_values("RMSE").iloc[0]["method"]
    tr, va, te = pairs[best]
    m = models[best]
    metrics, pred, truth = regression_metrics(m, te, meta["std"])
    print("Выбор по validation:", best, "test:", metrics)
    last = te[0][:, -1, 0].numpy() * meta["std"]
    avg = te[0][:, :, 0].mean(1).numpy() * meta["std"]
    table(
        [
            {"model": "RNN", **metrics},
            {
                "model": "last_observed_or_imputed",
                "MAE": mean_absolute_error(truth, last),
                "RMSE": math.sqrt(mean_squared_error(truth, last)),
            },
            {
                "model": "window_mean",
                "MAE": mean_absolute_error(truth, avg),
                "RMSE": math.sqrt(mean_squared_error(truth, avg)),
            },
        ],
        out,
        "baselines",
    )
    plt.plot(truth[:180] + meta["mean"], label="Истинный ряд")
    plt.plot(pred[:180] + meta["mean"], label="RNN")
    plt.plot(last[:180] + meta["mean"], ls=":", label="Последнее заполненное значение")
    plt.xlabel("Шаг test")
    plt.ylabel("Значение ряда")
    plt.title("Одношаговый прогноз, 30% пропусков")
    plt.legend()
    showfig("forecast", out)
    plt.plot(np.abs(pred - truth))
    plt.xlabel("Шаг test")
    plt.ylabel("Абсолютная ошибка")
    plt.title("Ошибка по времени")
    showfig("error_over_time", out)
    ids = np.argsort(np.abs(pred - truth))[-10:]
    table(
        [
            {
                "test_step": int(i),
                "truth": truth[i] + meta["mean"],
                "prediction": pred[i] + meta["mean"],
                "abs_error": abs(pred[i] - truth[i]),
            }
            for i in ids
        ],
        out,
        "ten_errors",
    )
    df.plot.bar(x="method", y="RMSE", legend=False)
    plt.ylabel("Validation RMSE")
    plt.xlabel("Метод заполнения")
    plt.title("Сравнение обработки пропусков")
    showfig("imputation_quality", out)
    note(
        f"Лучший из заранее выбранных методов по validation: {best}. Интерполяция выполняется отдельно внутри каждого прошлого окна; значения после момента прогноза недоступны. Среднее и стандартное отклонение оценены по наблюдаемым train-значениям. Признак пропуска различает настоящее среднее и искусственное заполнение. Результат относится к случайным пропускам MCAR с долей 30%."
    )
    return {
        "series": series,
        "missing": missing,
        "best": best,
        "model": m,
        "pairs": pairs,
        "meta": meta,
    }


def sequence_general(result, out):
    series = result["series"]
    missing = result["missing"]
    rows = []
    for window, hidden in [(10, 24), (30, 24), (60, 24), (30, 8), (30, 48)]:
        (tr, va, te), meta = imputed_windows(
            series, missing, window=window, method="linear"
        )
        seed_all()
        m = SequenceNet(hidden=hidden)
        m, h, t = fit(m, tr, va, epochs=12, lr=0.003, task="regression")
        metrics, _, _ = regression_metrics(m, va, meta["std"])
        rows.append(
            {
                "window": window,
                "hidden": hidden,
                **metrics,
                "parameters": count_params(m),
            }
        )
    table(rows, out, "window_hidden")
    m = result["model"]
    tr, va, te = result["pairs"][result["best"]]
    meta = result["meta"]
    hidden = m.hidden_states(te[0][:1]).detach().numpy()[0]
    plt.imshow(hidden.T, aspect="auto", cmap="viridis")
    plt.xlabel("Шаг внутри окна")
    plt.ylabel("Компонента hidden")
    plt.title("Скрытые состояния RNN")
    showfig("hidden_state", out)
    rows = []
    for horizon in [1, 5, 10, 20]:
        forecasts = []
        truth = []
        for i in range(0, len(te[0]) - horizon, 12):
            window = te[0][i : i + 1].clone()
            with torch.no_grad():
                for step in range(horizon):
                    p = m(window)
                    new = torch.zeros(1, 1, window.shape[-1])
                    new[0, 0, 0] = p[0, 0]
                    if new.shape[-1] > 1:
                        new[0, 0, 1] = 1.0
                    window = torch.cat([window[:, 1:], new], 1)
            forecasts.append(p.item() * meta["std"])
            truth.append(te[1][i + horizon - 1].item() * meta["std"])
        rows.append(
            {
                "horizon": horizon,
                "RMSE": math.sqrt(mean_squared_error(truth, forecasts)),
            }
        )
    table(rows, out, "recursive_horizons")
    plt.plot([r["horizon"] for r in rows], [r["RMSE"] for r in rows], "o-")
    plt.xlabel("Горизонт")
    plt.ylabel("RMSE")
    plt.title("Накопление ошибки рекурсивного прогноза")
    showfig("horizon_error", out)
    # Прошлые выходы и известный внешний сигнал.
    n = len(series)
    ex = np.sin(0.02 * np.arange(n))
    XX = np.array([np.r_[series[i - 10 : i], ex[i - 10 : i]] for i in range(10, n)])
    yy = series[10:]
    cut1 = int(n * 0.6) - 10
    cut2 = int(n * 0.8) - 10
    narx = Ridge(alpha=0.1).fit(XX[:cut1], yy[:cut1])
    pr = narx.predict(XX[cut2:])
    print(
        "NARX-like open-loop test RMSE:", math.sqrt(mean_squared_error(yy[cut2:], pr))
    )
    state = list(series[int(n * 0.8) - 10 : int(n * 0.8)])
    closed = []
    for end in range(int(n * 0.8), int(n * 0.8) + 100):
        p = float(narx.predict(np.r_[state[-10:], ex[end - 10 : end]][None])[0])
        closed.append(p)
        state.append(p)
    print(
        "NARX-like closed-loop first 100 RMSE:",
        math.sqrt(
            mean_squared_error(series[int(n * 0.8) : int(n * 0.8) + 100], closed)
        ),
    )
    # Прогноз на каждом шаге последовательности.
    seq = SequenceNet()
    sample = tensor(series[:31])[None, :, None]
    states = seq.hidden_states(sample[:, :-1])
    per_step = seq.head(states)
    print("many-to-one:", seq(sample[:, :-1]).shape, "many-to-many:", per_step.shape)
    opt = torch.optim.Adam(seq.parameters(), lr=0.01)
    for _ in range(80):
        opt.zero_grad()
        loss = nn.functional.mse_loss(
            seq.head(seq.hidden_states(sample[:, :-1])), sample[:, 1:]
        )
        loss.backward()
        opt.step()
    print("Toy many-to-many MSE:", float(loss.detach()))
    note(
        "В RNN контекст хранится в скрытом состоянии, в NARX-подобной модели задержанные входы и выходы подаются явно. Open-loop использует наблюдаемые выходы, closed-loop — собственные прогнозы. На длинном горизонте ошибки накапливаются."
    )


def weather_series():
    path = DATA / "weather.csv"
    if not path.exists():
        import urllib.request, zipfile

        p = DATA / "jena_climate.zip"
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
            p,
        )
        with zipfile.ZipFile(p) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, nrows=18000)
        df.iloc[::6][["Date Time", "T (degC)"]].to_csv(path, index=False)
    return pd.read_csv(path)["T (degC)"].to_numpy(dtype="float32")


def gated_noise(out):
    values = weather_series()
    missing = np.zeros(len(values), dtype=bool)
    (tr, va, te), meta = imputed_windows(values, missing, window=24)
    models = {}
    rows = []
    curves = {}
    print(
        "Источник: Jena climate, первые 3000 часовых наблюдений; окна 24 часа.",
        "Train/val/test:",
        len(tr[1]),
        len(va[1]),
        len(te[1]),
    )
    for kind, seed in [
        ("RNN", 11),
        ("LSTM", 11),
        ("GRU", 11),
        ("LSTM", 12),
        ("GRU", 12),
    ]:
        seed_all(seed)
        m = SequenceNet(kind, hidden=24)
        m, h, t = fit(m, tr, va, epochs=20, lr=0.003, task="regression", seed=seed)
        name = f"{kind}_{seed}"
        models[name] = m
        curves[name] = h
        metrics, _, _ = regression_metrics(m, va, meta["std"])
        rows.append(
            {"model": name, **metrics, "seconds": t, "parameters": count_params(m)}
        )
    table(rows, out, "training_runs")
    histories(curves, out, "gated_curves")
    rows = []
    rng = torch.Generator().manual_seed(11)
    perturbations = {"clean": te[0]}
    for sigma in [0.1, 0.3, 0.6]:
        perturbations[f"noise_{sigma}"] = te[0] + sigma * torch.randn(
            te[0].shape, generator=rng
        )
    for rate in [0.02, 0.05]:
        xx = te[0].clone()
        mask = torch.rand(xx.shape, generator=rng) < rate
        xx[mask] += 4 * torch.randn(xx[mask].shape, generator=rng)
        perturbations[f"outliers_{rate}"] = xx
    for name, m in models.items():
        for condition, x in perturbations.items():
            metrics, pred, truth = regression_metrics(m, (x, te[1]), meta["std"])
            rows.append({"model": name, "condition": condition, **metrics})
    df = table(rows, out, "noise_robustness")
    df.pivot(index="condition", columns="model", values="RMSE").plot.bar(
        figsize=(11, 4)
    )
    plt.ylabel("RMSE, °C")
    plt.xlabel("Условие test-входа")
    plt.title("LSTM/GRU: шум и выбросы")
    showfig("gated_noise_quality", out)
    for name in ["RNN_11", "LSTM_11", "GRU_11"]:
        metrics, p, t = regression_metrics(models[name], te, meta["std"])
        plt.plot(p[:120] + meta["mean"], label=name)
    plt.plot(t[:120] + meta["mean"], "k--", label="Наблюдения")
    plt.xlabel("Час test")
    plt.ylabel("Температура, °C")
    plt.legend()
    plt.title("Прогноз температуры")
    showfig("weather_forecast", out)
    last = te[0][:, -1, 0].numpy() * meta["std"]
    print(
        "Наивный прогноз test RMSE, °C:",
        math.sqrt(mean_squared_error(te[1].numpy().ravel() * meta["std"], last)),
    )
    m = models["LSTM_11"]
    metrics, p, t = regression_metrics(m, te, meta["std"])
    ids = np.argsort(abs(p - t))[-10:]
    table(
        [
            {
                "test_hour": int(i),
                "temperature": t[i] + meta["mean"],
                "prediction": p[i] + meta["mean"],
                "absolute_error_C": abs(p[i] - t[i]),
            }
            for i in ids
        ],
        out,
        "ten_errors",
    )
    note(
        "Шум добавляю только во входные окна, целевую температуру сохраняю. σ задана в единицах train standard deviation. У LSTM отдельная ячейка памяти, у GRU меньше параметров. Качество сравниваю по таблице."
    )
    return {"models": models, "train": tr, "val": va, "test": te, "meta": meta}
