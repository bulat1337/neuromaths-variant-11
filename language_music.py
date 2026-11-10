from course import *
import re, collections
from sequences import SequenceNet
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TextRNN(nn.Module):
    def __init__(self, vocab, kind="LSTM"):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 20, padding_idx=0)
        self.core = getattr(nn, kind)(20, 24, batch_first=True)
        self.head = nn.Linear(24, 2)

    def forward(self, x):
        # PAD не должен менять последнее состояние.
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        emb = self.embedding(x.long())
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.core(packed)
        if isinstance(h, tuple):
            h = h[0]
        return self.head(h[-1])


def text_general(out):
    rng = np.random.default_rng(11)
    positive = ["excellent", "great", "wonderful", "enjoyable", "brilliant"]
    negative = ["awful", "poor", "boring", "terrible", "dull"]
    topics = ["story", "movie", "acting", "music", "ending", "dialogue"]
    texts = []
    labels = []
    for i in range(600):
        label = i % 2
        word = rng.choice(positive if label else negative)
        topic = rng.choice(topics)
        texts.append(
            f"the {topic} was {word} and the movie felt {word} "
            + ("today" if i % 3 else "yesterday")
        )
        labels.append(label)
    ids = np.arange(len(labels))
    tr, te = train_test_split(ids, test_size=0.2, stratify=labels, random_state=11)
    tr, va = train_test_split(
        tr, test_size=0.25, stratify=np.array(labels)[tr], random_state=11
    )
    tokenize = lambda s: re.findall("[a-z]+", s.lower())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i in tr:
        for word in tokenize(texts[i]):
            vocab.setdefault(word, len(vocab))

    def encode(i):
        seq = [vocab.get(w, 1) for w in tokenize(texts[i])][:20]
        return seq + [0] * (20 - len(seq))

    XX = torch.tensor([encode(i) for i in ids])
    YY = torch.tensor(labels)
    pairs = [(XX[i], YY[i]) for i in [tr, va, te]]
    print(
        "Данные синтетические: настроение определяется выбранным словом. PAD=0, UNK=1, max_length=20."
    )
    print(texts[0], encode(0))
    print("Word tokens:", tokenize(texts[0]), "char tokens:", list(texts[0])[:20])
    rows = []
    models = {}
    curves = {}
    for kind in ["RNN", "LSTM", "GRU"]:
        seed_all()
        m = TextRNN(len(vocab), kind)
        m, h, t = fit(m, pairs[0], pairs[1], epochs=12, lr=0.005)
        metrics, _, _ = classify(m, pairs[1])
        rows.append({"architecture": kind, **metrics, "seconds": t})
        models[kind] = m
        curves[kind] = h
    table(rows, out, "text_architectures")
    histories(curves, out, "text_learning")
    metrics, pred, p = classify(models["LSTM"], pairs[2])
    ConfusionMatrixDisplay.from_predictions(YY[te], pred)
    plt.title("Синтетическая тональность, LSTM")
    showfig("text_confusion", out)
    table(
        [
            {
                "text": texts[i],
                "truth": int(YY[i]),
                "prediction": int(pred[j]),
                "length": len(tokenize(texts[i])),
            }
            for j, i in enumerate(te[:10])
        ],
        out,
        "ten_text_examples",
    )
    print("Test:", metrics)
    table(
        pd.DataFrame(
            classification_report(YY[te], pred, output_dict=True, zero_division=0)
        ).T.reset_index(names="class"),
        out,
        "text_per_class_metrics",
    )
    note(
        "В этом синтетическом корпусе класс задан ключевым словом. Высокая точность здесь показывает работу модели на простой задаче."
    )
    from gensim.models import Word2Vec, FastText

    corpus = [tokenize(texts[i]) for i in tr]
    from zlib import adler32

    stable_hash = lambda word: adler32(word.encode())
    w2v = Word2Vec(
        sentences=corpus,
        vector_size=20,
        window=3,
        min_count=1,
        workers=1,
        sg=1,
        epochs=40,
        seed=11,
        hashfxn=stable_hash,
    )
    fast = FastText(
        sentences=corpus,
        vector_size=20,
        window=3,
        min_count=1,
        workers=1,
        sg=1,
        epochs=40,
        seed=11,
        min_n=3,
        max_n=5,
        hashfxn=stable_hash,
    )
    rows = []
    for word in ["movie", "excellent", "boring"]:
        for name, model in [("Word2Vec", w2v), ("FastText", fast)]:
            rows.append(
                {
                    "model": name,
                    "word": word,
                    "neighbors": str(model.wv.most_similar(word, topn=3)),
                }
            )
    table(rows, out, "word_neighbors")
    print("Cosine movie/story:", w2v.wv.similarity("movie", "story"))
    print(
        "OOV excellentness in Word2Vec:",
        "excellentness" in w2v.wv.key_to_index,
        "FastText vector norm:",
        np.linalg.norm(fast.wv["excellentness"]),
    )
    analogy_rows = []
    for positive_words, negative_words, expected in [
        (["great", "awful"], ["excellent"], "negative adjective"),
        (["story", "excellent"], ["movie"], "positive adjective"),
        (["music", "boring"], ["movie"], "negative adjective"),
    ]:
        nearest = w2v.wv.most_similar(
            positive=positive_words, negative=negative_words, topn=3
        )
        answer = nearest[0][0]
        logical = answer in (negative if expected == "negative adjective" else positive)
        analogy_rows.append(
            {
                "query": "+".join(positive_words) + "−" + "−".join(negative_words),
                "expected_relation": expected,
                "answer": answer,
                "top3": str(nearest),
                "semantic_check": (
                    "согласуется с ожидаемой группой"
                    if logical
                    else "нелогичный перенос отношения"
                ),
            }
        )
    table(analogy_rows, out, "word_analogies")
    note(
        "Аналогии проверяю по заданным группам слов. На маленьком корпусе ближайшие соседи могут отражать сходный контекст без сохранения смысла аналогии."
    )
    words = list(w2v.wv.key_to_index)
    projection = PCA(2).fit_transform(np.array([w2v.wv[w] for w in words]))
    plt.scatter(projection[:, 0], projection[:, 1], s=15)
    for xy, word in zip(projection, words):
        plt.annotate(word, xy, fontsize=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Word2Vec: учебный корпус")
    showfig("word_vectors", out)
    note(
        "Word2Vec использует контексты слов, FastText — ещё и символьные n-граммы. Поэтому FastText строит вектор нового слова из знакомых частей. На маленьком корпусе рядом могут оказаться слова противоположной тональности."
    )


class MusicNet(nn.Module):
    def __init__(self, vocab, kind="LSTM", hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 20)
        self.core = getattr(nn, kind)(20, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x):
        sequence, _ = self.core(self.embedding(x.long()))
        return self.head(sequence[:, -1])


def music_generation(out):
    rng = np.random.default_rng(11)
    motifs = [
        [60, 64, 67, 72, 67, 64, 62, 60],
        [60, 62, 64, 67, 65, 64, 62, 60],
        [64, 67, 69, 67, 65, 64, 62, 60],
        [60, 67, 64, 69, 65, 67, 62, 60],
    ]
    melodies = []
    for i in range(160):
        notes = []
        for j in range(6):
            motif = motifs[int(rng.integers(4))]
            shift = int(rng.choice([0, 0, 0, 12, -12]))
            notes.extend([p + shift for p in motif])
        melodies.append(notes)
    pitches = sorted(set(p for notes in melodies[:100] for p in notes))
    vocab = {p: i for i, p in enumerate(pitches)}
    window = 12

    def dataset(mels):
        X = []
        y = []
        for notes in mels:
            seq = [vocab[p] for p in notes]
            for t in range(window, len(seq)):
                X.append(seq[t - window : t])
                y.append(seq[t])
        return torch.tensor(X), torch.tensor(y)

    tr = dataset(melodies[:100])
    va = dataset(melodies[100:130])
    te = dataset(melodies[130:])
    print(
        "Корпус процедурный, авторские учебные мотивы. Pitch vocabulary:",
        pitches,
        "число окон:",
        len(tr[1]),
        len(va[1]),
        len(te[1]),
    )
    rows = []
    models = {}
    curves = {}
    for kind, hidden in [
        ("RNN", 32),
        ("LSTM", 32),
        ("GRU", 32),
        ("LSTM", 16),
        ("LSTM", 48),
    ]:
        name = f"{kind}_{hidden}"
        seed_all()
        m = MusicNet(len(vocab), kind, hidden)
        m, h, t = fit(m, tr, va, epochs=12, batch=128, lr=0.005)
        metrics, _, _ = classify(m, va)
        rows.append(
            {
                "model": name,
                **metrics,
                "validation_perplexity": math.exp(h.val_loss.min()),
                "seconds": t,
                "parameters": count_params(m),
            }
        )
        models[name] = m
        curves[name] = h
    df = table(rows, out, "music_models")
    histories(curves, out, "music_learning")
    chosen = df.sort_values("validation_perplexity").iloc[0]["model"]
    m = models[chosen]
    test_metrics, test_pred, test_probabilities = classify(m, te)
    print("Выбрано по validation perplexity:", chosen, "test:", test_metrics)
    mistakes = np.flatnonzero(test_pred != te[1].numpy())[:10]
    table(
        [
            {
                "context": str([pitches[j] for j in te[0][i].tolist()]),
                "true_next": pitches[te[1][i].item()],
                "predicted_next": pitches[test_pred[i]],
                "confidence": float(test_probabilities[i].max()),
                "analysis": "После общего мотива выбрана другая ветвь продолжения; проверить неоднозначность переходов",
            }
            for i in mistakes
        ],
        out,
        "ten_music_errors",
    )
    # Переходы оцениваются только по train.
    transition = np.ones((len(vocab), len(vocab)))
    for melody in melodies[:100]:
        for a, b in zip(melody[:-1], melody[1:]):
            transition[vocab[a], vocab[b]] += 1
    markov_pred = transition[te[0][:, -1].numpy()].argmax(1)
    print("Markov baseline test accuracy:", accuracy_score(te[1], markov_pred))
    import wave, mido

    rows = []
    generations = []
    for i, (temperature, start_melody) in enumerate(
        [
            (0.5, 0),
            (1.0, 0),
            (1.5, 0),
            (1.0, 1),
            (1.0, 2),
            (0.8, 3),
            (1.2, 4),
            (0.7, 5),
            (1.0, 6),
            (1.3, 7),
        ]
    ):
        seed_all(11 + i)
        tokens = [vocab[p] for p in melodies[start_melody][:window]]
        for step in range(96):
            with torch.no_grad():
                logits = m(torch.tensor([tokens[-window:]]))[0] / temperature
            tokens.append(int(torch.multinomial(logits.softmax(0), 1)))
        notes = [pitches[j] for j in tokens[window:]]
        grams = [tuple(notes[j : j + 4]) for j in range(len(notes) - 3)]
        repeat = 1 - len(set(grams)) / len(grams)
        repeated = np.mean(np.diff(notes) == 0)
        rows.append(
            {
                "temperature": temperature,
                "initial_melody": start_melody,
                "unique_pitches": len(set(notes)),
                "repeated_4gram_fraction": repeat,
                "consecutive_repeat_fraction": repeated,
            }
        )
        name = f"generated_{i}_temperature_{temperature}"
        (Path(out) / (name + ".json")).write_text(json.dumps(notes))
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=500000))
        for pitch in notes:
            track.append(mido.Message("note_on", note=pitch, velocity=70, time=0))
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=120))
        mid.save(Path(out) / (name + ".mid"))
        rate = 16000
        t = np.arange(int(rate * 0.125)) / rate
        audio = []
        for pitch in notes:
            freq = 440 * 2 ** ((pitch - 69) / 12)
            envelope = np.minimum(t / 0.01, 1) * np.minimum((0.125 - t) / 0.02, 1)
            audio.append(0.3 * np.sin(2 * np.pi * freq * t) * np.maximum(envelope, 0))
        audio = np.concatenate(audio)
        filename = Path(out) / (name + ".wav")
        with wave.open(str(filename), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes((audio * 32767).astype("<i2").tobytes())
        if i < 3:
            display(Audio(filename=str(filename)))
        generations.append(notes)
    gd = table(rows, out, "generation_structure")
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for a, notes, temp in zip(ax, generations, [0.5, 1.0, 1.5]):
        a.step(np.arange(len(notes)), notes, where="post")
        a.set(ylabel="MIDI pitch", title=f"Temperature={temp}")
    ax[-1].set_xlabel("Номер события")
    showfig("generated_note_sequences", out)
    pd.DataFrame(
        {
            "example": np.arange(10),
            "context": [str([pitches[j] for j in row.tolist()]) for row in te[0][:10]],
            "true_next": [pitches[j] for j in te[1][:10].tolist()],
        }
    ).to_csv(Path(out) / "ten_generation_examples.csv", index=False)
    note(
        "Длительность ноты — 1/8 при 120 BPM. При обучении использую истинную историю, при генерации — прогнозы модели. Низкая temperature усиливает повторы, высокая увеличивает случайность. Десять продолжений сохраняются в MIDI и WAV. Корпус состоит из учебных мотивов."
    )
    torch.save(m.state_dict(), Path(out) / "music_model.pt")


def news_dataset():
    path = DATA / "newsgroups.json"
    if not path.exists():
        from sklearn.datasets import fetch_20newsgroups

        n = fetch_20newsgroups(
            subset="all",
            categories=[
                "sci.space",
                "rec.autos",
                "rec.sport.baseball",
                "comp.graphics",
            ],
            remove=("headers", "footers", "quotes"),
            data_home=str(DATA / "newsgroups_cache"),
        )
        pd.DataFrame({"text": n.data, "target": n.target}).to_json(
            path, orient="records"
        )
    df = pd.read_json(path)
    df = df[df.text.str.len() > 60].drop_duplicates("text")
    ids, _ = train_test_split(
        np.arange(len(df)), train_size=1000, stratify=df.target, random_state=11
    )
    df = df.iloc[ids].reset_index(drop=True)
    tr, te = train_test_split(
        np.arange(1000), test_size=0.2, stratify=df.target, random_state=11
    )
    tr, va = train_test_split(
        tr, test_size=0.25, stratify=df.target.iloc[tr], random_state=11
    )
    return df, tr, va, te


class BertClassifier(nn.Module):
    def __init__(self, encoder, classes=4):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, classes)

    def forward(self, x):
        ids = x[:, 0].long()
        mask = x[:, 1].long()
        encoded = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = (encoded * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp(
            min=1
        )
        return self.head(pooled)


def ensure_bert_assets():
    path = DATA / "bert_tiny"
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    if all((path / filename).is_file() for filename in required):
        return path

    import shutil
    import tempfile
    from huggingface_hub import hf_hub_download
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    model_id = "prajjwal1/bert-tiny"
    vocab = hf_hub_download(model_id, "vocab.txt")
    config_file = hf_hub_download(model_id, "config.json")
    weights_file = hf_hub_download(model_id, "pytorch_model.bin")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=BertWordPieceTokenizer(vocab, lowercase=True),
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    config = BertConfig(**json.loads(Path(config_file).read_text()))
    config._attn_implementation = "eager"
    model = BertModel(config)
    state = torch.load(weights_file, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("bert."): value
        for key, value in state.items()
        if not key.startswith("cls.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [
        key for key in missing if not key.endswith(("position_ids", "token_type_ids"))
    ]
    unexpected = [
        key
        for key in unexpected
        if not key.endswith(("position_ids", "token_type_ids"))
    ]
    if missing or unexpected:
        raise RuntimeError(f"Несовпадение параметров BERT: {missing=}, {unexpected=}")
    with tempfile.TemporaryDirectory(prefix=".bert-tiny-", dir=DATA) as temporary:
        temporary = Path(temporary)
        tokenizer.save_pretrained(temporary)
        model.save_pretrained(temporary)
        path.mkdir(exist_ok=True)
        for file in temporary.iterdir():
            shutil.copy2(file, path / file.name)
    return path


def bert_tokenization(out):
    from transformers import AutoTokenizer, BertModel, BertConfig

    path = ensure_bert_assets()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    pretrained = BertModel.from_pretrained(
        path, local_files_only=True, attn_implementation="eager"
    )
    df, tr, va, te = news_dataset()
    texts = df.text.tolist()
    y = df.target.to_numpy()
    words = lambda s: re.findall(r"[a-z]+(?:'[a-z]+)?", s.lower())
    counts = collections.Counter(w for i in tr for w in words(texts[i]))
    vocab = {"[PAD]": 0, "[UNK]": 1}
    vocab.update({w: i + 2 for i, (w, c) in enumerate(counts.most_common(6000))})
    maxlen = 64

    def word_encode(s):
        tokens = [vocab.get(w, 1) for w in words(s)][:maxlen]
        n = len(tokens)
        return [tokens + [0] * (maxlen - n), [1] * n + [0] * (maxlen - n)]

    WX = torch.tensor([word_encode(s) for s in texts])
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=maxlen,
        return_tensors="pt",
    )
    SX = torch.stack([enc["input_ids"], enc["attention_mask"]], 1)
    Y = torch.tensor(y)
    print(
        "Источник: 20 Newsgroups, четыре темы, заголовки/подписи/цитаты удалены; 600/200/200. Word vocabulary обучен только на train. BERT: prajjwal1/bert-tiny (2 слоя, hidden=128)."
    )
    print("Word tokens:", words(texts[te[0]])[:20])
    print("WordPiece:", tokenizer.tokenize(texts[te[0]])[:30])
    stats = []
    for name, x in [("word", WX), ("subword", SX)]:
        tokenids = x[te, 0]
        mask = x[te, 1]
        unk = 1 if name == "word" else tokenizer.unk_token_id
        stats.append(
            {
                "tokenizer": name,
                "mean_tokens_untruncated": np.mean(
                    [
                        (
                            len(words(texts[i]))
                            if name == "word"
                            else len(tokenizer.tokenize(texts[i]))
                        )
                        for i in te
                    ]
                ),
                "UNK_token_fraction": float(
                    ((tokenids == unk) & mask.bool()).sum() / mask.sum()
                ),
                "mean_kept_tokens": float(mask.sum(1).float().mean()),
                "vocab_size": len(vocab) if name == "word" else len(tokenizer),
            }
        )
    table(stats, out, "tokenization_statistics")
    models = {}
    rows = []
    curves = {}
    test_predictions = {}
    vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
    xt = vectorizer.fit_transform([texts[i] for i in tr])
    baseline = LogisticRegression(max_iter=1000).fit(xt, y[tr])
    vp = baseline.predict(vectorizer.transform([texts[i] for i in va]))
    tp = baseline.predict(vectorizer.transform([texts[i] for i in te]))
    test_predictions["TFIDF_LogReg"] = tp
    rows.append(
        {
            "model": "TFIDF_LogReg",
            "validation_accuracy": accuracy_score(y[va], vp),
            "validation_macro_f1": f1_score(y[va], vp, average="macro"),
            "trainable_parameters": baseline.coef_.size,
            "seconds": np.nan,
        }
    )
    for name, x, size in [
        ("word_scratch", WX, len(vocab)),
        ("subword_scratch", SX, len(tokenizer)),
    ]:
        seed_all()
        config = BertConfig(
            vocab_size=size,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=128,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        m = BertClassifier(BertModel(config, add_pooling_layer=False))
        m, h, t = fit(m, (x[tr], Y[tr]), (x[va], Y[va]), epochs=5, batch=32, lr=0.001)
        metrics, pred, p = classify(m, (x[va], Y[va]))
        rows.append(
            {
                "model": name,
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "trainable_parameters": count_params(m),
                "seconds": t,
            }
        )
        curves[name] = h
        models[name] = (m, x)
        test_predictions[name] = classify(m, (x[te], Y[te]))[1]
    pretrained.eval()
    with torch.no_grad():
        embeddings = []
        for part in SX.split(32):
            mask = part[:, 1]
            e = pretrained(input_ids=part[:, 0], attention_mask=mask).last_hidden_state
            embeddings.append((e * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True))
        embeddings = torch.cat(embeddings)
    seed_all()
    head = MLP(128, 4, ())
    head, h, t = fit(
        head, (embeddings[tr], Y[tr]), (embeddings[va], Y[va]), epochs=60, lr=0.01
    )
    metrics, _, _ = classify(head, (embeddings[va], Y[va]))
    rows.append(
        {
            "model": "BERT_frozen",
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "trainable_parameters": count_params(head),
            "seconds": t,
        }
    )
    curves["BERT_frozen"] = h
    test_predictions["BERT_frozen"] = classify(head, (embeddings[te], Y[te]))[1]
    seed_all()
    finetuned = BertClassifier(copy.deepcopy(pretrained))
    finetuned, h, t = fit(
        finetuned, (SX[tr], Y[tr]), (SX[va], Y[va]), epochs=4, batch=32, lr=0.0003
    )
    metrics, _, _ = classify(finetuned, (SX[va], Y[va]))
    rows.append(
        {
            "model": "BERT_finetuned",
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "trainable_parameters": count_params(finetuned),
            "seconds": t,
        }
    )
    curves["BERT_finetuned"] = h
    test_predictions["BERT_finetuned"] = classify(finetuned, (SX[te], Y[te]))[1]
    table(
        pd.DataFrame(
            classification_report(
                y[te],
                test_predictions["BERT_finetuned"],
                output_dict=True,
                zero_division=0,
            )
        ).T.reset_index(names="class"),
        out,
        "bert_per_class_metrics",
    )
    comparison = table(rows, out)
    histories(
        {k: v for k, v in curves.items() if k != "BERT_frozen"}, out, "bert_learning"
    )
    evaluation = []
    for name, pred in test_predictions.items():
        evaluation.append(
            {
                "model": name,
                "test_accuracy": accuracy_score(y[te], pred),
                "test_macro_f1": f1_score(y[te], pred, average="macro"),
            }
        )
    ed = table(evaluation, out, "final_test_comparison")
    ed.plot.bar(x="model", y="test_macro_f1", legend=False)
    plt.ylabel("Macro-F1")
    plt.xlabel("Модель")
    plt.ylim(0, 1)
    plt.title("Заранее заданные сравнения, test")
    showfig("tokenizer_model_quality", out)
    # Сравнение токенизаторов при одинаковых размерах encoder.
    errorrows = []
    strata = []
    for name in ["word_scratch", "subword_scratch", "BERT_finetuned"]:
        pred = test_predictions[name]
        for j, i in enumerate(te):
            ww = words(texts[i])
            rare = sum(counts[w] <= 2 for w in ww)
            errorrows.append(
                {
                    "model": name,
                    "row": int(i),
                    "true": int(y[i]),
                    "predicted": int(pred[j]),
                    "correct": int(pred[j] == y[i]),
                    "words": len(ww),
                    "rare_words": rare,
                    "text": texts[i][:200],
                }
            )
        for label, mask in [
            ("short", np.array([len(words(texts[i])) <= 80 for i in te])),
            ("long", np.array([len(words(texts[i])) > 80 for i in te])),
            (
                "rare_words_ge10",
                np.array(
                    [sum(counts[w] <= 2 for w in words(texts[i])) >= 10 for i in te]
                ),
            ),
        ]:
            strata.append(
                {
                    "model": name,
                    "subset": label,
                    "n": int(mask.sum()),
                    "accuracy": (
                        accuracy_score(y[te][mask], pred[mask])
                        if mask.any()
                        else np.nan
                    ),
                }
            )
    errorsdf = pd.DataFrame(errorrows)
    errorsdf.to_csv(Path(out) / "all_tokenization_errors.csv", index=False)
    display(errorsdf[errorsdf.correct == 0].head(10))
    table(strata, out, "length_rare_word_quality")
    example = tokenizer(
        "An astronaut explores unfamiliar interplanetary landscapes.",
        return_tensors="pt",
    )
    pretrained.config.output_attentions = True
    with torch.no_grad():
        output = pretrained(**example, output_attentions=True)
    attention = output.attentions[0][0, 0].numpy()
    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"][0])
    plt.figure(figsize=(9, 7))
    plt.imshow(attention, cmap="Blues")
    plt.xticks(range(len(tokens)), tokens, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(len(tokens)), tokens, fontsize=8)
    plt.title("BERT tiny: attention, слой 1, голова 1")
    plt.colorbar()
    showfig("bert_attention", out)
    q = tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    weights = (q @ q.T / math.sqrt(2)).softmax(-1)
    context = weights @ q
    assert torch.allclose(weights.sum(1), torch.ones(3))
    print("Self-attention weights:\n", weights, "\ncontext:\n", context)
    plt.imshow(weights, cmap="Blues")
    plt.xlabel("Ключ")
    plt.ylabel("Запрос")
    plt.title("Игрушечный self-attention")
    showfig("toy_attention", out)
    note(
        "Word-level даёт UNK для новых слов. WordPiece разбивает слова на части и обычно удлиняет последовательность. Для сравнения токенизации использую word_scratch и subword_scratch с одинаковыми размерами encoder. Размеры embedding-таблиц различаются. У предобученного BERT также меняются исходные веса. Attention-карта показывает веса одной головы и сама по себе не объясняет причину ответа."
    )
    note(
        "Позиционные embeddings задают порядок, residual-связи помогают передавать градиент, LayerNorm нормализует признаки токена. При mean pooling исключаю PAD через attention_mask. Все пять моделей заданы до оценки test; бюджет обучения небольшой."
    )
    torch.save(finetuned.state_dict(), Path(out) / "bert_classifier.pt")
