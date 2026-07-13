"""TextFeaturizer — features de texto de AMBIVALÊNCIA/HESITÂNCIA (BAH).

Motor reutilizável (léxico/regex + VADER + classificador de emoção) que produz um bloco
de features hand-crafted de suporte ao embedding de texto, alinhado ao construto do BAH.
Espelha o contrato do :class:`~src.features.hesitation.HesitationExtractor`.

Granularidade por grupo (verificado nos dados: 1 vídeo = 1 resposta; ~24 palavras/janela
vs ~117–151/vídeo; ~9 janelas/vídeo):

- **A1 — RESPOSTA (broadcast).** Índices psicométricos de ambivalência sobre o transcript do
  vídeo inteiro. P e N (intensidade positiva/negativa) de DUAS fontes: léxico (VADER) e emoção
  (RoBERTa-emotion, probs agrupadas em polos). Índices: intensidade `(P+N)/2`, inconsistência
  `|P−N|`, Kaplan `min(P,N)`, Griffin `(P+N)/2−|P−N|`. **Nunca polaridade líquida.** Ambivalência
  é propriedade da resposta — por janela (~24 palavras) o léxico é esparso e os polos opostos caem
  em janelas diferentes. Fontes: Thompson/Zanna/Griffin 1995; Kaplan 1972; Conner et al. 2021.
- **A3 — JANELA + RESPOSTA.** Por janela: entropia da distribuição de emoção + gap entre as 2 top
  emoções (mistura local). Por resposta (broadcast): flutuação = std/amplitude da valência de
  emoção entre as janelas ("oscila entre estados") + conflito de polos. Fontes: MEDA; HSEmotion.
- **A4 — JANELA.** Contraste (but/however/...) + shift de polaridade intra-janela (VADER na 1ª vs
  2ª metade). Fonte: stance≠sentimento (Springer 2025) — núcleo barato.
- **H1 — JANELA.** Hedges por subcategoria (1ª pessoa epistêmica, advérbios, aproximadores,
  incerteza explícita, atribuição sem-agente) + opcional score contextual por supervisão distante
  (Ganter & Strube 2009; lição do CoNLL-2010: não disparar cego numa palavra-gatilho).

Contrato: ``extract(texts, video_ids) -> (n_windows, dim)`` (os ``video_id`` agrupam as janelas
p/ o broadcast de A1/A3-resposta). ``feature_names`` canônico; vetor montado por lookup (à prova
de desalinhamento). H2 (fillers) foi descartada: o Whisper descarta ~96% dos fillers.
"""

from __future__ import annotations

import re

import numpy as np

from src.conf import resolve_device
from src.logger import get_logger

log = get_logger("features.text_features")

_WORD_RE = re.compile(r"\b[\w']+\b")
_SENT_SPLIT = re.compile(r"[.!?]+")

# =============================================================================
# Léxicos (constantes editáveis) — inglês (dataset BAH é EN)
# =============================================================================

HEDGE_LEXICON: dict[str, list[str]] = {
    # 1ª pessoa epistêmica (hesitar por autodúvida)
    "fp": ["i think", "i guess", "i believe", "i feel", "i suppose", "i'd say",
           "i would say", "i mean", "i assume"],
    # advérbios epistêmicos
    "adv": ["maybe", "perhaps", "probably", "possibly", "likely", "presumably",
            "apparently", "seemingly"],
    # aproximadores / vagueza
    "approx": ["sort of", "kind of", "kinda", "somewhat", "a little", "a bit",
               "more or less", "or so", "roughly", "somehow"],
    # incerteza explícita
    "unc": ["not sure", "i'm not sure", "i am not sure", "i don't know", "i dont know",
            "dunno", "not really sure", "hard to say", "who knows", "no idea"],
    # atribuição sem-agente (hesitar por deflexão)
    "agentless": ["some people say", "it is said", "it is believed", "they say",
                  "it seems", "it appears", "some say", "people say"],
}
# Pistas de ALTA PRECISÃO p/ supervisão distante do classificador contextual de hedge (H1).
HEDGE_SEED = (
    HEDGE_LEXICON["fp"] + HEDGE_LEXICON["unc"] + ["maybe", "probably", "sort of", "kind of"]
)

CONTRAST_MARKERS: list[str] = [
    "but", "however", "although", "though", "yet", "on the other hand",
    "at the same time", "even though", "whereas", "nonetheless", "nevertheless", "still",
]

# Eixo de domínio do professor (proxy de desejo × resistência = ambivalência motivacional).
DESIRE_WORDS: list[str] = [
    "want", "wish", "hope", "would like", "intend", "plan", "should",
    "need to", "supposed to", "like to", "love to", "willing",
]
RESISTANCE_WORDS: list[str] = [
    "can't", "cannot", "won't", "avoid", "hard", "difficult", "struggle",
    "procrastinate", "never", "not able", "don't feel like", "reluctant", "hate",
]

_NORM_SUFFIX = {"raw": "", "per_word": "_ratew", "per_100": "_per100"}


class TextFeaturizer:
    """Vetor de texto de ambivalência/hesitância por janela (A1/A3/A4/H1).

    A1 e a flutuação de A3 são de RESPOSTA (broadcast p/ as janelas do vídeo); A4, H1 e a
    entropia de A3 são por JANELA. Dim fixa/determinística derivada dos flags/config.
    """

    name: str = "text_features"

    def __init__(
        self,
        *,
        use_a1: bool = True,
        use_a3: bool = True,
        use_a4: bool = True,
        use_h1: bool = True,
        use_emotion: bool = True,
        use_hedge_classifier: bool = False,
        emotion_model: str = "cardiffnlp/twitter-roberta-base-emotion",
        positive_labels: list[str] | None = None,
        negative_labels: list[str] | None = None,
        max_length: int = 128,
        batch_size: int = 32,
        norm: list[str] | None = None,
        device: str = "auto",
    ) -> None:
        self.use_a1 = bool(use_a1)
        self.use_a3 = bool(use_a3)
        self.use_a4 = bool(use_a4)
        self.use_h1 = bool(use_h1)
        self.use_emotion = bool(use_emotion)
        self.use_hedge_classifier = bool(use_hedge_classifier)
        self.emotion_model = str(emotion_model)
        self.positive_labels = list(positive_labels) if positive_labels else ["joy", "optimism"]
        self.negative_labels = list(negative_labels) if negative_labels else ["anger", "sadness"]
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.norm = list(norm) if norm else ["per_word", "per_100", "raw"]
        self.device_str = device

        # Handles carregados sob demanda (lazy); torch/vader são dropados no pickle.
        self._emotion = None
        self._emotion_tok = None
        self._pos_idx: list[int] | None = None
        self._neg_idx: list[int] | None = None
        self._vader = None
        self._hedge_vec = None  # CountVectorizer (fitted)
        self._hedge_clf = None  # LogisticRegression (fitted)
        self._fitted_hedge = False

        self._feature_names = self._build_feature_names()
        self._dim = len(self._feature_names)

    # --- picklabilidade: não serializa o modelo torch nem o VADER (recarregam) --------
    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        for k in ("_emotion", "_emotion_tok", "_vader"):
            state[k] = None
        return state

    # =========================================================================
    # Config
    # =========================================================================

    @classmethod
    def from_config(cls, cfg=None, *, device: str = "auto") -> TextFeaturizer:
        """Instancia a partir de um nó ``text_features`` (DictConfig ou dict); tolera ``None``."""

        def _get(key, default):
            if cfg is None:
                return default
            if hasattr(cfg, "get"):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        def _list(key, default):
            v = _get(key, None)
            if v is None:
                return default
            return list(v)

        return cls(
            use_a1=bool(_get("use_a1", True)),
            use_a3=bool(_get("use_a3", True)),
            use_a4=bool(_get("use_a4", True)),
            use_h1=bool(_get("use_h1", True)),
            use_emotion=bool(_get("use_emotion", True)),
            use_hedge_classifier=bool(_get("use_hedge_classifier", False)),
            emotion_model=str(_get("emotion_model", "cardiffnlp/twitter-roberta-base-emotion")),
            positive_labels=_list("positive_labels", ["joy", "optimism"]),
            negative_labels=_list("negative_labels", ["anger", "sadness"]),
            max_length=int(_get("max_length", 128)),
            batch_size=int(_get("batch_size", 32)),
            norm=_list("norm", ["per_word", "per_100", "raw"]),
            device=device,
        )

    # =========================================================================
    # Contrato
    # =========================================================================

    @property
    def dim(self) -> int:
        """Dimensão fixa do vetor de texto."""
        return self._dim

    def feature_names(self) -> list[str]:
        """Lista canônica (ordem das colunas de :meth:`extract`)."""
        return list(self._feature_names)

    def fit(self, texts: list[str]) -> TextFeaturizer:
        """Treina o classificador contextual de hedge (H1) por supervisão distante.

        Só age se ``use_h1`` e ``use_hedge_classifier``. Rótulo fraco: janela contém uma pista
        de hedge de alta precisão (:data:`HEDGE_SEED`) → 1. Features: BoW (uni+bigrama) do texto.
        Sem esse fit, ``text_h1_hedge_score`` cai para 0 (dim mantida).
        """
        if not (self.use_h1 and self.use_hedge_classifier):
            return self
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.linear_model import LogisticRegression

            texts = [t if isinstance(t, str) else "" for t in texts]
            y = np.array([1 if self._has_seed(t) else 0 for t in texts], dtype=np.int64)
            if y.sum() < 5 or (len(y) - y.sum()) < 5:
                log.warning("hedge classifier: rótulos fracos insuficientes; hedge_score = 0")
                return self
            self._hedge_vec = CountVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=5000, lowercase=True
            )
            X = self._hedge_vec.fit_transform(texts)
            self._hedge_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            self._hedge_clf.fit(X, y)
            self._fitted_hedge = True
            log.info(f"hedge classifier treinado: {int(y.sum())}/{len(y)} janelas com pista.")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"hedge classifier falhou no fit ({exc}); hedge_score = 0")
        return self

    def extract(self, texts: list[str], video_ids: list[str] | None = None) -> np.ndarray:
        """Vetor de texto por janela → matriz ``(n, dim)`` float32.

        ``video_ids`` agrupa as janelas para o broadcast de A1/A3-resposta. Se ``None``, cada
        janela é sua própria "resposta" (útil p/ testes; A1 fica esparso, como esperado).
        """
        n = len(texts)
        if n == 0:
            return np.zeros((0, self._dim), dtype=np.float32)
        texts = [t if isinstance(t, str) else "" for t in texts]
        vids = list(video_ids) if video_ids is not None else [f"_w{i}" for i in range(n)]

        # Emoção por janela (uma vez) — reusada em A3-janela, A3-flutuação e A1-emoção.
        need_emo = self.use_emotion and (self.use_a1 or self.use_a3)
        emo_win = self._emotion_probs(texts) if need_emo else None

        # Agrupa índices por vídeo preservando ordem.
        groups: dict[str, list[int]] = {}
        for i, v in enumerate(vids):
            groups.setdefault(v, []).append(i)

        rows: list[dict[str, float]] = [dict() for _ in range(n)]
        for _v, idxs in groups.items():
            resp_dict = self._response_features(
                [texts[i] for i in idxs], emo_win[idxs] if emo_win is not None else None
            )
            for i in idxs:
                win_dict = self._window_features(
                    texts[i], emo_win[i] if emo_win is not None else None
                )
                rows[i] = {**win_dict, **resp_dict}  # response é broadcast

        out = np.asarray(
            [[float(r.get(name, 0.0)) for name in self._feature_names] for r in rows],
            dtype=np.float32,
        )
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_one(self, text: str, video_id: str | None = None) -> np.ndarray:
        """Conveniência: uma janela (como sua própria resposta se ``video_id`` None)."""
        return self.extract([text], [video_id or "_w0"])[0]

    # =========================================================================
    # Blocos de RESPOSTA (A1, A3-flutuação/conflito) — broadcast
    # =========================================================================

    def _response_features(
        self, win_texts: list[str], emo_win: np.ndarray | None
    ) -> dict[str, float]:
        """Features de resposta (vídeo inteiro): A1 + parte de A3. Broadcast p/ as janelas."""
        out: dict[str, float] = {}
        resp_text = self._dedup_join(win_texts)
        n_words = max(len(_WORD_RE.findall(resp_text)), 1)

        if self.use_a1:
            # Fonte léxica (VADER) sobre a resposta inteira.
            p_lex, n_lex = self._vader_pn(resp_text)
            out.update(self._pn_indices("text_a1_vader", p_lex, n_lex))
            # Fonte emoção: média das probs por janela → distribuição da resposta.
            if emo_win is not None and emo_win.size:
                resp_probs = emo_win.mean(axis=0)
                p_emo = float(resp_probs[self._pos_idx].sum())
                n_emo = float(resp_probs[self._neg_idx].sum())
                out.update(self._pn_indices("text_a1_emo", p_emo, n_emo))
            # Eixo de domínio do professor (desejo × resistência).
            d = self._count(resp_text, DESIRE_WORDS)
            r = self._count(resp_text, RESISTANCE_WORDS)
            out.update(self._count_vals("text_a1_desire", d, n_words))
            out.update(self._count_vals("text_a1_resistance", r, n_words))
            dw, rw = d / n_words, r / n_words
            out["text_a1_desire_x_resistance"] = dw * rw
            out["text_a1_desire_min_resistance"] = min(dw, rw)

        if self.use_a3 and emo_win is not None and emo_win.shape[0] >= 1:
            # Flutuação da valência entre as janelas do vídeo (oscila entre estados).
            val = emo_win[:, self._pos_idx].sum(axis=1) - emo_win[:, self._neg_idx].sum(axis=1)
            out["text_a3_emotion_fluct_std"] = float(np.std(val))
            out["text_a3_emotion_fluct_range"] = float(val.max() - val.min())
            resp_probs = emo_win.mean(axis=0)
            p_emo = float(resp_probs[self._pos_idx].sum())
            n_emo = float(resp_probs[self._neg_idx].sum())
            out["text_a3_pole_conflict"] = min(p_emo, n_emo)
            out["text_a3_valence_resp"] = p_emo - n_emo
        return out

    # =========================================================================
    # Blocos de JANELA (A4, H1, A3-entropia)
    # =========================================================================

    def _window_features(self, text: str, emo_probs: np.ndarray | None) -> dict[str, float]:
        """Features por janela: A4 (contraste + shift), H1 (hedges), A3-entropia."""
        out: dict[str, float] = {}
        n_words = max(len(_WORD_RE.findall(text)), 1)

        if self.use_a4:
            c = self._count(text, CONTRAST_MARKERS)
            out.update(self._count_vals("text_a4_contrast", c, n_words))
            shift, flip = self._polarity_shift(text)
            out["text_a4_polarity_shift"] = shift
            out["text_a4_sign_flip"] = flip

        if self.use_h1:
            total = 0
            for sub, words in HEDGE_LEXICON.items():
                c = self._count(text, words)
                total += c
                out.update(self._count_vals(f"text_h1_{sub}", c, n_words))
            out.update(self._count_vals("text_h1_total", total, n_words))
            if self.use_hedge_classifier:
                out["text_h1_hedge_score"] = self._hedge_score(text)

        if self.use_a3 and emo_probs is not None and emo_probs.size:
            p = np.clip(emo_probs, 1e-9, 1.0)
            out["text_a3_emotion_entropy"] = float(-(p * np.log(p)).sum())
            top2 = np.sort(emo_probs)[::-1][:2]
            out["text_a3_emotion_top2gap"] = float(top2[0] - top2[1]) if top2.size >= 2 else 0.0
        return out

    # =========================================================================
    # Helpers de cálculo
    # =========================================================================

    def _pn_indices(self, prefix: str, p: float, n: float) -> dict[str, float]:
        """Índices psicométricos a partir de P e N (nunca polaridade líquida)."""
        return {
            f"{prefix}_P": float(p),
            f"{prefix}_N": float(n),
            f"{prefix}_intensity": float((p + n) / 2.0),
            f"{prefix}_inconsist": float(abs(p - n)),
            f"{prefix}_kaplan": float(min(p, n)),
            f"{prefix}_griffin": float((p + n) / 2.0 - abs(p - n)),
        }

    def _count_vals(self, base: str, count: int, n_words: int) -> dict[str, float]:
        """Contagem normalizada conforme ``self.norm`` (raw / por-palavra / por-100)."""
        out: dict[str, float] = {}
        for kind in self.norm:
            if kind == "raw":
                out[base] = float(count)
            elif kind == "per_word":
                out[f"{base}_ratew"] = count / n_words
            elif kind == "per_100":
                out[f"{base}_per100"] = 100.0 * count / n_words
        return out

    @staticmethod
    def _count(text: str, phrases: list[str]) -> int:
        low = text.lower()
        return sum(len(re.findall(r"\b" + re.escape(p) + r"\b", low)) for p in phrases)

    @staticmethod
    def _has_seed(text: str) -> bool:
        low = text.lower()
        return any(re.search(r"\b" + re.escape(p) + r"\b", low) for p in HEDGE_SEED)

    @staticmethod
    def _dedup_join(win_texts: list[str]) -> str:
        seen: set[str] = set()
        parts: list[str] = []
        for t in win_texts:
            t = (t or "").strip()
            if t and t not in seen:
                seen.add(t)
                parts.append(t)
        return " ".join(parts)

    def _polarity_shift(self, text: str) -> tuple[float, float]:
        """|VADER(1ª metade) − VADER(2ª metade)| e flag de troca de sinal (vira-volta)."""
        words = _WORD_RE.findall(text)
        if len(words) < 4:
            return 0.0, 0.0
        mid = len(words) // 2
        c1 = self._vader_compound(" ".join(words[:mid]))
        c2 = self._vader_compound(" ".join(words[mid:]))
        return abs(c1 - c2), (1.0 if c1 * c2 < 0 else 0.0)

    def _hedge_score(self, text: str) -> float:
        if not self._fitted_hedge or self._hedge_clf is None:
            return 0.0
        try:
            X = self._hedge_vec.transform([text or ""])
            return float(self._hedge_clf.predict_proba(X)[0, 1])
        except Exception:  # noqa: BLE001
            return 0.0

    # --- VADER (lazy) --------------------------------------------------------
    def _vader_obj(self):
        if self._vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader = SentimentIntensityAnalyzer()
        return self._vader

    def _vader_pn(self, text: str) -> tuple[float, float]:
        s = self._vader_obj().polarity_scores(text or "")
        return float(s["pos"]), float(s["neg"])

    def _vader_compound(self, text: str) -> float:
        return float(self._vader_obj().polarity_scores(text or "")["compound"])

    # --- Emoção (lazy) -------------------------------------------------------
    def _emotion_probs(self, texts: list[str]) -> np.ndarray:
        """Probabilidades de emoção (softmax) por texto → ``(n, n_labels)``."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._emotion is None:
            log.info(f"Carregando classificador de emoção: {self.emotion_model}")
            self._emotion_tok = AutoTokenizer.from_pretrained(self.emotion_model)
            self._emotion = AutoModelForSequenceClassification.from_pretrained(self.emotion_model)
            self._emotion.to(resolve_device(self.device_str))
            self._emotion.eval()
            id2label = {int(k): v.lower() for k, v in self._emotion.config.id2label.items()}
            pos = {x.lower() for x in self.positive_labels}
            neg = {x.lower() for x in self.negative_labels}
            self._pos_idx = [i for i, lab in id2label.items() if lab in pos]
            self._neg_idx = [i for i, lab in id2label.items() if lab in neg]
            log.info(f"Emoção: rótulos={id2label} pos_idx={self._pos_idx} neg_idx={self._neg_idx}")

        device = next(self._emotion.parameters()).device
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start : start + self.batch_size]
                batch = [t if (isinstance(t, str) and t.strip()) else "" for t in chunk]
                enc = self._emotion_tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                logits = self._emotion(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                out.append(probs.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    # =========================================================================
    # Nomes canônicos (dim determinística)
    # =========================================================================

    def _count_names(self, base: str) -> list[str]:
        names: list[str] = []
        for kind in self.norm:
            if kind == "raw":
                names.append(base)
            elif kind == "per_word":
                names.append(f"{base}_ratew")
            elif kind == "per_100":
                names.append(f"{base}_per100")
        return names

    def _build_feature_names(self) -> list[str]:
        """Lista canônica das colunas (ordem estável, derivada só dos flags/config)."""
        names: list[str] = []
        idx_suffix = ["P", "N", "intensity", "inconsist", "kaplan", "griffin"]

        if self.use_a1:
            names += [f"text_a1_vader_{s}" for s in idx_suffix]
            if self.use_emotion:
                names += [f"text_a1_emo_{s}" for s in idx_suffix]
            names += self._count_names("text_a1_desire")
            names += self._count_names("text_a1_resistance")
            names += ["text_a1_desire_x_resistance", "text_a1_desire_min_resistance"]
        if self.use_a4:
            names += self._count_names("text_a4_contrast")
            names += ["text_a4_polarity_shift", "text_a4_sign_flip"]
        if self.use_h1:
            for sub in HEDGE_LEXICON:
                names += self._count_names(f"text_h1_{sub}")
            names += self._count_names("text_h1_total")
            if self.use_hedge_classifier:
                names += ["text_h1_hedge_score"]
        if self.use_a3 and self.use_emotion:
            names += ["text_a3_emotion_entropy", "text_a3_emotion_top2gap"]  # janela
            names += [
                "text_a3_emotion_fluct_std", "text_a3_emotion_fluct_range",
                "text_a3_pole_conflict", "text_a3_valence_resp",
            ]  # resposta
        return names
