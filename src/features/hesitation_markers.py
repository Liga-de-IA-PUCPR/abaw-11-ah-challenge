"""11 marcadores LÉXICOS de hesitação sobre a transcrição completa (nível de vídeo).

Sinal da cabeça auxiliar do plano MoE (``references/improvement_plan.md``, figura
"11 marcadores de hesitação — hedges, pausas, contraste, negação, repetição"). Diferente do
bloco ``text_*`` do tabular (por janela, ~49 colunas, com classificador de emoção), aqui o
vetor é compacto, determinístico (só regex) e calculado sobre a RESPOSTA inteira — a
hesitação verbal se espalha pela fala e, por janela (~24 palavras), as contagens são esparsas.

Os léxicos de hedge e contraste são os de :mod:`src.features.text_features` (fonte única);
este módulo só acrescenta negação, pausas preenchidas/marcadas, repetições e autorreparo.
Cada marcador é uma taxa por 100 palavras (``per_100``), então vídeos curtos e longos ficam
na mesma escala.

| marcador                 | o que conta                                                  |
|--------------------------|--------------------------------------------------------------|
| ``hm_hedge_epistemic``   | 1ª pessoa epistêmica ("I think", "I guess", …)               |
| ``hm_hedge_adverb``      | advérbios epistêmicos ("maybe", "probably", …)               |
| ``hm_hedge_approximator``| vagueza ("kind of", "sort of", "a bit", …)                   |
| ``hm_uncertainty``       | incerteza explícita ("not sure", "I don't know", …)          |
| ``hm_filled_pause``      | pausas preenchidas que o Whisper manteve ("um", "uh", …)     |
| ``hm_pause_punct``       | pausas marcadas na transcrição ("...", "—", " - ")           |
| ``hm_contrast``          | contraste ("but", "however", "on the other hand", …)         |
| ``hm_negation``          | negação ("not", "never", "n't", "nothing", …)                |
| ``hm_word_repetition``   | palavra repetida em sequência ("I I", "the the")             |
| ``hm_phrase_repetition`` | bigrama repetido em sequência ("I was I was") — recomeço     |
| ``hm_self_repair``       | autorreparo/reformulação ("I mean", "or rather", "sorry")    |
"""

from __future__ import annotations

import re

import numpy as np

from src.features.text_features import CONTRAST_MARKERS, HEDGE_LEXICON

_WORD_RE = re.compile(r"[a-z']+")

NEGATIONS: list[str] = [
    "not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "nowhere",
    "cannot", "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "isn't",
    "aren't", "wasn't", "weren't", "haven't", "hasn't", "shouldn't", "couldn't",
]  # fmt: skip
FILLED_PAUSES: list[str] = ["um", "umm", "uh", "uhh", "er", "erm", "hmm", "mm", "ah", "eh"]
SELF_REPAIRS: list[str] = [
    "i mean", "or rather", "sorry", "let me rephrase", "what i mean", "i meant",
    "or i guess", "well actually", "no wait",
]  # fmt: skip
_PAUSE_PUNCT = re.compile(r"\.\.\.|…|—|–| - ")

HESITATION_MARKER_NAMES: list[str] = [
    "hm_hedge_epistemic",
    "hm_hedge_adverb",
    "hm_hedge_approximator",
    "hm_uncertainty",
    "hm_filled_pause",
    "hm_pause_punct",
    "hm_contrast",
    "hm_negation",
    "hm_word_repetition",
    "hm_phrase_repetition",
    "hm_self_repair",
]
N_HESITATION_MARKERS = len(HESITATION_MARKER_NAMES)  # 11


def _count(text: str, phrases: list[str]) -> int:
    """Ocorrências (com fronteira de palavra) de qualquer frase da lista em ``text``."""
    return sum(len(re.findall(r"\b" + re.escape(p) + r"\b", text)) for p in phrases)


def _repetitions(words: list[str]) -> tuple[int, int]:
    """(palavras repetidas em sequência, bigramas repetidos em sequência)."""
    word_rep = sum(1 for a, b in zip(words, words[1:], strict=False) if a == b)
    bigrams = list(zip(words, words[1:], strict=False))
    phrase_rep = sum(1 for i in range(len(bigrams) - 2) if bigrams[i] == bigrams[i + 2])
    return word_rep, phrase_rep


def extract_hesitation_markers(text: str) -> dict[str, float]:
    """Os 11 marcadores (taxa por 100 palavras) de uma transcrição. Zeros se vazia."""
    low = (text or "").lower().replace("’", "'")
    words = _WORD_RE.findall(low)
    if not words:
        return dict.fromkeys(HESITATION_MARKER_NAMES, 0.0)
    word_rep, phrase_rep = _repetitions(words)
    counts = {
        "hm_hedge_epistemic": _count(low, HEDGE_LEXICON["fp"]),
        "hm_hedge_adverb": _count(low, HEDGE_LEXICON["adv"]),
        "hm_hedge_approximator": _count(low, HEDGE_LEXICON["approx"]),
        "hm_uncertainty": _count(low, HEDGE_LEXICON["unc"]),
        "hm_filled_pause": _count(low, FILLED_PAUSES),
        "hm_pause_punct": len(_PAUSE_PUNCT.findall(low)),
        "hm_contrast": _count(low, CONTRAST_MARKERS),
        "hm_negation": _count(low, NEGATIONS),
        "hm_word_repetition": word_rep,
        "hm_phrase_repetition": phrase_rep,
        "hm_self_repair": _count(low, SELF_REPAIRS),
    }
    per100 = 100.0 / len(words)
    return {name: float(counts[name]) * per100 for name in HESITATION_MARKER_NAMES}


def extract_hesitation_markers_vector(text: str) -> np.ndarray:
    """Como :func:`extract_hesitation_markers`, mas ``np.ndarray (11,)`` na ordem canônica."""
    feats = extract_hesitation_markers(text)
    return np.array([feats[n] for n in HESITATION_MARKER_NAMES], dtype=np.float32)
