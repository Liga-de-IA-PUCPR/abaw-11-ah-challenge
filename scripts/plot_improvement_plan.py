"""Gera figuras de arquitetura/fluxo (docs/improvement_plan.md) com Graphviz.

Usa `graphviz` (binding Python do `dot`) em vez de desenho manual em matplotlib:
o roteamento de setas e o layout em grafo dirigido ficam a cargo do Graphviz,
evitando cruzamentos e sobreposições que surgem ao posicionar caixas à mão.
"""

from __future__ import annotations

from pathlib import Path

import graphviz

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"

COLORS = {
    "text": "#4C72B0",
    "audio": "#DD8452",
    "asr": "#55A868",
    "face": "#8172B2",
    "scene": "#C44E52",
    "fusion": "#937860",
    "ensemble": "#64B5CD",
    "neutral": "#4D4D4D",
}

FONT = "Georgia,serif"


def _node_attrs(color: str, *, shape: str = "box", style: str = "filled,rounded", **extra):
    attrs = {
        "shape": shape,
        "style": style,
        "fillcolor": color + "1A",
        "color": color,
        "fontcolor": "#1a1a1a",
        "fontname": FONT,
        "fontsize": "11",
        "penwidth": "1.3",
    }
    attrs.update(extra)
    return attrs


def plot_architecture() -> Path:  # noqa: PLR0915 (layout de figura, uma peça só)
    g = graphviz.Digraph(
        "architecture",
        graph_attr={
            "rankdir": "LR",
            "splines": "spline",
            "fontname": FONT,
            "fontsize": "18",
            "label": (
                "Arquitetura proposta — fusão ancorada em texto + Mixture-of-Experts"
                " como agregador central (SoftMoE, vision-toolbelt)"
            ),
            "labelloc": "t",
            "bgcolor": "white",
            "pad": "0.4",
            "nodesep": "0.35",
            "ranksep": "0.9",
            "compound": "true",
        },
        node_attr={"fontname": FONT, "fontsize": "10.5"},
        edge_attr={"fontname": FONT, "fontsize": "9", "color": "#666666"},
    )

    # ---------------- LEFT: branches por modalidade ------------------
    with g.subgraph(name="cluster_left") as c:
        c.attr(
            label="LEFT · branches por modalidade",
            fontname=FONT,
            fontsize="13",
            style="rounded",
            color="#888888",
        )

        c.node("in_text", "transcript", **_node_attrs(COLORS["text"], shape="note"))
        c.node(
            "text_main",
            "RoBERTa-GoEmotions\n(fine-tune, 4 camadas cong.)\n[768-d]",
            _attributes=_node_attrs(COLORS["text"]),
        )
        c.node(
            "text_aux",
            "11 marcadores de hesitação\n(hedges, pausas, contraste,\nnegação, repetição)\n[11-d]",
            _attributes=_node_attrs(COLORS["text"], style="filled,rounded,dashed"),
        )
        c.edge("in_text", "text_main", color=COLORS["text"])
        c.edge("in_text", "text_aux", color=COLORS["text"], style="dashed")

        c.node("in_audio", "waveform 16kHz", **_node_attrs(COLORS["audio"], shape="cds"))
        c.node(
            "audio_main",
            "wav2vec2-emotion +\ncabeça temporal\n(supervisão por janela)\n[1024-d]",
            _attributes=_node_attrs(COLORS["audio"]),
        )
        c.edge("in_audio", "audio_main", color=COLORS["audio"])

        c.node(
            "in_asr", "Whisper chunk\ntimestamps", **_node_attrs(COLORS["asr"], shape="cylinder")
        )
        c.node(
            "asr_main",
            "ASR-erased time\n16 features de gaps\n(reset 30s tratado)\n[16-d]",
            _attributes=_node_attrs(COLORS["asr"]),
        )
        c.edge("in_asr", "asr_main", color=COLORS["asr"])

        c.node(
            "in_face",
            "recortes vision-toolbelt\nface / olhos / boca",
            _attributes=_node_attrs(COLORS["face"], shape="component"),
        )
        c.node(
            "face_main",
            "backbone → Transformer\n→ [μ,σ,μΔ,σΔ]\n[512-d]",
            _attributes=_node_attrs(COLORS["face"]),
        )
        c.edge("in_face", "face_main", color=COLORS["face"])

        c.node(
            "scene",
            "Scene (opcional)\nVideoMAE-v2 congelado\n16 frames",
            _attributes=_node_attrs(COLORS["scene"], style="filled,rounded,dashed"),
        )
        c.edge("in_face", "scene", color=COLORS["scene"], style="dashed")

    # ---------------- CENTER: Text Residual Fusion --------------------
    with g.subgraph(name="cluster_center") as c:
        c.attr(
            label="CENTER · Fusão ancorada em texto + SoftMoE",
            fontname=FONT,
            fontsize="13",
            style="rounded",
            color="#888888",
        )

        for key in ("text", "audio", "asr", "face"):
            c.node(
                f"proj_{key}", "projection\nblock", **_node_attrs(COLORS[key], shape="trapezium")
            )

        c.node(
            "gate",
            "reliability gate\ng_m = σ(MLP[b; h_m])",
            _attributes=_node_attrs(COLORS["fusion"], shape="invtrapezium"),
        )
        c.node(
            "sum",
            "⊕",
            **_node_attrs(COLORS["fusion"], shape="circle", fixedsize="true", width="0.5"),
        )
        c.node(
            "moe",
            "SoftMoE (MoEFusionHead)\n\n"
            "z = LN(b + Σ_m g_m·d_m(h_m))\n"
            "router: Linear → softmax(K)\n"
            "K=3–4 ExpertMLP → p(A/H)",
            _attributes=_node_attrs(COLORS["fusion"], shape="box3d"),
        )
        c.node(
            "balance_note",
            "load-balancing loss\n(só se 1 expert > 70% do tráfego)",
            _attributes=_node_attrs(
                COLORS["fusion"], shape="note", style="filled,rounded,dashed", fontsize="9"
            ),
        )

        c.edge("proj_text", "gate", color=COLORS["text"])
        c.edge("proj_audio", "gate", color=COLORS["audio"])
        c.edge("proj_asr", "gate", color=COLORS["asr"])
        c.edge("proj_face", "gate", color=COLORS["face"])

        c.edge("gate", "sum", color=COLORS["fusion"], label="g_m")
        c.edge("sum", "moe", color=COLORS["fusion"])
        c.edge("moe", "balance_note", color=COLORS["fusion"], style="dotted", arrowhead="none")

        c.node(
            "anchor_note",
            "texto = âncora b · demais modalidades = ajustes residuais gateados",
            _attributes=_node_attrs("#555555", shape="plaintext", fontsize="9"),
        )
        c.edge("moe", "anchor_note", style="invis")

    g.edge("text_main", "proj_text", color=COLORS["text"], lhead="cluster_center")
    g.edge(
        "text_aux",
        "moe",
        color=COLORS["text"],
        style="dashed",
        label="cabeça auxiliar\n(peso 0.3)",
        fontsize="8",
    )
    g.edge("audio_main", "proj_audio", color=COLORS["audio"])
    g.edge("asr_main", "proj_asr", color=COLORS["asr"])
    g.edge("face_main", "proj_face", color=COLORS["face"])

    # ---------------- RIGHT: MoERouter (sucessor do router CA/GNN) -----
    with g.subgraph(name="cluster_right") as c:
        c.attr(
            label="RIGHT · MoERouter — sucessor do router CA/GNN",
            fontname=FONT,
            fontsize="13",
            style="rounded",
            color="#888888",
        )
        members = [
            ("m_fusion", "MoEFusionHead (ours)", COLORS["fusion"]),
            ("m_text", "Texto fine-tune (unimodal)", COLORS["text"]),
            ("m_audio", "Áudio temporal (unimodal)", COLORS["audio"]),
            ("m_asr", "ASR-erased time (unimodal)", COLORS["asr"]),
            ("m_router", "CA / GNN atuais (se ajudarem no OOF)", COLORS["neutral"]),
        ]
        for node_id, label, color in members:
            c.node(node_id, label, **_node_attrs(color))

        c.node(
            "embed_note",
            "cada membro passa a expor um\nvetor pré-logit (embeddings.npy),\nnão só p(x) escalar",
            _attributes=_node_attrs(
                COLORS["neutral"], shape="note", style="filled,rounded,dashed", fontsize="8.5"
            ),
        )
        c.node(
            "moe_router",
            "MoERouter (SoftMoE)\n\n"
            "router: Linear(embeddings) → softmax(K)\n"
            "escolhe/combina membros por amostra\n"
            "τ = 0.5 fixo",
            _attributes=_node_attrs(COLORS["ensemble"], shape="box3d"),
        )
        c.node("decision", "A/H  vs.  No A/H", **_node_attrs(COLORS["ensemble"], shape="ellipse"))
        for node_id, _label, color in members:
            c.edge(node_id, "moe_router", color=color)
        c.edge(
            "embed_note", "moe_router", style="dotted", arrowhead="none", color=COLORS["neutral"]
        )
        c.edge("moe_router", "decision", color=COLORS["ensemble"])

    g.edge("moe", "m_fusion", color=COLORS["fusion"])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_base = FIG_DIR / "architecture"
    g.render(filename=str(out_base), format="png", cleanup=True)
    g.render(filename=str(out_base), format="pdf", cleanup=True)
    return out_base.with_suffix(".png")


def plot_workflow() -> Path:
    g = graphviz.Digraph(
        "workflow",
        graph_attr={
            "rankdir": "TB",
            "splines": "ortho",
            "fontname": FONT,
            "fontsize": "16",
            "label": (
                "Workflow iterativo: rodadas de agregação via MoE, cada uma com gate de decisão OOF"
            ),
            "labelloc": "t",
            "bgcolor": "white",
            "pad": "0.4",
            "nodesep": "0.4",
            "ranksep": "0.55",
        },
        node_attr={"fontname": FONT, "fontsize": "11"},
        edge_attr={"fontname": FONT, "fontsize": "9", "color": "#333333"},
    )

    stages = [
        (
            "r0",
            "Rodada 0 · Fundação",
            "protocolo OOF: 5-fold por participante + baseline router atual",
            "neutral",
            False,
            "baseline OOF medido",
        ),
        (
            "r1",
            "Rodada 1 · Texto como âncora",
            "GoEmotions fine-tune + marcadores (TextSequenceDataset pendente)",
            "text",
            False,
            "OOF AP texto ≥ 0.80",
        ),
        (
            "r2",
            "Rodada 2 · 1º MoE (texto + tabular)",
            "prova a plumbing: MoE não colapsa, embeddings.npy habilitado",
            "fusion",
            False,
            "MoE ≥ texto sozinho no OOF",
        ),
        (
            "r3",
            "Rodada 3 · + ASR-erased time",
            "3º expert candidato, 16 features de gaps",
            "asr",
            False,
            "MoE melhora vs. Rodada 2",
        ),
        (
            "r4",
            "Rodada 4 · + Áudio de emoção",
            "wav2vec2-emotion + cabeça temporal (Mamba só se gargalo)",
            "audio",
            True,
            "MoE melhora vs. Rodada 3",
        ),
        (
            "r5",
            "Rodada 5 · + Canal visual",
            "vision-toolbelt: face/olhos/boca → SoftMoE de crops → expert",
            "face",
            True,
            "MoE melhora vs. Rodada 4",
        ),
        (
            "r6",
            "Rodada 6 · Consolidação",
            "MoERouter final, re-treino com holdout, 1 medição no public test",
            "ensemble",
            False,
            "test > 0.7454, medido 1x",
        ),
    ]

    prev_id = None
    prev_criterion = None
    for node_id, title, subtitle, color_key, needs_video, criterion in stages:
        color = COLORS[color_key]
        tag = "[vídeo] requer download" if needs_video else "sem vídeo"
        tag_color = "#C44E52" if needs_video else "#55A868"
        label = (
            f'<<table border="0" cellborder="0" cellspacing="0">'
            f'<tr><td align="left"><b>{title}</b></td>'
            f'<td align="right"><font color="{tag_color}" point-size="9">{tag}</font></td></tr>'
            f'<tr><td colspan="2" align="left"><font point-size="9.5">{subtitle}</font></td></tr>'
            f"</table>>"
        )
        g.node(node_id, label, **_node_attrs(color, shape="box", width="6"))
        if prev_id is not None:
            g.edge(
                prev_id,
                node_id,
                xlabel=f"critério: {prev_criterion}",
                fontsize="9",
                fontcolor="#555555",
            )
        prev_id = node_id
        prev_criterion = criterion

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_base = FIG_DIR / "workflow"
    g.render(filename=str(out_base), format="png", cleanup=True)
    g.render(filename=str(out_base), format="pdf", cleanup=True)
    return out_base.with_suffix(".png")


def main() -> None:
    arch = plot_architecture()
    flow = plot_workflow()
    print(f"saved: {arch}")
    print(f"saved: {flow}")


if __name__ == "__main__":
    main()
