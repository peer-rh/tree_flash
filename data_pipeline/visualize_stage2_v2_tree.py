from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path

import h5py
from transformers import AutoTokenizer
import sys

sys.setrecursionlimit(10000)



@dataclass(frozen=True)
class VisualTreeNode:
    index: int
    token_id: int
    display_token_id: int
    parent_index: int
    position_id: int
    rank: int
    path_prob: float
    child_indices: tuple[int, ...]
    source: str
    main_path_position: int | None = None
    anchor_index: int | None = None
    anchor_main_path_position: int | None = None
    stored_node_index: int | None = None


@dataclass(frozen=True)
class LoadedStage2V2Tree:
    input_path: str
    sequence_index: int
    record_idx: int
    response_start_position: int
    tokenizer_name_or_path: str
    nodes: tuple[VisualTreeNode, ...]


RANK_ZERO_COLOR = "#94a3b8"
RANK_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#d97706",
    "#0891b2",
    "#c026d3",
    "#4f46e5",
    "#ea580c",
)


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _format_token_text(tokenizer, token_id: int) -> str:
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
    if not text:
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        text = f"<{token}>"
    return text.replace(" ", "·").replace("\n", "↵\n").replace("\t", "⇥")


def load_stage2_v2_tree(
    input_path: str | Path,
    *,
    sequence_index: int = 0,
    tokenizer_name_or_path: str | None = None,
) -> LoadedStage2V2Tree:
    path = Path(input_path)
    with h5py.File(path, "r") as hf:
        format_version = _decode_attr(hf.attrs.get("format_version", ""))
        if format_version != "stage2_v2":
            raise ValueError(f"Expected a Stage 2 v2 file, found format_version={format_version!r}.")

        num_sequences = int(hf["record_idx"].shape[0])
        if sequence_index < 0 or sequence_index >= num_sequences:
            raise IndexError(f"sequence_index {sequence_index} is out of range for {num_sequences} sequences.")

        main_offsets = hf["main_path_offsets"]
        main_start = int(main_offsets[sequence_index])
        main_end = int(main_offsets[sequence_index + 1])
        main_path_ids = [int(value) for value in hf["main_path_ids"][main_start:main_end].tolist()]
        if not main_path_ids:
            raise ValueError("Selected sequence has an empty main path.")

        seq_anchor_offsets = hf["sequence_anchor_offsets"]
        anchor_start = int(seq_anchor_offsets[sequence_index])
        anchor_end = int(seq_anchor_offsets[sequence_index + 1])
        node_offsets = hf["anchor_node_offsets"]

        nodes: list[VisualTreeNode] = []
        main_node_indices: dict[int, int] = {}
        child_lists: dict[int, list[int]] = {}

        for position_id, token_id in enumerate(main_path_ids):
            index = len(nodes)
            main_node_indices[position_id] = index
            child_lists[index] = []
            nodes.append(
                VisualTreeNode(
                    index=index,
                    token_id=token_id,
                    display_token_id=token_id,
                    parent_index=index - 1 if index > 0 else -1,
                    position_id=position_id,
                    rank=0,
                    path_prob=1.0,
                    child_indices=(),
                    source="main",
                    main_path_position=position_id,
                )
            )

        for idx in range(1, len(nodes)):
            child_lists[idx - 1].append(idx)

        for anchor_index, anchor_table_index in enumerate(range(anchor_start, anchor_end)):
            anchor_main_path_position = int(hf["anchor_main_path_positions"][anchor_table_index])
            if anchor_main_path_position < 0 or anchor_main_path_position >= len(main_path_ids):
                raise ValueError(
                    f"anchor_main_path_position {anchor_main_path_position} is out of range for main path length {len(main_path_ids)}."
                )

            node_start = int(node_offsets[anchor_table_index])
            node_end = int(node_offsets[anchor_table_index + 1])
            token_ids = [int(value) for value in hf["node_token_ids"][node_start:node_end].tolist()]
            parent_indices = [int(value) for value in hf["node_parent_indices"][node_start:node_end].tolist()]
            depths = [int(value) for value in hf["node_depths"][node_start:node_end].tolist()]
            path_probs = [float(value) for value in hf["node_path_probs"][node_start:node_end].tolist()]
            ranks = [int(value) for value in hf["node_ranks"][node_start:node_end].tolist()]

            if not token_ids:
                continue

            anchor_node_indices: dict[int, int] = {}
            for stored_node_index in range(1, len(token_ids)):
                combined_index = len(nodes)
                anchor_node_indices[stored_node_index] = combined_index
                child_lists[combined_index] = []
                parent_index = parent_indices[stored_node_index]
                if parent_index == 0:
                    combined_parent_index = main_node_indices[anchor_main_path_position]
                else:
                    combined_parent_index = anchor_node_indices[parent_index]
                nodes.append(
                    VisualTreeNode(
                        index=combined_index,
                        token_id=token_ids[stored_node_index],
                        display_token_id=token_ids[stored_node_index],
                        parent_index=combined_parent_index,
                        position_id=anchor_main_path_position + depths[stored_node_index],
                        rank=ranks[stored_node_index],
                        path_prob=path_probs[stored_node_index],
                        child_indices=(),
                        source="anchor",
                        anchor_index=anchor_index,
                        anchor_main_path_position=anchor_main_path_position,
                        stored_node_index=stored_node_index,
                    )
                )
                child_lists[combined_parent_index].append(combined_index)

        def child_sort_key(child_index: int) -> tuple[int, int, int, int]:
            child = nodes[child_index]
            if child.source == "main":
                return (0, child.main_path_position or 0, 0, child.index)
            return (
                1,
                child.anchor_main_path_position or 0,
                child.rank,
                child.stored_node_index or child.index,
            )

        updated_nodes = []
        for node in nodes:
            children = sorted(child_lists[node.index], key=child_sort_key)
            updated_nodes.append(
                VisualTreeNode(
                    index=node.index,
                    token_id=node.token_id,
                    display_token_id=node.display_token_id,
                    parent_index=node.parent_index,
                    position_id=node.position_id,
                    rank=node.rank,
                    path_prob=node.path_prob,
                    child_indices=tuple(children),
                    source=node.source,
                    main_path_position=node.main_path_position,
                    anchor_index=node.anchor_index,
                    anchor_main_path_position=node.anchor_main_path_position,
                    stored_node_index=node.stored_node_index,
                )
            )

        resolved_tokenizer = tokenizer_name_or_path or _decode_attr(hf.attrs.get("tokenizer_name_or_path", ""))
        if not resolved_tokenizer:
            raise ValueError("Tokenizer name/path is missing; pass --tokenizer explicitly.")

        return LoadedStage2V2Tree(
            input_path=str(path),
            sequence_index=sequence_index,
            record_idx=int(hf["record_idx"][sequence_index]),
            response_start_position=int(hf["response_start_positions"][sequence_index]),
            tokenizer_name_or_path=resolved_tokenizer,
            nodes=tuple(updated_nodes),
        )


def _assign_x_slots(nodes: tuple[VisualTreeNode, ...]) -> dict[int, float]:
    root_candidates = [node.index for node in nodes if node.parent_index < 0]
    if len(root_candidates) != 1:
        raise ValueError(f"Expected exactly one root node, found {len(root_candidates)}.")

    x_slots: dict[int, float] = {}
    next_leaf_slot = 0

    def visit(node_index: int) -> float:
        nonlocal next_leaf_slot
        children = list(nodes[node_index].child_indices)
        if not children:
            slot = float(next_leaf_slot)
            next_leaf_slot += 1
        else:
            child_slots = [visit(child_index) for child_index in children]
            slot = sum(child_slots) / len(child_slots)
        x_slots[node_index] = slot
        return slot

    visit(root_candidates[0])
    return x_slots


def _rank_color(rank: int) -> str:
    if rank == 0:
        return RANK_ZERO_COLOR
    return RANK_COLORS[(rank - 1) % len(RANK_COLORS)]


def _probability_opacity(path_prob: float, *, source: str) -> float:
    if source == "main":
        return 0.95
    clamped = min(max(float(path_prob), 0.0), 1.0)
    return 0.2 + 0.75 * (clamped ** 0.5)


def render_stage2_v2_tree_html(tree: LoadedStage2V2Tree, tokenizer) -> str:
    x_slots = _assign_x_slots(tree.nodes)
    min_position_id = min(node.position_id for node in tree.nodes)
    max_position_id = max(node.position_id for node in tree.nodes)
    num_anchor_nodes = sum(1 for node in tree.nodes if node.source == "anchor")
    num_main_nodes = sum(1 for node in tree.nodes if node.source == "main")
    num_anchors = len({node.anchor_index for node in tree.nodes if node.anchor_index is not None})

    node_width = 152
    node_height = 38
    x_step = 184
    y_step = 112
    margin_x = 56
    margin_y = 32

    positioned_nodes: list[dict[str, object]] = []
    for node in tree.nodes:
        x_center = margin_x + x_slots[node.index] * x_step + node_width / 2
        y_top = margin_y + (node.position_id - min_position_id) * y_step
        token_text = _format_token_text(tokenizer, node.display_token_id)
        positioned_nodes.append(
            {
                "node": node,
                "x_center": x_center,
                "y_top": y_top,
                "token_text": token_text,
                "fill": _rank_color(node.rank),
                "opacity": _probability_opacity(node.path_prob, source=node.source),
            }
        )

    max_slot = max(x_slots.values()) if x_slots else 0.0
    svg_width = int(margin_x * 2 + (max_slot + 1) * x_step + node_width)
    svg_height = int(margin_y * 2 + (max_position_id - min_position_id + 1) * y_step + node_height)
    node_by_index = {entry["node"].index: entry for entry in positioned_nodes}

    edge_parts: list[str] = []
    node_parts: list[str] = []
    for entry in positioned_nodes:
        node = entry["node"]
        x_center = float(entry["x_center"])
        y_top = float(entry["y_top"])
        if node.parent_index >= 0:
            parent = node_by_index[node.parent_index]
            parent_x = float(parent["x_center"])
            parent_y = float(parent["y_top"]) + node_height
            edge_parts.append(
                (
                    f'<line x1="{parent_x:.1f}" y1="{parent_y:.1f}" '
                    f'x2="{x_center:.1f}" y2="{y_top:.1f}" '
                    'stroke="#94a3b8" stroke-width="2.2" stroke-linecap="round" />'
                )
            )

        token_text = html.escape(str(entry["token_text"]))
        tooltip = html.escape(
            "\n".join(
                [
                    f"node_index: {node.index}",
                    f"token_id: {node.token_id}",
                    f"rank: {node.rank}",
                    f"path_prob: {node.path_prob:.6f}",
                    f"position_id: {node.position_id}",
                    f"source: {node.source}",
                    f"parent_index: {node.parent_index}",
                    (
                        f"anchor_index: {node.anchor_index}"
                        if node.anchor_index is not None
                        else f"main_path_position: {node.main_path_position}"
                    ),
                ]
            )
        )
        node_parts.append(
            (
                f'<g class="tree-node {node.source}-node rank-{node.rank}" data-node-index="{node.index}" '
                f'data-source="{node.source}" data-rank="{node.rank}" data-prob="{node.path_prob:.6f}" '
                f'data-position-id="{node.position_id}">'
                f"<title>{tooltip}</title>"
                f'<rect x="{x_center - node_width / 2:.1f}" y="{y_top:.1f}" '
                f'width="{node_width}" height="{node_height}" rx="12" ry="12" '
                f'fill="{entry["fill"]}" fill-opacity="{float(entry["opacity"]):.3f}" '
                'stroke="#334155" stroke-width="1.2" />'
                f'<text x="{x_center:.1f}" y="{y_top + node_height / 2 + 5:.1f}" '
                'text-anchor="middle" font-size="14" font-weight="600" fill="#0f172a">'
                f"{token_text}</text>"
                "</g>"
            )
        )

    metadata_items = [
        ("Input", tree.input_path),
        ("Sequence", str(tree.sequence_index)),
        ("Record", str(tree.record_idx)),
        ("Response Start", str(tree.response_start_position)),
        ("Tokenizer", tree.tokenizer_name_or_path),
        ("Main Nodes", str(num_main_nodes)),
        ("Anchor Nodes", str(num_anchor_nodes)),
        ("Anchors", str(num_anchors)),
    ]
    metadata_html = "".join(
        f'<li><span class="meta-label">{html.escape(label)}:</span> {html.escape(value)}</li>'
        for label, value in metadata_items
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Stage 2 v2 Sequence Tree</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: "Iosevka Etoile", "IBM Plex Sans", "Helvetica Neue", sans-serif;
      background:
        radial-gradient(circle at top left, #dbeafe 0, transparent 28%),
        radial-gradient(circle at top right, #fde68a 0, transparent 24%),
        #f8fafc;
      color: #0f172a;
    }}
    body {{
      margin: 0;
      padding: 28px;
    }}
    .panel {{
      max-width: fit-content;
      border: 1px solid #cbd5e1;
      border-radius: 18px;
      padding: 20px 22px 24px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 22px 48px rgba(15, 23, 42, 0.10);
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 20px;
      letter-spacing: 0.02em;
    }}
    .meta {{
      margin: 0 0 14px;
      padding: 0;
      list-style: none;
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, auto));
      gap: 6px 18px;
      font-size: 13px;
    }}
    .meta-label {{
      color: #475569;
      font-weight: 700;
    }}
    .note {{
      margin: 0 0 16px;
      font-size: 12px;
      color: #475569;
    }}
    svg {{
      display: block;
      overflow: visible;
    }}
    text {{
      dominant-baseline: middle;
      pointer-events: none;
    }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>Stage 2 v2 Sequence Tree</h1>
    <ul class="meta">{metadata_html}</ul>
    <p class="note">The full sequence backbone is rendered together with every anchor subtree. Y position is the sequence position id.</p>
    <svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
      {''.join(edge_parts)}
      {''.join(node_parts)}
    </svg>
  </div>
</body>
</html>
"""


def write_stage2_v2_tree_html(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sequence_index: int = 0,
    tokenizer_name_or_path: str | None = None,
) -> Path:
    tree = load_stage2_v2_tree(
        input_path,
        sequence_index=sequence_index,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(tree.tokenizer_name_or_path)
    html_text = render_stage2_v2_tree_html(tree, tokenizer)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a raw Stage 2 v2 sequence tree as standalone HTML.")
    parser.add_argument("--input", required=True, help="Path to a Stage 2 v2 HDF5 file.")
    parser.add_argument("--sequence-index", type=int, default=0, help="Zero-based sequence index.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer override. Defaults to the file's tokenizer_name_or_path attribute.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path = write_stage2_v2_tree_html(
        args.input,
        args.output,
        sequence_index=args.sequence_index,
        tokenizer_name_or_path=args.tokenizer,
    )
    print(f"Wrote Stage 2 v2 sequence tree visualization to {output_path}")


if __name__ == "__main__":
    main()
