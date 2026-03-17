from __future__ import annotations

from evaluation_llm.catalog import LabelCatalog
from evaluation_llm.interfaces import PromptBuilder
from evaluation_llm.types import FragmentExample, LabelCard, PromptContext, PromptPackage


def _render_label_card(card: LabelCard, style: str) -> str:
    if style == "name_only":
        return f"- {card.accession} | {card.name}"
    if style == "short_desc":
        return f"- {card.accession} | {card.name} | {card.short_desc}"
    if style == "rich_desc":
        go_terms = ", ".join(card.go_terms[:3]) if card.go_terms else "None"
        return (
            f"- {card.accession} | {card.name}\n"
            f"  Type: {card.label_type}\n"
            f"  Description: {card.description}\n"
            f"  GO terms: {go_terms}\n"
            f"  Literature count: {card.literature_count}"
        )
    raise ValueError(f"Unsupported label_card_style={style!r}")


def _annotate_full_sequence(example: FragmentExample) -> str:
    annotated = example.seq_full
    for index, (start, end) in reversed(list(enumerate(example.ranges(), start=1))):
        if start <= 0 or end < start:
            continue
        left = annotated[: start - 1]
        middle = annotated[start - 1 : end]
        right = annotated[end:]
        annotated = f"{left}<frag{index}:{start}-{end}>{middle}</frag{index}>{right}"
    return annotated


def _render_fragment(example: FragmentExample) -> str:
    parts = []
    for index, (fragment, start, end) in enumerate(
        zip(example.fragment_parts, example.start_parts, example.end_parts),
        start=1,
    ):
        parts.append(f"{index}. residues {start}-{end}: {fragment}")
    return "\n".join(parts)


def _render_few_shot_example(example: FragmentExample, card: LabelCard) -> str:
    return (
        f"Fragment example:\n{_render_fragment(example)}\n"
        f"Correct label:\n{card.accession} | {card.name}\n"
        'Expected JSON:\n{"top_ids":["' + card.accession + '"],"confidence":1.0,"abstain":false}'
    )


class FragmentPromptBuilder(PromptBuilder):
    def build(
        self,
        context: PromptContext,
        catalog: LabelCatalog,
    ) -> PromptPackage:
        candidate_cards = tuple(sorted(context.candidate_cards, key=lambda card: card.accession))
        few_shot_examples = context.few_shot_examples

        sections = [
            "You are doing fragment-level protein function label selection.",
            "Choose the best InterPro accession for the fragment from the candidate labels.",
            'Return JSON only with the schema {"top_ids":["IPR..."],"confidence":0.0,"abstain":false}.',
            "Use canonical InterPro accessions in top_ids, ordered from best to worst.",
        ]
        if context.candidate_records:
            sections.append(
                f"Candidate strategy: {context.config.candidate_strategy}. "
                f"Candidate count: {len(context.candidate_records)}."
            )

        if few_shot_examples:
            rendered_examples = []
            for example in few_shot_examples:
                card = catalog.by_accession[example.interpro_id]
                rendered_examples.append(_render_few_shot_example(example, card))
            sections.append("Few-shot examples:\n" + "\n\n".join(rendered_examples))

        query_lines = [
            f"Query uid: {context.example.uid}",
            f"Fragment count: {len(context.example.fragment_parts)}",
            "Fragment parts:",
            _render_fragment(context.example),
        ]
        if context.config.include_full_sequence:
            query_lines.extend(
                [
                    f"Full sequence length: {len(context.example.seq_full)}",
                    "Full sequence with fragment tags:",
                    _annotate_full_sequence(context.example),
                ]
            )
        sections.append("\n".join(query_lines))

        rendered_cards = "\n".join(
            _render_label_card(card, context.config.label_card_style)
            for card in candidate_cards
        )
        sections.append("Candidate labels:\n" + rendered_cards)
        sections.append(
            "If you are uncertain, set abstain=true and leave top_ids empty. "
            "Do not invent accessions outside the candidate list."
        )

        return PromptPackage(
            prompt="\n\n".join(sections),
            candidate_cards=candidate_cards,
            few_shot_examples=few_shot_examples,
        )
