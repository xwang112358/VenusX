import json
import os
import argparse


def check_nulls(json_path: str) -> dict:
    with open(json_path) as f:
        entries = json.load(f)

    total = len(entries)
    empty_description = []
    empty_literature = []
    both_empty = []

    for entry in entries:
        meta = entry.get("metadata", {})
        accession = meta.get("accession", "UNKNOWN")
        desc = meta.get("description", None)
        lit = meta.get("literature", None)

        no_desc = desc is None or desc == ""
        no_lit = lit is None or lit == {}

        if no_desc:
            empty_description.append(accession)
        if no_lit:
            empty_literature.append(accession)
        if no_desc and no_lit:
            both_empty.append(accession)

    return {
        "total": total,
        "empty_description": empty_description,
        "empty_literature": empty_literature,
        "both_empty": both_empty,
    }


def write_summary(data_dir: str, output_path: str):
    keywords = ["active_site", "binding_site", "conserved_site", "domain", "motif"]
    lines = []

    for keyword in keywords:
        json_path = os.path.join(data_dir, keyword, f"{keyword}_des.json")
        if not os.path.exists(json_path):
            lines.append(f"[{keyword}] SKIPPED — file not found: {json_path}\n")
            continue

        stats = check_nulls(json_path)
        total = stats["total"]
        n_no_desc = len(stats["empty_description"])
        n_no_lit = len(stats["empty_literature"])
        n_both = len(stats["both_empty"])

        lines.append(f"{'='*60}\n")
        lines.append(f"[{keyword}]  {json_path}\n")
        lines.append(f"  Total entries        : {total}\n")
        lines.append(f"  Empty description    : {n_no_desc} ({100*n_no_desc/total:.1f}%)\n")
        lines.append(f"  Empty literature     : {n_no_lit} ({100*n_no_lit/total:.1f}%)\n")
        lines.append(f"  Both empty           : {n_both} ({100*n_both/total:.1f}%)\n")

        if stats["both_empty"]:
            lines.append(f"  Both-empty accessions:\n")
            for acc in stats["both_empty"]:
                lines.append(f"    {acc}\n")

        lines.append("\n")

    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"Summary written to {output_path}")
    print("".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check null description/literature in *_des.json files")
    parser.add_argument("--data_dir", type=str, default="data/interpro_2503")
    parser.add_argument("--output", type=str, default="data/interpro_2503/des_null_summary.txt")
    args = parser.parse_args()

    write_summary(args.data_dir, args.output)
