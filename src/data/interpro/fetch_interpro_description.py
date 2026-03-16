import json
import time
import argparse
import requests
from tqdm import tqdm


INTERPRO_API = "https://www.ebi.ac.uk/interpro/api/entry/interpro/"


def fetch_metadata(accession: str, retries: int = 3, delay: float = 1.0) -> dict:
    url = f"{INTERPRO_API}{accession}/"
    for _ in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                api_meta = resp.json().get("metadata", {})

                descriptions = api_meta.get("description", [])
                description = " ".join(d.get("text", "") for d in descriptions if d.get("text"))

                # literature is a dict of PMID -> citation object
                literature = api_meta.get("literature", {})

                return {"description": description, "literature": literature}
            elif resp.status_code == 404:
                return {"description": "", "literature": {}}
            else:
                time.sleep(delay)
        except requests.RequestException:
            time.sleep(delay)
    return {"description": "", "literature": {}}


def enrich_entries(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        entries = json.load(f)

    for entry in tqdm(entries, desc=f"Fetching metadata for {input_path}"):
        accession = entry.get("metadata", {}).get("accession", "")
        if accession:
            result = fetch_metadata(accession)
            entry["metadata"]["description"] = result["description"]
            entry["metadata"]["literature"] = result["literature"]
        time.sleep(0.2)  # be polite to the API

    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Saved {len(entries)} entries to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich InterPro JSON with descriptions from the API")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file (e.g. active_site.json)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file (e.g. active_site_des.json)")
    args = parser.parse_args()

    enrich_entries(args.input, args.output)
