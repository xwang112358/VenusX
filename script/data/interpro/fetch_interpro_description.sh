#!/bin/bash

DATA_DIR="data/interpro_2503"

for KEYWORD in active_site binding_site conserved_site domain motif; do
    echo "=== Fetching descriptions for ${KEYWORD} ==="
    python src/data/interpro/fetch_interpro_description.py \
        --input "${DATA_DIR}/${KEYWORD}/${KEYWORD}.json" \
        --output "${DATA_DIR}/${KEYWORD}/${KEYWORD}_des.json"
done