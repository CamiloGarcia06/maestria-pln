#!/usr/bin/env bash
# Descarga FCE v2.1 y W&I+LOCNESS v2.1 desde la página de BEA-2019 (sin registro).
# Las licencias no permiten redistribuir los corpus, por eso datos/crudos/ no va al repositorio.
set -euo pipefail

destino="$(dirname "$0")/../datos/crudos"
mkdir -p "$destino"
cd "$destino"

for archivo in fce_v2.1.bea19.tar.gz wi+locness_v2.1.bea19.tar.gz; do
  [ -f "$archivo" ] || curl -sSfLO "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/$archivo"
  tar xzf "$archivo"
done
ls -d fce wi+locness
