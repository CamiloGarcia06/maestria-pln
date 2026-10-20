FROM python:3.12-slim

ARG UID=1000
ARG GID=1000

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl git make \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && python -m spacy download en_core_web_sm

RUN groupadd -g "$GID" pln && useradd -m -u "$UID" -g "$GID" pln
USER pln
# Punto de montaje del volumen hf-cache; si no existe en la imagen, Docker lo crea como root
RUN mkdir -p /home/pln/.cache/huggingface
WORKDIR /work

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/work"]
