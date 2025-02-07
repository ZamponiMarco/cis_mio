FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /cis_mio

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    libcdd-dev \
    libgmp-dev \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /cis_mio/

RUN pip install -r requirements.txt

RUN git clone https://github.com/osqp/miosqp.git /tmp/miosqp \
    && cd /tmp/miosqp \
    && python setup.py install \
    && rm -rf /tmp/miosqp

COPY src/ /cis_mio/src/
COPY script/ /cis_mio/script/
COPY paper_data /cis_mio/paper_data

CMD ["/bin/bash"]
