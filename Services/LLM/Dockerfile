FROM python:3.11-slim

ENV OLLAMA_VERSION=latest
ENV OLLAMA_HOST="0.0.0.0"

RUN apt-get update && apt-get install -y \
    curl \
    libssl-dev \
    libffi-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz && \
    tar -C /usr -xzf ollama-linux-amd64.tgz && \
    rm ollama-linux-amd64.tgz

RUN nohup ollama serve & sleep 5 && ollama pull llama3

EXPOSE 11434

CMD ["sh", "-c", "ollama serve"]