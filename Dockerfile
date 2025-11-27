# Use a imagem oficial do Python 3.13 slim
FROM python:3.13-slim

# Metadados
LABEL maintainer="Renato"
LABEL description="AiBrickBreaker - Jogo Brick Breaker com Agente de Reinforcement Learning"
LABEL version="1.0"

# Instala dependências do sistema necessárias para o Pygame
# SDL2 e bibliotecas gráficas são necessárias mesmo em modo headless ou dummy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    libfreetype6 \
    libportmidi0 \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de requisitos
COPY requirements.txt .
COPY requirements_rl.txt .

# Instala as dependências Python
# Une os dois arquivos de requirements
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_rl.txt

# Copia o restante do código fonte
COPY . .

# Cria diretórios para modelos e logs
RUN mkdir -p models logs

# Define a variável de ambiente para evitar criação de arquivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Comando padrão (pode ser substituído por 'python train.py' ou 'python demo.py')
# Nota: Para rodar com GUI via Docker, é necessário configuração de X11 Forwarding no host
CMD ["python", "main.py"]
