# AiBrickBreaker - v1.0

Este projeto é uma implementação do clássico jogo **Brick Breaker** (Arkanoid/Breakout) em Python utilizando `pygame`, integrado com um agente de Inteligência Artificial baseado em Reinforcement Learning (PPO via `Stable Baselines3`).

## 📂 Estrutura do Projeto

```
AiBrickBreaker/
├── models/             # Modelos de IA salvos (.zip)
├── logs/               # Logs do TensorBoard
├── src/                # Código fonte
│   ├── config.py       # Configurações globais (Física, RL, Cores)
│   ├── game.py         # Lógica principal do jogo
│   ├── sprites.py      # Classes (Paddle, Ball, Brick)
│   └── rl_env.py       # Wrapper Gymnasium para RL
├── main.py             # Jogo modo Humano
├── train.py            # Script de Treinamento da IA
├── demo.py             # Demonstração da IA jogando
├── Dockerfile          # Configuração Docker
├── requirements.txt    # Dependências do Jogo
└── requirements_rl.txt # Dependências de IA
```

## 🚀 Instalação

Pré-requisitos: Python 3.13+

1. **Crie um ambiente virtual (recomendado):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_rl.txt
   ```

## 🎮 Como Usar

### 1. Jogar Manualmente (Humano)
Controle a raquete com as setas **Esquerda** e **Direita**.
```bash
python main.py
```

### 2. Treinar a Inteligência Artificial
Inicia o processo de aprendizado. O agente jogará milhares de partidas em velocidade acelerada.
*   **Para parar:** Pressione **'q'** na janela do jogo. O modelo será salvo automaticamente em `models/ppo_brickbreaker.zip`.
```bash
python train.py
```

### 3. Assistir a IA Jogar (Demo)
Carrega o modelo salvo e joga em velocidade normal (60 FPS), mostrando as probabilidades de decisão no terminal.
```bash
python demo.py
```

## ⚙️ Configuração

Todas as variáveis do jogo podem ser ajustadas em **`src/config.py`**:

*   **ENABLE_SOUND:** Habilitar/Desabilitar sons.
*   **SCREEN_WIDTH/HEIGHT:** Tamanho da janela.
*   **Reward Settings:** Ajuste de recompensas para o treino.
*   **Network Architecture:** Tamanho da rede neural da IA.

## 🐳 Docker

Para construir a imagem Docker:
```bash
docker build -t aibrickbreaker .
```

*Nota: Executar aplicações GUI (pygame) via Docker requer configuração de X11 Forwarding no host, o que varia por sistema operacional.*

## 📝 Notas da Versão 1.0
*   Implementação completa do Reward Shaping para aprendizado acelerado.
*   Modo de demonstração probabilístico para simular comportamento de treino.
*   Código refatorado e modularizado com `config.py`.
*   Documentação (docstrings) em Português.

---
**Autor:** Renato (via Gemini Agent) | **Data:** 27/11/2025