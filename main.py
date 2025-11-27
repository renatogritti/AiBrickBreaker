"""
-----------------------------------------------------------------------
Arquivo: main.py
Data: 27/11/2025
Versão: 1.0
Autor: Renato Gritti
Descrição:
    Ponto de entrada para jogar o Brick Breaker manualmente (humano).
-----------------------------------------------------------------------
"""

from src.game import Game

if __name__ == "__main__":
    # Cria uma instância do jogo e o executa no loop principal.
    game = Game()
    game.run()