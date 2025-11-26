"""
Este é o ponto de entrada principal para o jogo Brick Breaker.
Ele inicializa e executa a lógica do jogo contida na classe `Game`.
"""

from src.game import Game

if __name__ == "__main__":
    # Cria uma instância do jogo e o executa.
    game = Game()
    game.run()
