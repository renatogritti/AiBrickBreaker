"""
-----------------------------------------------------------------------
Arquivo: src/game.py
Data: 27/11/2025
Versão: 1.0
Autor: Renato Gritti
Descrição:
    Contém a classe principal Game, que gerencia o loop do jogo, 
    estados (menu, jogando, game over), física, colisões e interface 
    com o agente de Reinforcement Learning (método step).
-----------------------------------------------------------------------
"""

import pygame
import numpy as np
import random
from src.config import *
from src.sprites import Paddle, Ball, Brick

class Game:
    """
    Gerencia a lógica principal do jogo Brick Breaker.
    """

    def __init__(self):
        """
        Inicializa o motor do Pygame, a janela, o relógio e os elementos do jogo.
        """
        pygame.init()
        
        if ENABLE_SOUND:
            pygame.mixer.init()
            self.generate_bip_sounds()
            
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(CAPTION)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        self.running = True
        self.game_over = False
        self.game_won = False
        self.level_complete = False

        # Grupos de Sprites
        self.all_sprites = pygame.sprite.Group()
        self.bricks = pygame.sprite.Group()
        self.balls = pygame.sprite.Group()
        self.paddle = Paddle()
        
        self.all_sprites.add(self.paddle)
        
        self.reset_game()

    def generate_bip_sounds(self):
        """
        Gera sons procedurais se o som estiver habilitado.
        (Atualmente desabilitado via config.py por padrão)
        """
        pass

    def reset_game(self):
        """
        Reinicia o jogo completo (Score 0, Vidas 3, Nível 1).
        """
        self.game_over = False
        self.score = 0
        self.lives = 3
        self.level = 1
        
        self.all_sprites.empty()
        self.bricks.empty()
        self.balls.empty()

        self.paddle = Paddle()
        self.ball = Ball()
        self.all_sprites.add(self.paddle, self.ball)
        self.balls.add(self.ball)
        
        self.create_bricks()
        self.reset_ball()

    def reset_ball(self):
        """
        Reposiciona a bola na raquete com parâmetros aleatórios para evitar
        repetição de cenários (importante para o treino de RL).
        """
        for ball in self.balls:
            ball.kill()
        self.balls.empty()

        self.ball = Ball()
        self.balls.add(self.ball)

        self.ball_launched = True
        
        # Aleatoriedade na posição inicial (offset do centro da raquete)
        # Limita para não sair da largura da raquete
        max_offset = (PADDLE_WIDTH // 2) - BALL_RADIUS
        random_offset = random.randint(-max_offset, max_offset)
        
        self.ball.rect.centerx = self.paddle.rect.centerx + random_offset
        self.ball.rect.bottom = self.paddle.rect.top
        
        # Aumento de velocidade progressivo por nível
        speed_multiplier = 1 + (self.level - 1) * BALL_SPEED_INCREASE
        
        # Direção aleatória (Esquerda ou Direita)
        direction_x = random.choice([-1, 1])
        
        # Magnitude da velocidade horizontal aleatória
        random_speed_x = random.uniform(BALL_RANDOM_SPEED_MIN, BALL_RANDOM_SPEED_MAX)
        
        self.ball.speed_x = random_speed_x * speed_multiplier * direction_x
        
        # Garante que a bola suba
        self.ball.speed_y = BALL_SPEED_Y_INITIAL * speed_multiplier

    def create_bricks(self):
        """
        Gera a matriz de tijolos para o nível atual.
        """
        self.bricks.empty()
        # Remove tijolos antigos do grupo geral, mas mantém paddle e ball
        for sprite in self.all_sprites:
            if isinstance(sprite, Brick):
                sprite.kill()

        colors = BRICK_COLORS
        rows = 5
        cols = 10
        
        for i in range(rows):
            for j in range(cols):
                is_special = False
                # Chance de tijolo especial a partir do nível 2
                if self.level >= 2 and random.random() < SPECIAL_BRICK_CHANCE:
                    is_special = True
                
                x = j * (BRICK_WIDTH + BRICK_GAP) + BRICK_OFFSET_LEFT
                y = i * (BRICK_HEIGHT + BRICK_GAP) + BRICK_OFFSET_TOP
                color = colors[i % len(colors)]
                
                brick = Brick(x, y, color, is_special=is_special)
                self.all_sprites.add(brick)
                self.bricks.add(brick)

    def run(self):
        """
        Loop principal para execução humana (main.py).
        """
        self.reset_game()
        while self.running:
            self.clock.tick(FPS_HUMAN)
            self.events()
            self.update()
            self.draw()
        pygame.quit()

    def step(self, action=None, fps=0):
        """
        Executa um único passo (frame) da simulação para o Agente de RL.

        Args:
            action (int, optional): Ação escolhida pelo agente.
            fps (int, optional): Limite de quadros. 0 para treino (máx speed), 
                                 60 para demo (tempo real).

        Returns:
            tuple: (estado, recompensa, done)
        """
        self.clock.tick(fps)
        self.events() # Processa a fila de eventos (ex: botão fechar)
        
        # Salva estado anterior para calcular delta de pontuação/vidas
        prev_score = self.score
        prev_lives = self.lives
        
        self.current_hit_paddle = False # Flag resetada a cada frame
        
        self.update(action)
        self.draw() # Desenha (necessário para o humano ver o que acontece na demo/treino)
        
        # Cálculo da Recompensa (Reward Function)
        reward = 0
        
        # 1. Reward Shaping: Incentivar seguir a bola
        # Calcula distância horizontal normalizada entre centros
        dist_x = abs(self.paddle.rect.centerx - self.ball.rect.centerx)
        max_dist = SCREEN_WIDTH
        norm_dist = dist_x / max_dist # 0 (perto) a 1 (longe)
        
        # Só recompensa se a bola estiver descendo (vindo em direção à raquete)
        if self.ball.speed_y > 0: 
            # Quanto menor a distância, maior a recompensa
            reward += REWARD_TRACKING_FACTOR * (1.0 - norm_dist)

        # 2. Recompensa por Pontuar (Quebrar Tijolo)
        if self.score > prev_score:
            reward += REWARD_HIT_BRICK
            
        # 3. Recompensa por Rebater na Raquete (Sobrevivência Ativa)
        if self.current_hit_paddle:
            reward += REWARD_HIT_PADDLE
            
        # 4. Penalidade por Perder Vida
        if self.lives < prev_lives:
            reward += REWARD_LOSE_LIFE # Valor negativo no config

        # Verifica condição de término
        done = self.lives == 0 or not self.running
        
        return self.get_state(), reward, done

    def get_state(self):
        """
        Constrói o vetor de observação do ambiente.
        
        Returns:
            np.array: [Paddle X, Ball X, Ball Y, Ball Vel X, Ball Vel Y] normalizados.
        """
        # Normalização simples (0 a 1 ou -1 a 1)
        p_x = self.paddle.rect.centerx / SCREEN_WIDTH
        b_x = self.ball.rect.centerx / SCREEN_WIDTH
        b_y = self.ball.rect.centery / SCREEN_HEIGHT
        
        # Velocidades normalizadas por um máximo estimado
        max_speed = 20.0
        b_vx = self.ball.speed_x / max_speed
        b_vy = self.ball.speed_y / max_speed
        
        return np.array([p_x, b_x, b_y, b_vx, b_vy], dtype=np.float32)

    def events(self):
        """
        Trata eventos de entrada do sistema (teclado, fechar janela).
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: # Tecla Q para sair
                    self.running = False

    def update(self, action=None):
        """
        Atualiza a lógica de jogo (movimento, colisão).
        """
        if self.game_over:
            return

        self.paddle.update(action)

        for ball in self.balls:
            ball.update()
            self.check_collisions(ball)

    def check_collisions(self, ball):
        """
        Gerencia física de colisão da bola com paredes, raquete e tijolos.
        """
        # Paredes Laterais
        if ball.rect.left <= 0 or ball.rect.right >= SCREEN_WIDTH:
            ball.speed_x *= -1
            # Som desabilitado
            
        # Teto
        if ball.rect.top <= 0:
            ball.speed_y *= -1
            # Som desabilitado

        # Raquete
        if pygame.sprite.collide_rect(ball, self.paddle):
            # 1. Deflexão baseada no ponto de impacto (Angle Deflection)
            # Calcula onde a bola bateu na raquete (-1 esquerda, 0 centro, 1 direita)
            relative_intersect_x = (self.paddle.rect.centerx - ball.rect.centerx)
            normalized_relative_intersection_x = relative_intersect_x / (PADDLE_WIDTH / 2)
            
            # Inverte direção Y (rebate)
            ball.speed_y *= -1
            
            # Muda a direção X baseada no ponto de impacto (efeito de "curva")
            # Quanto mais na ponta, mais horizontal a bola sai.
            # MAX_BOUNCE_ANGLE poderia ser aprox 75 graus (em radianos) ou fator linear
            bounce_factor = 5.0 # Fator de força lateral
            ball.speed_x = -normalized_relative_intersection_x * bounce_factor
            
            # 2. Transferência de Momento (Paddle Momentum)
            # Se a raquete estiver se movendo, adiciona velocidade à bola
            if hasattr(self.paddle, 'current_vel_x'):
                ball.speed_x += self.paddle.current_vel_x * 0.3 # 30% da velocidade da raquete
                
            # 3. Aceleração Dinâmica (Speed Variation)
            # Aumenta levemente a velocidade total a cada batida para tensão
            current_speed = np.sqrt(ball.speed_x**2 + ball.speed_y**2)
            new_speed = min(current_speed * 1.05, 12.0) # Aumenta 5%, max 12.0
            
            # Normaliza vetor e aplica nova velocidade
            speed_ratio = new_speed / current_speed
            ball.speed_x *= speed_ratio
            ball.speed_y *= speed_ratio
            
            # Garante componente Y mínima para a bola não ficar horizontal demais
            min_speed_y = 3.0
            if abs(ball.speed_y) < min_speed_y:
                 # Dá um "kick" vertical mantendo o sinal
                ball.speed_y = -min_speed_y if ball.speed_y < 0 else min_speed_y

            # Ajusta a bola para cima da raquete para evitar "grudar"
            ball.rect.bottom = self.paddle.rect.top
            self.current_hit_paddle = True 

        # Tijolos
        hits = pygame.sprite.spritecollide(ball, self.bricks, True)
        if hits:
            self.score += 10 # Pontuação fixa por tijolo (pode ir para config se desejar)
            ball.speed_y *= -1
            
        # Nível Concluído
        if not self.bricks:
            self.level += 1
            self.create_bricks()
            self.reset_ball()

        # Chão (Perde Vida)
        if ball.rect.top > SCREEN_HEIGHT:
            ball.kill()
            self.lives -= 1
            if self.lives > 0:
                self.reset_ball()
            else:
                self.lives = 0
                self.reset_game()

    def draw(self):
        """
        Renderiza o estado atual na tela.
        """
        self.screen.fill(BLACK)
        self.all_sprites.draw(self.screen)
        self.balls.draw(self.screen)
        
        # HUD (Head-Up Display)
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        lives_text = self.font.render(f"Lives: {self.lives}", True, WHITE)
        level_text = self.font.render(f"Level: {self.level}", True, WHITE)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (SCREEN_WIDTH - 120, 10))
        self.screen.blit(level_text, (SCREEN_WIDTH // 2 - 50, 10))

        if self.game_over:
            self.draw_game_over()

        pygame.display.flip()

    def draw_game_over(self):
        """
        Desenha a tela de fim de jogo.
        """
        game_over_font = pygame.font.Font(None, 72)
        
        text_surface = game_over_font.render("GAME OVER", True, RED)
        score_surface = self.font.render(f"Final Score: {self.score}", True, WHITE)
        
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        
        self.screen.blit(text_surface, (cx - text_surface.get_width() // 2, cy - 50))
        self.screen.blit(score_surface, (cx - score_surface.get_width() // 2, cy + 50))
