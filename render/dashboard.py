import pygame
import time


class TrainingDashboard:
    def __init__(self, env):
        self.env = env

        pygame.init()
        self.width, self.height = 900, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("SafeCity Training Dashboard")

        self.font_head = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_data = pygame.font.SysFont("Consolas", 18)
        self.font_emerg = pygame.font.SysFont("Arial", 20, bold=True)

    def draw(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

        self.screen.fill((40, 44, 52))

        # --- HEADER ---
        pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.width, 60))
        title = self.font_head.render(
            f"TRAINING IN CORSO - Auto Finite: {self.env.cars_finished}",
            True, (255, 255, 255)
        )
        self.screen.blit(title, (20, 20))

        start_y = 80
        col_width = self.width // 2
        now = time.time()

        for i, inc in enumerate(self.env.manager.intersections):
            x = 20 + i * col_width
            y = start_y

            pygame.draw.rect(
                self.screen, (60, 63, 65),
                (x, y, col_width - 40, 450),
                border_radius=10
            )

            phase = inc.phases[inc.current_phase_idx]
            head = self.font_head.render(
                f"INCROCIO {inc.id + 1} (Fase: {phase})",
                True, (255, 255, 255)
            )
            self.screen.blit(head, (x + 20, y + 15))

            y += 60

            dirs = ['N', 'S', 'E', 'O']
            colors = {
                'N': (0, 255, 0),
                'S': (255, 50, 50),
                'E': (100, 100, 255),
                'O': (255, 255, 0)
            }

            for d in dirs:
                q = inc.current_queues.get(d, 0)
                w = inc.current_waits.get(d, 0.0)
                emerg = inc.current_emergencies.get(d, False)

                bg = (100, 0, 0) if emerg and int(now * 4) % 2 == 0 else (50, 50, 50)
                pygame.draw.rect(self.screen, bg, (x + 15, y, col_width - 70, 80), 5)

                label = self.font_head.render(d, True, colors[d])
                self.screen.blit(label, (x + 25, y + 10))

                self.screen.blit(
                    self.font_data.render(f"Code: {q}", True, (255, 255, 255)),
                    (x + 25, y + 40)
                )

                self.screen.blit(
                    self.font_data.render(f"Wait: {w:.1f}s", True, (200, 200, 200)),
                    (x + 150, y + 40)
                )

                if emerg:
                    e = self.font_emerg.render("EMERGENZA", True, (255, 255, 255))
                    self.screen.blit(e, (x + 150, y + 10))

                y += 90

        pygame.display.flip()

    def close(self):
        pygame.quit()
