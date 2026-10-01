import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BrickBreakerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        # Game dimensions
        self.WIDTH = 600
        self.HEIGHT = 400

        # Paddle configuration
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 10
        self.PADDLE_SPEED = 8
        self.PADDLE_Y = self.HEIGHT - 30

        # Ball configuration
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 5

        # Brick configuration
        self.BRICK_ROWS = 4
        self.BRICK_COLS = 8
        self.BRICK_WIDTH = 60
        self.BRICK_HEIGHT = 20
        self.BRICK_GAP = 10
        self.BRICK_OFFSET_TOP = 40
        self.BRICK_OFFSET_LEFT = (self.WIDTH - (self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_GAP))) // 2

        # Total number of bricks
        self.TOTAL_BRICKS = self.BRICK_ROWS * self.BRICK_COLS

        # Number of lives
        self.INITIAL_LIVES = 3

        # Action space: 0=Stay, 1=Left, 2=Right
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # [paddle_x, ball_x, ball_y, ball_dx, ball_dy, remaining_bricks, lives]
        low = np.array(
            [
                0.0,  # paddle_x min
                0.0,  # ball_x min
                0.0,  # ball_y min
                -20.0,  # ball_dx min
                -20.0,  # ball_dy min
                0.0,  # remaining_bricks min
                0.0,  # lives min
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                self.WIDTH,  # paddle_x max
                self.WIDTH,  # ball_x max
                self.HEIGHT,  # ball_y max
                20.0,  # ball_dx max
                20.0,  # ball_dy max
                self.TOTAL_BRICKS,  # remaining_bricks max
                self.INITIAL_LIVES,  # lives max
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, shape=(7,), dtype=np.float32)

        self.render_mode = render_mode

        self._init_game_state()

    def _init_game_state(self):
        # Paddle starts centered
        self.paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2.0

        # Ball starts on paddle, moving upward-right
        self.ball_x = self.WIDTH / 2.0
        self.ball_y = self.PADDLE_Y - self.BALL_RADIUS - 1
        self.ball_dx = self.BALL_SPEED
        self.ball_dy = -self.BALL_SPEED

        # Create brick grid: 1 = alive, 0 = broken
        self.bricks = np.ones((self.BRICK_ROWS, self.BRICK_COLS), dtype=np.int32)

        # Score, lives, remaining bricks
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.remaining_bricks = self.TOTAL_BRICKS

    def _get_observation(self):
        return np.array(
            [
                self.paddle_x,
                self.ball_x,
                self.ball_y,
                self.ball_dx,
                self.ball_dy,
                float(self.remaining_bricks),
                float(self.lives),
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        return {
            "score": self.score,
            "lives": self.lives,
            "remaining_bricks": self.remaining_bricks,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_game_state()
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        # Apply action to paddle
        if action == 1:  # Move Left
            self.paddle_x -= self.PADDLE_SPEED
        elif action == 2:  # Move Right
            self.paddle_x += self.PADDLE_SPEED

        # Clamp paddle to screen bounds
        self.paddle_x = np.clip(self.paddle_x, 0, self.WIDTH - self.PADDLE_WIDTH)

        # Ball hits wall -> survive
        prev_ball_y = self.ball_y
        prev_ball_x = self.ball_x

        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Wall collisions (left / right)
        if self.ball_x - self.BALL_RADIUS < 0:
            self.ball_x = self.BALL_RADIUS
            self.ball_dx *= -1
        if self.ball_x + self.BALL_RADIUS > self.WIDTH:
            self.ball_x = self.WIDTH - self.BALL_RADIUS
            self.ball_dx *= -1

        # Wall collision (top)
        if self.ball_y - self.BALL_RADIUS < 0:
            self.ball_y = self.BALL_RADIUS
            self.ball_dy *= -1

        # Paddle collision check
        paddle_left = self.paddle_x
        paddle_right = self.paddle_x + self.PADDLE_WIDTH
        paddle_top = self.PADDLE_Y
        paddle_bottom = self.PADDLE_Y + self.PADDLE_HEIGHT

        ball_in_paddle_x = paddle_left <= self.ball_x <= paddle_right
        ball_in_paddle_y = paddle_top - self.BALL_RADIUS <= self.ball_y <= paddle_bottom

        was_above = prev_ball_y + self.BALL_RADIUS <= paddle_top
        is_inside = ball_in_paddle_x and ball_in_paddle_y and was_above

        if is_inside:
            self.ball_y = paddle_top - self.BALL_RADIUS
            self.ball_dy = -abs(self.ball_dy)
            # Add variation to ball_dx based on where it hit the paddle
            hit_pos = (self.ball_x - paddle_left) / self.PADDLE_WIDTH  # 0..1
            self.ball_dx = (hit_pos - 0.5) * 2.0 * self.BALL_SPEED
            reward += 0.1  # Reward for hitting ball with paddle

        # Brick collisions
        for row in range(self.BRICK_ROWS):
            for col in range(self.BRICK_COLS):
                if self.bricks[row, col] == 0:
                    continue
                bx = self.BRICK_OFFSET_LEFT + col * (self.BRICK_WIDTH + self.BRICK_GAP)
                by = self.BRICK_OFFSET_TOP + row * (self.BRICK_HEIGHT + self.BRICK_GAP)
                bw = self.BRICK_WIDTH
                bh = self.BRICK_HEIGHT

                # Check AABB collision between ball and brick
                closest_x = np.clip(self.ball_x, bx, bx + bw)
                closest_y = np.clip(self.ball_y, by, by + bh)
                dx = self.ball_x - closest_x
                dy = self.ball_y - closest_y
                if (dx * dx + dy * dy) <= (self.BALL_RADIUS * self.BALL_RADIUS):
                    self.bricks[row, col] = 0
                    self.remaining_bricks -= 1
                    self.score += 10
                    reward += 10.0  # Reward for breaking a brick
                    # Determine bounce direction
                    overlap_left = (self.ball_x + self.BALL_RADIUS) - bx
                    overlap_right = (bx + bw) - (self.ball_x - self.BALL_RADIUS)
                    overlap_top = (self.ball_y + self.BALL_RADIUS) - by
                    overlap_bottom = (by + bh) - (self.ball_y - self.BALL_RADIUS)
                    min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)
                    if min_overlap == overlap_top or min_overlap == overlap_bottom:
                        self.ball_dy *= -1
                    else:
                        self.ball_dx *= -1
                    break  # Only break one brick per frame

        # Check win condition: all bricks broken
        if self.remaining_bricks == 0:
            terminated = True

        # Ball falls below screen (lose life)
        if self.ball_y - self.BALL_RADIUS > self.HEIGHT:
            self.lives -= 1
            if self.lives <= 0:
                reward += -100.0  # Penalty for Game Over
                terminated = True
            else:
                reward += -10.0  # Penalty for losing a life
                # Reset ball to paddle
                self.ball_x = self.paddle_x + self.PADDLE_WIDTH / 2.0
                self.ball_y = self.PADDLE_Y - self.BALL_RADIUS - 1
                self.ball_dx = self.BALL_SPEED
                self.ball_dy = -self.BALL_SPEED

        # Small survival reward each step (if game not over)
        if not terminated:
            reward += 0.01  # Tiny reward for staying alive

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def get_render_dict(self):
        """
        Returns a serializable dict of the current game state for rendering in the browser.
        """
        bricks_list = []
        for row in range(self.BRICK_ROWS):
            for col in range(self.BRICK_COLS):
                if self.bricks[row, col] == 1:
                    bx = self.BRICK_OFFSET_LEFT + col * (self.BRICK_WIDTH + self.BRICK_GAP)
                    by = self.BRICK_OFFSET_TOP + row * (self.BRICK_HEIGHT + self.BRICK_GAP)
                    bricks_list.append(
                        {
                            "x": float(bx),
                            "y": float(by),
                            "w": float(self.BRICK_WIDTH),
                            "h": float(self.BRICK_HEIGHT),
                            "row": int(row),
                            "col": int(col),
                        }
                    )

        return {
            "width": self.WIDTH,
            "height": self.HEIGHT,
            "paddle": {
                "x": float(self.paddle_x),
                "y": float(self.PADDLE_Y),
                "w": float(self.PADDLE_WIDTH),
                "h": float(self.PADDLE_HEIGHT),
            },
            "ball": {
                "x": float(self.ball_x),
                "y": float(self.ball_y),
                "r": float(self.BALL_RADIUS),
            },
            "bricks": bricks_list,
            "score": int(self.score),
            "lives": int(self.lives),
            "remaining_bricks": int(self.remaining_bricks),
        }
