import torch
import numpy as np
from snack_game import SnakeGame, Direction, Point
from model import Linear_QNet
from config import device  # Use the same device (MPS or CPU)

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Player:
    def __init__(self):
        self.model = Linear_QNet(11, 256, 3)
        self.model.load('./model/model.pth')  # load trained weights
        self.model.to(device)
        self.model.eval()  # evaluation mode (no dropout/batchnorm)

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).to(device)
        prediction = self.model(state_tensor)
        move = torch.argmax(prediction).item()

        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

def play():
    player = Player()
    game = SnakeGame()

    # Infinite loop until snake dies
    while True:
        # Get the current state
        state_old = player.get_state(game)

        # Get move from trained model
        final_move = player.get_action(state_old)

        # Perform move and get game status
        reward, done, score = game.play_step(final_move)

        if done:
            print('Game Over. Score:', score)
            game.reset()

# Add the method to Player to get state from the game
def player_get_state(self, game):
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game.is_collision(point_r)) or
        (dir_l and game.is_collision(point_l)) or
        (dir_u and game.is_collision(point_u)) or
        (dir_d and game.is_collision(point_d)),

        # Danger right
        (dir_u and game.is_collision(point_r)) or
        (dir_d and game.is_collision(point_l)) or
        (dir_l and game.is_collision(point_u)) or
        (dir_r and game.is_collision(point_d)),

        # Danger left
        (dir_d and game.is_collision(point_r)) or
        (dir_u and game.is_collision(point_l)) or
        (dir_r and game.is_collision(point_u)) or
        (dir_l and game.is_collision(point_d)),

        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,

        # Food location
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y  # food down
    ]

    return np.array(state, dtype=int)

# Add the method dynamically to the Player class
Player.get_state = player_get_state

if __name__ == '__main__':
    play()
