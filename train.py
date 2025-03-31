import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque

# Tic-Tac-Toe Game
class TicTacToe:
    def __init__(self, size=3):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # Initialize an empty board

    def is_winner(self, player):
        return (
            any(np.all(row == player) for row in self.board) or
            any(np.all(col == player) for col in self.board.T) or
            np.all(np.diag(self.board) == player) or
            np.all(np.diag(np.fliplr(self.board)) == player)
        )

    def is_draw(self):
        return not np.any(self.board == 0) and not self.is_winner(1) and not self.is_winner(-1)

    def get_available_moves(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]

    def make_move(self, x, y, player):
        if self.board[x, y] == 0:
            self.board[x, y] = player
            return True
        return False

    def minimax(self, is_maximizing):
        if self.is_winner(1):
            return 1
        if self.is_winner(-1):
            return -1
        if self.is_draw():
            return 0
        
        best_score = -np.inf if is_maximizing else np.inf
        player = 1 if is_maximizing else -1

        for i, j in self.get_available_moves():
            self.board[i, j] = player
            score = self.minimax(not is_maximizing)
            self.board[i, j] = 0
            best_score = max(best_score, score) if is_maximizing else min(best_score, score)
        
        return best_score

    def best_move(self):
        best_score = -np.inf
        move = None
        for i, j in self.get_available_moves():
            self.board[i, j] = 1
            score = self.minimax(False)
            self.board[i, j] = 0
            if score > best_score:
                best_score = score
                move = (i, j)
        return move

    def print_board(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()

# DQN Model for 5x5
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training setup
class DQNTrainer:
    def __init__(self, state_dim, action_dim, model_path="dqn_model.pth"):
        self.GAMMA = 0.99
        self.LR = 0.001
        self.EPSILON_START = 1.0
        self.EPSILON_END = 0.01
        self.EPSILON_DECAY = 0.995
        self.BATCH_SIZE = 64
        self.MEMORY_SIZE = 10000
        self.TARGET_UPDATE = 10
        self.MODEL_PATH = model_path
        
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.epsilon = self.EPSILON_START

        self.wins = 0
        self.losses = 0
        self.draws = 0

        self.load_model()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 24)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                return self.policy_net(state_tensor).argmax().item()

    def train_step(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = random.sample(self.memory, self.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + self.GAMMA * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_transition(self, *args):
        self.memory.append(args)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.MODEL_PATH)
        print(f"Model saved to {self.MODEL_PATH}")

    def load_model(self):
        if os.path.exists(self.MODEL_PATH):
            self.policy_net.load_state_dict(torch.load(self.MODEL_PATH))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded from {self.MODEL_PATH}")

# Play with AI using Minimax for 3x3
def play_with_minimax():
    game = TicTacToe(size=3)
    while not game.is_draw() and not game.is_winner(1) and not game.is_winner(-1):
        game.print_board()
        x, y = map(int, input("Enter your move (format x y): ").split())
        if game.make_move(x, y, -1):
            if game.is_winner(-1):
                game.print_board()
                print("You win!")
                break
            elif game.is_draw():
                game.print_board()
                print("It's a draw!")
                break
            
            move = game.best_move()
            if move:
                game.make_move(move[0], move[1], 1)
                print(f"AI chose move: {move}")
            
            if game.is_winner(1):
                game.print_board()
                print("AI wins!")
                break
        else:
            print("Invalid move. Try again!")

# Training loop for 5x5 Tic-Tac-Toe
def train_dqn():
    trainer = DQNTrainer(state_dim=25, action_dim=25)
    episode = 0

    while True:
        game = TicTacToe(size=5)
        state = np.zeros(25)
        done = False
        total_reward = 0

        while not done:
            action = trainer.select_action(state)
            x, y = divmod(action, 5)  # Convert action to 2D coordinates
            if game.make_move(x, y, 1):  # AI plays as '1'
                if game.is_winner(1):
                    reward = 1
                    done = True
                elif game.is_draw():
                    reward = 0
                    done = True
                else:
                    reward = 0  # Continue the game
                    next_state = state.copy()
                    next_state[action] = 1
            else:
                reward = -1  # Invalid move
                done = True
                next_state = state.copy()

            total_reward += reward
            trainer.store_transition(state, action, reward, next_state, done)
            state = next_state
            trainer.train_step()

        if reward == 1:
            trainer.wins += 1
        elif reward == -1:
            trainer.losses += 1
        else:
            trainer.draws += 1
        
        if episode % trainer.TARGET_UPDATE == 0:
            trainer.update_target_network()
        
        if episode % 1000 == 0:
            trainer.save_model()
        
        trainer.epsilon = max(trainer.EPSILON_END, trainer.epsilon * trainer.EPSILON_DECAY)
        
        if episode % 10000 == 0:
            total_games = trainer.wins + trainer.losses + trainer.draws
            win_rate = trainer.wins / total_games if total_games > 0 else 0
            loss_rate = trainer.losses / total_games if total_games > 0 else 0
            draw_rate = trainer.draws / total_games if total_games > 0 else 0
            
            print(f"Episode {episode}: Wins: {trainer.wins}, Losses: {trainer.losses}, Draws: {trainer.draws}, Win Rate: {win_rate:.2%}, Loss Rate: {loss_rate:.2%}, Draw Rate: {draw_rate:.2%}, Epsilon: {trainer.epsilon:.4f}")
        
        episode += 1

# Uncomment the following line to play with the Minimax AI in 3x3 mode
# play_with_minimax()

# Uncomment the following line to start training the DQN for 5x5 mode
train_dqn()



