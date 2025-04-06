import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# --------------------------------------
# Class TicTacToe - Lớp quản lý trò chơi Tic-Tac-Toe
# --------------------------------------
class TicTacToe:
    def __init__(self, size=3):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)

    def is_winner(self, player):
        win_length = 3 if self.size == 3 else 4 if self.size == 5 else 5
        for i in range(self.size):
            for j in range(self.size - win_length + 1):
                if np.all(self.board[i, j:j + win_length] == player): return True
                if np.all(self.board[j:j + win_length, i] == player): return True
        for i in range(self.size - win_length + 1):
            for j in range(self.size - win_length + 1):
                if np.all(np.diag(self.board[i:i + win_length, j:j + win_length]) == player): return True
                if np.all(np.diag(np.fliplr(self.board[i:i + win_length, j:j + win_length])) == player): return True
        return False

    def is_draw(self):
        return not np.any(self.board == 0) and not self.is_winner(1) and not self.is_winner(-1)

    def get_available_moves(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]

    def make_move(self, x, y, player):
        if self.board[x, y] == 0:
            self.board[x, y] = player
            return True
        return False

    def print_board(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()

    def easy_move(self):
        return random.choice(self.get_available_moves())

    def medium_move(self, player):
        """Chế độ trung bình: chặn hoặc thắng thông minh hơn"""
        opponent = -player
        best_move = None

        # Kiểm tra xem có thể thắng không
        for i, j in self.get_available_moves():
            self.board[i, j] = player
            if self.is_winner(player):
                self.board[i, j] = 0
                return (i, j)
            self.board[i, j] = 0

        # Kiểm tra xem đối thủ có thể thắng không, nếu có thì chặn
        for i, j in self.get_available_moves():
            self.board[i, j] = opponent
            if self.is_winner(opponent):
                self.board[i, j] = 0
                return (i, j)
            self.board[i, j] = 0

        # Kiểm tra những nước đi có thể chặn chuỗi dài của đối thủ (2 hoặc 3 quân liên tiếp)
        for i, j in self.get_available_moves():
            block_move = self.can_block(i, j, opponent)
            if block_move:
                print(f"Blocking move at {block_move} to stop opponent from winning!")
                return block_move

        # Dùng alpha-beta depth = 2
        _, move = self.alpha_beta_move(depth=2, is_maximizing_player=(player == -1))
        return move

    def can_block(self, x, y, player):
        """Kiểm tra xem nếu AI (hoặc đối thủ) có thể tạo thành một chuỗi quân đủ dài (3 quân cho 5x5, 4 quân cho 7x7)"""
        if self.size == 5:
            win_length = 3  # Đối với bàn cờ 5x5, đối thủ có thể thắng với 3 quân liên tiếp
        elif self.size == 7:
            win_length = 4  # Đối với bàn cờ 7x7, đối thủ có thể thắng với 4 quân liên tiếp
        else:
            return None  # Chế độ không có hiệu lực đối với các bàn cờ khác

        # Các hướng kiểm tra (hàng, cột, chéo)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

        for dx, dy in directions:
            count = 0  # Đếm quân của đối thủ
            potential_block = None  # Vị trí cần chặn
            blocked = False  # Biến kiểm tra xem có thể chặn được hay không

            # Kiểm tra phía trước và sau
            for i in range(1, win_length + 1):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx, ny] == player:
                        count += 1
                    elif self.board[nx, ny] == 0:  # Vị trí trống, có thể là chỗ cần chặn
                        potential_block = (nx, ny)
                    else:
                        break
                else:
                    break

            # Kiểm tra phía sau (ngược hướng)
            for i in range(1, win_length + 1):
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx, ny] == player:
                        count += 1
                    elif self.board[nx, ny] == 0:  # Vị trí trống, có thể là chỗ cần chặn
                        potential_block = (nx, ny)
                    else:
                        break
                else:
                    break

            # Kiểm tra xem nếu có thể tạo thành chuỗi liên tiếp bằng cách nối quân vào các vị trí trống
            if count >= win_length - 1 and potential_block:
                return potential_block
            
            # Kiểm tra các trường hợp như "chuỗi quân liên tiếp có thể nối được"
            # Đây là phần kiểm tra sâu hơn các vị trí trống ở giữa các quân đối thủ
            # Tạo lại kiểm tra nếu có 3 quân đối thủ và 1 ô trống
            if count == win_length - 1:  # Trường hợp có 3 quân đối thủ và 1 ô trống giữa
                if potential_block:
                    return potential_block

        return None  # Không tìm thấy vị trí cần chặn

    def evaluate_board(self):
        # Nếu AI thắng, trả về điểm cao, nếu người chơi thắng, trả về điểm thấp
        if self.is_winner(-1):
            return 10
        elif self.is_winner(1):
            return -10
        
        score = 0

        # Đánh giá các dòng
        for row in self.board:
            row_list = row.tolist()  # Chuyển đổi mảng numpy thành list
            if row_list.count(-1) == 2 and row_list.count(0) == 1:  # AI có thể thắng
                score += 1
            if row_list.count(1) == 2 and row_list.count(0) == 1:   # Người chơi có thể thắng
                score -= 1

        # Đánh giá các cột
        for col in range(self.size):
            column = self.board[:, col]
            column_list = column.tolist()  # Chuyển đổi mảng numpy thành list
            if column_list.count(-1) == 2 and column_list.count(0) == 1:
                score += 1
            if column_list.count(1) == 2 and column_list.count(0) == 1:
                score -= 1

        # Đánh giá các chéo
        diagonal1 = [self.board[i, i] for i in range(self.size)]
        diagonal2 = [self.board[i, self.size - 1 - i] for i in range(self.size)]
        
        if diagonal1.count(-1) == 2 and diagonal1.count(0) == 1:
            score += 1
        if diagonal1.count(1) == 2 and diagonal1.count(0) == 1:
            score -= 1

        if diagonal2.count(-1) == 2 and diagonal2.count(0) == 1:
            score += 1
        if diagonal2.count(1) == 2 and diagonal2.count(0) == 1:
            score -= 1

        return score

    def alpha_beta_move(self, depth, is_maximizing_player, alpha=-np.inf, beta=np.inf):
        if depth == 0 or self.is_winner(1) or self.is_winner(-1) or self.is_draw():
            return self.evaluate_board(), None

        best_move = None
        if is_maximizing_player:
            max_eval = -np.inf
            for i, j in self.get_available_moves():
                self.board[i, j] = -1
                eval, _ = self.alpha_beta_move(depth-1, False, alpha, beta)
                self.board[i, j] = 0
                if eval > max_eval:
                    max_eval = eval
                    best_move = (i, j)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = np.inf
            for i, j in self.get_available_moves():
                self.board[i, j] = 1
                eval, _ = self.alpha_beta_move(depth-1, True, alpha, beta)
                self.board[i, j] = 0
                if eval < min_eval:
                    min_eval = eval
                    best_move = (i, j)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def hard_move(self):
        if self.size == 3:
            return self.alpha_beta_move(depth=9, is_maximizing_player=True)[1]
        else:
            return self.dqn_move()


# --------------------------------------
# Class TicTacToeAI - mở rộng thêm DQN
# --------------------------------------
class TicTacToeAI(TicTacToe):
    def __init__(self, size=3):
        super().__init__(size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = self.create_model()
        self.target_net = self.create_model()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99

    def create_model(self):
        model = nn.Sequential(
            nn.Linear(self.size * self.size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.size * self.size)
        )
        return model.to(self.device)

    def dqn_move(self):
        state = self.board.flatten().astype(np.float32)
        state = torch.tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)

        available = self.get_available_moves()
        if random.random() < 0.1:
            return random.choice(available)

        flat_available = [i * self.size + j for i, j in available]
        q_values_np = q_values.cpu().numpy().flatten()
        best_flat = max(flat_available, key=lambda idx: q_values_np[idx])
        return divmod(best_flat, self.size)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(state)
        next_q_values = self.target_net(next_state)
        target = q_values.clone()

        for i in range(self.batch_size):
            if done[i]:
                target[i, action[i]] = reward[i]
            else:
                best_action = torch.argmax(self.policy_net(next_state[i:i+1])).item()
                target[i, action[i]] = reward[i] + self.gamma * next_q_values[i, best_action]

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, board_size):
        torch.save(self.policy_net.state_dict(), f"dqn_{board_size}x{board_size}.pth")


# --------------------------------------
# Algorithm: xử lý trận đấu
# --------------------------------------
class Algorithm:
    @staticmethod
    def play_game(size, mode):
        game = TicTacToeAI(size=size)
        current_player = 1

        print(f"\nStarting game on {size}x{size} board in {mode} mode...\n")
        game.print_board()

        while True:
            if current_player == 1:
                try:
                    move = input("Your move (row col): ")
                    x, y = map(int, move.strip().split())
                    if not game.make_move(x, y, 1):
                        print("Invalid move. Try again.")
                        continue
                except:
                    print("Invalid input format. Use: row col")
                    continue
            else:
                if mode == "easy":
                    move = game.easy_move()
                elif mode == "medium":
                    move = game.medium_move(-1)
                elif mode == "hard":
                    move = game.hard_move()
                game.make_move(move[0], move[1], -1)
                print(f"AI moves at: {move}")

            game.print_board()

            if game.is_winner(current_player):
                print("You win!" if current_player == 1 else "AI wins!")
                break
            elif game.is_draw():
                print("It's a draw!")
                break

            current_player *= -1

# --------------------------------------
# Main
# --------------------------------------
class Main:
    @staticmethod
    def main():
        mode_choice = input("Choose mode (1: Play, 2: Training): ")

        if mode_choice == "1":
            size = int(input("Choose board size (3, 5, 7): "))
            difficulty = input("Choose difficulty (1: Easy, 2: Medium, 3: Hard): ")
            mode = {"1": "easy", "2": "medium", "3": "hard"}.get(difficulty)

            if mode:
                Algorithm.play_game(size, mode)
            else:
                print("Invalid choice!")
        elif mode_choice == "2":
            size = int(input("Choose board size (5, 7): "))
            if size not in [5, 7]:
                print("Training mode is only available for 5x5 and 7x7 boards.")
            else:
                print("Training mode...")
                Algorithm.play_game(size, "hard")
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    Main.main()
