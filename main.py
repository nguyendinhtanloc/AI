import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
from copy import deepcopy

class TicTacToe:
    """Lớp quản lý trò chơi Tic-Tac-Toe, bao gồm logic bàn cờ và kiểm tra thắng/thua/hòa."""
    
    def __init__(self, size=3):
        """Khởi tạo bàn cờ với kích thước size x size."""
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)

    def is_winner(self, player):
        """Kiểm tra xem người chơi (1: X, -1: O) có thắng không."""
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
        """Kiểm tra hòa: bàn cờ đầy và không ai thắng."""
        return not np.any(self.board == 0) and not self.is_winner(1) and not self.is_winner(-1)

    def get_available_moves(self):
        """Trả về danh sách các ô trống (nước đi hợp lệ) dưới dạng [(x, y)]."""
        indices = np.where(self.board == 0)
        return list(zip(indices[0], indices[1]))

    def make_move(self, x, y, player):
        """Thực hiện nước đi tại (x, y) cho người chơi (1: X, -1: O)."""
        if self.board[x, y] == 0:
            self.board[x, y] = player
            return True
        return False

    def print_board(self):
        """In bàn cờ với ký hiệu: . (trống), X (người chơi), O (AI)."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()

    def easy_move(self):
        """Chọn ngẫu nhiên một nước đi từ các ô trống (chế độ dễ)."""
        return random.choice(self.get_available_moves())

    def medium_move(self, player):
        """Chọn nước đi thông minh hơn (chế độ trung bình): thắng, chặn, hoặc tạo chuỗi."""
        opponent = -player
        best_move = None
        for i, j in self.get_available_moves():
            self.board[i, j] = player
            if self.is_winner(player):
                self.board[i, j] = 0
                return (i, j)
            self.board[i, j] = 0
        for i, j in self.get_available_moves():
            self.board[i, j] = opponent
            if self.is_winner(opponent):
                self.board[i, j] = 0
                return (i, j)
            self.board[i, j] = 0
        for i, j in self.get_available_moves():
            if self.can_extend_chain(i, j, player):
                return (i, j)
        for i, j in self.get_available_moves():
            block_move = self.can_block(i, j, opponent)
            if block_move:
                print(f"Blocking move at {block_move} to stop opponent from winning!")
                return block_move
        _, move = self.alpha_beta_move(depth=2, is_maximizing_player=(player == -1))
        return move

    def can_extend_chain(self, x, y, player):
        """Kiểm tra xem đặt tại (x, y) có tạo chuỗi 3 (5x5) hoặc 4 (7x7) quân không."""
        if self.board[x, y] != 0:
            return False
        target_length = 3 if self.size == 5 else 4
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]
        for dx, dy in directions:
            count = 1
            for dir in [1, -1]:
                for i in range(1, target_length):
                    nx = x + dir * i * dx
                    ny = y + dir * i * dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                        count += 1
                    else:
                        break
            if count >= target_length:
                return True
            nx1, ny1 = x - dx, y - dy
            nx2, ny2 = x + dx, y + dy
            if all(0 <= nx < self.size and 0 <= ny < self.size for nx, ny in [(nx1, ny1), (nx2, ny2)]):
                if self.board[nx1, ny1] == player and self.board[nx2, ny2] == player:
                    if target_length <= 3:
                        return True
                    else:
                        nx3 = x + 2 * dx
                        ny3 = y + 2 * dy
                        if 0 <= nx3 < self.size and 0 <= ny3 < self.size and self.board[nx3, ny3] == player:
                            return True
                        nx3 = x - 2 * dx
                        ny3 = y - 2 * dy
                        if 0 <= nx3 < self.size and 0 <= ny3 < self.size and self.board[nx3, ny3] == player:
                            return True
        return False

    def can_block(self, x, y, player):
        """Kiểm tra xem đặt tại (x, y) có chặn được chuỗi nguy hiểm của đối PROBLEMS OUTPUT TERMINAL thủ không."""
        win_length = 3 if self.size == 5 else 4
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        for dx, dy in directions:
            count = 0
            potential_block = None
            for i in range(1, win_length + 1):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx, ny] == player:
                        count += 1
                    elif self.board[nx, ny] == 0:
                        potential_block = (nx, ny)
                    else:
                        break
                else:
                    break
            for i in range(1, win_length + 1):
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx, ny] == player:
                        count += 1
                    elif self.board[nx, ny] == 0:
                        potential_block = (nx, ny)
                    else:
                        break
                else:
                    break
            if count >= win_length - 1 and potential_block:
                return potential_block
            if count == win_length - 1:
                if potential_block:
                    return potential_block
        return None

    def evaluate_board(self):
        """Đánh giá bàn cờ: +10 nếu AI thắng, -10 nếu người chơi thắng, hoặc điểm dựa trên chuỗi."""
        if self.is_winner(-1):
            return 10
        elif self.is_winner(1):
            return -10
        score = 0
        for row in self.board:
            row_list = row.tolist()
            if row_list.count(-1) == 2 and row_list.count(0) == 1:
                score += 1
            if row_list.count(1) == 2 and row_list.count(0) == 1:
                score -= 1
        for col in range(self.size):
            column = self.board[:, col]
            column_list = column.tolist()
            if column_list.count(-1) == 2 and column_list.count(0) == 1:
                score += 1
            if column_list.count(1) == 2 and column_list.count(0) == 1:
                score -= 1
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
        """Tìm nước đi tối ưu bằng thuật toán alpha-beta pruning."""
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

    def hard_move(self, player):
        """Chế độ khó: Sử dụng DQN hoặc kiểm tra thắng/chặn trực tiếp."""
        opponent = -player
        best_move = None
        for i, j in self.get_available_moves():
            self.board[i, j] = player
            if self.is_winner(player):
                self.board[i, j] = 0
                return (i, j)
            self.board[i, j] = 0
        for i, j in self.get_available_moves():
            self.board[i, j] = opponent
            if self.is_winner(opponent):
                self.board[i, j] = 0
                return (i, j)
            self.board[i, j] = 0
        for i, j in self.get_available_moves():
            if self.can_extend_chain(i, j, player):
                return (i, j)
        for i, j in self.get_available_moves():
            block_move = self.can_block(i, j, opponent)
            if block_move:
                print(f"Blocking move at {block_move} to stop opponent from winning!")
                return block_move
        return self.dqn_move(episode=100000)

class TicTacToeAI(TicTacToe):
    """Lớp mở rộng TicTacToe với DQN, sử dụng CNN và các kỹ thuật học tăng cường tiên tiến."""
    
    def __init__(self, size=3):
        """Khởi tạo AI với mô hình CNN cải tiến, bộ nhớ, và các tham số huấn luyện."""
        super().__init__(size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = self.create_model(size).to(self.device)
        self.target_net = self.create_model(size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30000, gamma=0.1)
        self.memory = []
        self.elite_memory = deque(maxlen=10000)
        self.memory_capacity = 50000
        self.priorities = []
        self.elite_priorities = deque(maxlen=10000)
        self.batch_size = 256
        self.gamma = 0.99
        self.per_epsilon = 1e-6
        self.per_alpha = 0.6

    def create_model(self, size):
        """Tạo mô hình CNN cải tiến với thêm lớp convolution, batch norm, và dropout."""
        class DQN(nn.Module):
            def __init__(self, size):
                super(DQN, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm2d(128)
                self.fc1 = nn.Linear(128 * size * size, 512)
                self.dropout = nn.Dropout(0.3)
                self.fc2 = nn.Linear(512, size * size)

            def forward(self, x):
                x = x.view(-1, 1, size, size)
                x = torch.relu(self.bn1(self.conv1(x)))
                x = torch.relu(self.bn2(self.conv2(x)))
                x = torch.relu(self.bn3(self.conv3(x)))
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        return DQN(size)

    def augment_state(self, state, full_augmentation=False):
        """Tạo các phiên bản đối xứng, giới hạn 2 phiên bản ngẫu nhiên nếu không quan trọng."""
        states = [state]
        if full_augmentation:
            states.extend([
                np.rot90(state, k=1),
                np.rot90(state, k=2),
                np.rot90(state, k=3),
                np.fliplr(state),
                np.flipud(state)
            ])
        else:
            choices = [
                np.rot90(state, k=1),
                np.rot90(state, k=2),
                np.rot90(state, k=3),
                np.fliplr(state),
                np.flipud(state)
            ]
            states.extend(random.sample(choices, 2))
        return states

    def store_experience(self, state, action, reward, next_state, done):
        """Lưu kinh nghiệm, ưu tiên chất lượng cao và giảm data augmentation."""
        if not done:
            game_temp = TicTacToe(self.size)
            game_temp.board = next_state
            x, y = divmod(action, self.size)
            if game_temp.can_extend_chain(x, y, -1):
                reward += 0.1
            if game_temp.can_block(x, y, 1):
                reward += 0.2

        is_high_quality = done or abs(reward) >= 0.5
        full_augmentation = is_high_quality
        state_symmetries = self.augment_state(state, full_augmentation)
        next_state_symmetries = self.augment_state(next_state, full_augmentation)
        
        for sym_state, sym_next_state in zip(state_symmetries, next_state_symmetries):
            experience = (sym_state, action, reward, sym_next_state, done)
            self.memory.append(experience)
            self.priorities.append(1.0)
            if is_high_quality:
                self.elite_memory.append(experience)
                self.elite_priorities.append(1.0)
        
            if len(self.memory) > self.memory_capacity:
                min_priority_idx = np.argmin(self.priorities)
                self.memory.pop(min_priority_idx)
                self.priorities.pop(min_priority_idx)

    def dqn_move(self, episode=0, total_episodes=100000, use_alpha_beta=False):
        """Chọn nước đi với epsilon-greedy cải tiến."""
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (1 - episode / (0.8 * total_episodes))

        if use_alpha_beta and episode < 5000 and random.random() < 0.5:
            move = self.alpha_beta_move(depth=2, is_maximizing_player=True)[1]
            if move is not None:
                return move

        state = self.board.astype(np.float32)
        state = torch.tensor(state, device=self.device).unsqueeze(0)

        if random.random() < epsilon:
            return random.choice(self.get_available_moves())

        with torch.no_grad():
            q_values = self.policy_net(state)
        available = self.get_available_moves()
        flat_available = [i * self.size + j for i, j in available]
        q_values_np = q_values.cpu().numpy().flatten()
        best_flat = max(flat_available, key=lambda idx: q_values_np[idx])
        return divmod(best_flat, self.size)

    def replay(self):
        """Huấn luyện mô hình, lấy mẫu từ cả memory và elite_memory."""
        total_experiences = len(self.memory) + len(self.elite_memory)
        if total_experiences < 64:
            return

        current_batch_size = min(self.batch_size, total_experiences)
        memory_samples = int(current_batch_size * 0.8) if len(self.memory) > 0 else 0
        elite_samples = current_batch_size - memory_samples if len(self.elite_memory) >= int(current_batch_size * 0.2) else 0
        if elite_samples > 0:
            memory_samples = current_batch_size - elite_samples
        else:
            memory_samples = current_batch_size

        batch = []
        indices = []

        if memory_samples > 0 and len(self.memory) > 0:
            priorities = np.array(self.priorities) + self.per_epsilon
            probabilities = priorities ** self.per_alpha
            probabilities /= probabilities.sum()
            mem_indices = np.random.choice(len(self.memory), min(memory_samples, len(self.memory)), p=probabilities)
            batch.extend([self.memory[idx] for idx in mem_indices])
            indices.extend([(0, idx) for idx in mem_indices])

        if elite_samples > 0 and len(self.elite_memory) > 0:
            priorities = np.array(self.elite_priorities) + self.per_epsilon
            probabilities = priorities ** self.per_alpha
            probabilities /= probabilities.sum()
            elite_indices = np.random.choice(len(self.elite_memory), min(elite_samples, len(self.elite_memory)), p=probabilities)
            batch.extend([self.elite_memory[idx] for idx in elite_indices])
            indices.extend([(1, idx) for idx in elite_indices])

        if len(batch) == 0:
            return

        current_batch_size = len(batch)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(state)
        with torch.no_grad():
            next_q_values = self.policy_net(next_state)
            next_q_target_values = self.target_net(next_state)
        target = q_values.clone()

        for i in range(current_batch_size):
            if done[i]:
                target[i, action[i]] = reward[i]
            else:
                best_action = torch.argmax(next_q_values[i]).item()
                target[i, action[i]] = reward[i] + self.gamma * next_q_target_values[i, best_action]

        loss = nn.MSELoss()(q_values, target)
        td_errors = torch.abs(q_values - target).detach().max(dim=1)[0].cpu().numpy()

        for (mem_type, idx), error in zip(indices, td_errors):
            if mem_type == 0:
                self.priorities[idx] = error
            else:
                self.elite_priorities[idx] = error

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

    def save_model(self, board_size):
        """Lưu mô hình vào file."""
        torch.save(self.policy_net.state_dict(), f"dqn_{board_size}x{board_size}.pth")

    def save_memory(self, filename):
        """Lưu bộ nhớ kinh nghiệm vào file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def save_elite_memory(self, filename):
        """Lưu bộ nhớ chất lượng cao vào file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.elite_memory, f)

    def load_memory(self, filename):
        """Tải bộ nhớ kinh nghiệm từ file."""
        try:
            with open(filename, 'rb') as f:
                self.memory = pickle.load(f)
                self.priorities = [1.0] * len(self.memory)
        except FileNotFoundError:
            print(f"Memory file {filename} not found. Starting with empty memory.")

    def load_elite_memory(self, filename):
        """Tải bộ nhớ chất lượng cao từ file."""
        try:
            with open(filename, 'rb') as f:
                self.elite_memory = pickle.load(f)
                self.elite_priorities = deque([1.0] * len(self.elite_memory), maxlen=10000)
        except FileNotFoundError:
            print(f"Elite memory file {filename} not found. Starting with empty elite memory.")

    def update_target_network(self):
        """Cập nhật mô hình mục tiêu từ mô hình chính."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

class Algorithm:
    """Lớp xử lý logic trận đấu và huấn luyện AI."""
    
    @staticmethod
    def evaluate_model(ai, size, num_games=100):
        """Đánh giá mô hình bằng cách đấu với đối thủ ngẫu nhiên."""
        wins = 0
        for _ in range(num_games):
            game = TicTacToe(size)
            current_player = 1
            while True:
                if current_player == 1:
                    move = game.easy_move()
                    game.make_move(*move, 1)
                else:
                    move = ai.dqn_move(episode=100000)
                    game.make_move(*move, -1)
                
                if game.is_winner(-1):
                    wins += 1
                    break
                if game.is_winner(1) or game.is_draw():
                    break
                current_player *= -1
        return wins / num_games

    @staticmethod
    def play_game(size, mode):
        """Chơi một trận với người chơi, hỗ trợ chế độ dễ/trung bình/khó."""
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
                    move = game.hard_move(-1)
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

    @staticmethod
    def train_game(size, episodes=100000, target_update=200, load_model_path=None, load_memory_path=None, load_elite_memory_path=None):
        """Huấn luyện hai AI đấu với nhau, lưu mô hình tốt nhất, hỗ trợ tải dữ liệu cũ."""
        print(f"Training Double DQN on {size}x{size} board for {episodes} episodes.\n")
        ai1 = TicTacToeAI(size=size)
        ai2 = TicTacToeAI(size=size)

        if load_model_path:
            try:
                ai1.policy_net.load_state_dict(torch.load(load_model_path))
                ai1.target_net.load_state_dict(ai1.policy_net.state_dict())
                ai2.policy_net.load_state_dict(torch.load(load_model_path))
                ai2.target_net.load_state_dict(ai2.policy_net.state_dict())
                print(f"Loaded pre-trained model from {load_model_path}")
            except FileNotFoundError:
                print(f"Model file {load_model_path} not found. Starting with new model.")

        if load_memory_path:
            ai1.load_memory(load_memory_path)
            ai2.load_memory(load_memory_path)
            print(f"Loaded memory from {load_memory_path}")

        if load_elite_memory_path:
            ai1.load_elite_memory(load_elite_memory_path)
            ai2.load_elite_memory(load_elite_memory_path)
            print(f"Loaded elite memory from {load_elite_memory_path}")

        win_ai1, win_ai2, draws = 0, 0, 0
        best_win_rate = 0

        try:
            for episode in range(episodes):
                game = TicTacToe(size)
                current_player = 1
                state = game.board.astype(np.float32)

                while True:
                    flat_state = state.copy()
                    move = ai1.dqn_move(episode, episodes, use_alpha_beta=True) if current_player == 1 else ai2.dqn_move(episode, episodes, use_alpha_beta=True)
                    x, y = move
                    game.make_move(x, y, current_player)
                    reward = 0

                    if game.is_winner(current_player):
                        reward = 1
                        done = True
                    elif game.is_draw():
                        reward = 0.5
                        done = True
                    else:
                        done = False

                    next_state = game.board.astype(np.float32)
                    action_index = x * size + y

                    if current_player == 1:
                        ai1.store_experience(flat_state, action_index, reward if done else 0, next_state, done)
                    else:
                        ai2.store_experience(flat_state, action_index, -reward if done else 0, next_state, done)

                    if done:
                        if episode % 100 == 0:
                            print(f"\nFinal board at Episode {episode}:")
                            game.print_board()
                        if reward == 1:
                            if current_player == 1:
                                print("AI 1 (X) wins!")
                                win_ai1 += 1
                            else:
                                print("AI 2 (O) wins!")
                                win_ai2 += 1
                        else:
                            print("It's a draw!")
                            draws += 1
                        break

                    current_player *= -1
                    state = next_state

                if len(ai1.memory) >= 64:
                    ai1.replay()
                if len(ai2.memory) >= 64:
                    ai2.replay()

                if episode % target_update == 0:
                    ai1.update_target_network()
                    ai2.update_target_network()

                if episode % 1000 == 0:
                    print(f"\n--- Episode {episode} Summary ---")
                    print(f"AI 1 (X) wins: {win_ai1}")
                    print(f"AI 2 (O) wins: {win_ai2}")
                    print(f"Draws        : {draws}")
                    win_rate = Algorithm.evaluate_model(ai1, size)
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        ai1.save_model(size)
                        ai1.save_memory(f"memory_{size}x{size}.pkl")
                        ai1.save_elite_memory(f"elite_memory_{size}x{size}.pkl")
                        print(f"Saved model to dqn_{size}x{size}.pth, memory to memory_{size}x{size}.pkl, elite memory to elite_memory_{size}x{size}.pkl")
                    print(f"Win rate: {win_rate:.2f}")
                    print("-----------------------------------\n")

        except KeyboardInterrupt:
            if len(ai1.memory) >= 10000:
                win_rate = Algorithm.evaluate_model(ai1, size)
                if win_rate >= 0.5:
                    ai1.save_model(size)
                    ai1.save_memory(f"memory_{size}x{size}_interrupt_{episode}.pkl")
                    ai1.save_elite_memory(f"elite_memory_{size}x{size}_interrupt_{episode}.pkl")
                    ai2.save_model(size)
                    ai2.save_memory(f"memory_{size}x{size}_interrupt_{episode}.pkl")
                    ai2.save_elite_memory(f"elite_memory_{size}x{size}_interrupt_{episode}.pkl")
                    print(f"\nTraining stopped. Saved models to dqn_{size}x{size}.pth, memory to memory_{size}x{size}_interrupt_{episode}.pkl, elite memory to elite_memory_{size}x{size}_interrupt_{episode}.pkl (win_rate: {win_rate:.2f})")
                else:
                    print(f"\nTraining stopped. Model not saved due to low win_rate ({win_rate:.2f} < 0.5)")
            else:
                print(f"\nTraining stopped. Model and memory not saved due to insufficient experiences ({len(ai1.memory)} < 10000)")

class Main:
    """Lớp chính để chạy chương trình, cho phép chọn chế độ chơi hoặc huấn luyện."""
    
    @staticmethod
    def main():
        """Hàm chính: hiển thị menu và xử lý lựa chọn người dùng."""
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
                Algorithm.train_game(size)
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    Main.main()