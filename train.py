import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class TicTacToe:
    '''Class TicTacToe: Lớp quản lý trò chơi Tic-Tac-Toe'''

    def __init__(self, size=3):
        '''Khởi tạo bảng trò chơi với kích thước cho trước'''
        self.size = size
        self.board = np.zeros((size, size), dtype=int)

    def is_winner(self, player):
        '''Kiểm tra xem người chơi có chiến thắng không'''
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
        '''Kiểm tra nếu trận đấu là hòa (tất cả ô đã đi và không có ai thắng)'''
        return not np.any(self.board == 0) and not self.is_winner(1) and not self.is_winner(-1)

    def get_available_moves(self):
        '''Trả về danh sách các nước đi hợp lệ (các ô trống)'''
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]

    def make_move(self, x, y, player):
        '''Thực hiện nước đi của người chơi tại vị trí (x, y)'''
        if self.board[x, y] == 0:
            self.board[x, y] = player
            
            return True
        
        return False

    def print_board(self):
        '''In ra bảng trò chơi hiện tại dưới dạng bảng ký tự (X, O, .)'''
        symbols = {0: '.', 1: 'X', -1: 'O'}

        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))

        print()

    def easy_move(self):
        '''Chế độ dễ: Chọn ngẫu nhiên một nước đi từ các ô trống'''
        return random.choice(self.get_available_moves())

    def medium_move(self, player):
        '''Chế độ trung bình: Chặn hoặc thắng thông minh hơn'''
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

        # Ưu tiên tạo chuỗi tấn công
        for i, j in self.get_available_moves():
            if self.can_extend_chain(i, j, player):
                return (i, j)

        # Kiểm tra những nước đi có thể chặn chuỗi dài của đối thủ (2 hoặc 3 quân liên tiếp)
        for i, j in self.get_available_moves():
            block_move = self.can_block(i, j, opponent)

            if block_move:
                print(f"Blocking move at {block_move} to stop opponent from winning!")

                return block_move

        # Dùng alpha-beta depth = 2
        _, move = self.alpha_beta_move(depth=2, is_maximizing_player=(player == -1))
        
        return move

    def can_extend_chain(self, x, y, player):
        """Kiểm tra nếu đặt tại (x,y) có giúp nối thành chuỗi liên tiếp (kể cả nối vào giữa)"""

        if self.board[x, y] != 0:
            return False  # Ô đã bị chiếm

        # Mục tiêu chuỗi theo kích thước bàn cờ
        if self.size == 5:
            target_length = 3
        elif self.size == 7:
            target_length = 4
        else:
            return False

        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]

        for dx, dy in directions:
            # 1. Kiểm tra mở rộng 2 đầu (bình thường)
            count = 1  # tính luôn (x, y)
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

            # 2. Kiểm tra nối vào giữa: [player - empty(x,y) - player]
            nx1, ny1 = x - dx, y - dy
            nx2, ny2 = x + dx, y + dy
            if all(0 <= nx < self.size and 0 <= ny < self.size for nx, ny in [(nx1, ny1), (nx2, ny2)]):
                if self.board[nx1, ny1] == player and self.board[nx2, ny2] == player:
                    # Đã có 2 đầu, thêm ô giữa là nối chuỗi
                    if target_length <= 3:
                        return True  # đủ chuỗi
                    else:
                        # Cần kiểm tra thêm: có ít nhất một quân nữa cùng hướng?
                        nx3 = x + 2 * dx
                        ny3 = y + 2 * dy
                        if 0 <= nx3 < self.size and 0 <= ny3 < self.size and self.board[nx3, ny3] == player:
                            return True  # ví dụ chuỗi X . X X
                        nx3 = x - 2 * dx
                        ny3 = y - 2 * dy
                        if 0 <= nx3 < self.size and 0 <= ny3 < self.size and self.board[nx3, ny3] == player:
                            return True  # ví dụ chuỗi X X . X

        return False

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
            if count == win_length - 1:  # Trường hợp có 3 quân đối thủ và 1 ô trống giữa
                if potential_block:
                    return potential_block

        return None  # Không tìm thấy vị trí cần chặn

    def evaluate_board(self):
        '''Đánh giá bảng cờ, trả về điểm cho AI hoặc người chơi'''
        if self.is_winner(-1):
            return 10 # AI thắng
        elif self.is_winner(1):
            return -10 # Người chơi thắng
        
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
        '''Thuật toán alpha-beta tìm nước đi tối ưu với độ sâu cho trước'''
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
        '''Chế độ khó: Sử dụng alpha-beta hoặc DQN tùy vào kích thước bàn cờ'''
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

        # Ưu tiên tạo chuỗi tấn công
        for i, j in self.get_available_moves():
            if self.can_extend_chain(i, j, player):
                return (i, j)

        # Kiểm tra những nước đi có thể chặn chuỗi dài của đối thủ (2 hoặc 3 quân liên tiếp)
        for i, j in self.get_available_moves():
            block_move = self.can_block(i, j, opponent)

            if block_move:
                print(f"Blocking move at {block_move} to stop opponent from winning!")

                return block_move

        if self.size == 3:
            return self.alpha_beta_move(depth=9, is_maximizing_player=True)[1]
        else:
            return self.dqn_move()

class TicTacToeAI(TicTacToe):
    '''Class TicTacToeAI: Mở rộng thêm DQN'''

    def __init__(self, size=3):
        super().__init__(size) # Khởi tạo lớp cha TicTacToe
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Chọn thiết bị (GPU nếu có, CPU nếu không)
        self.policy_net = self.create_model() # Mô hình chính để lựa chọn hành động
        self.target_net = self.create_model() # Mô hình mục tiêu để cập nhật
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001) # Bộ tối ưu Adam
        self.memory = deque(maxlen=10000) # Bộ nhớ để lưu trữ kinh nghiệm (experience replay)
        self.batch_size = 64 # Kích thước batch cho quá trình huấn luyện
        self.gamma = 0.99 # Hệ số giảm giá (discount factor) cho Q-learning

    def create_model(self):
        '''Hàm tạo mô hình mạng neural (neural network) với các lớp Fully Connected (FC)'''
        model = nn.Sequential(
            nn.Linear(self.size * self.size, 128), # Lớp đầu tiên chuyển từ kích thước bàn cờ sang 128 nút
            nn.ReLU(), # Hàm kích hoạt ReLU
            nn.Linear(128, 128), # Lớp thứ hai với 128 nút
            nn.ReLU(), # Hàm kích hoạt ReLU
            nn.Linear(128, self.size * self.size) # Lớp cuối cùng trả về giá trị Q cho mỗi ô trên bàn cờ
        )

        return model.to(self.device) # Đưa mô hình vào device (CPU hoặc GPU)

    def dqn_move(self):
        '''Hàm quyết định nước đi (move) của AI sử dụng DQN'''
        state = self.board.flatten().astype(np.float32) # Chuyển đổi trạng thái bàn cờ thành mảng 1 chiều
        state = torch.tensor(state).unsqueeze(0).to(self.device) # Chuyển thành tensor và đưa lên device

        with torch.no_grad():
            q_values = self.policy_net(state) # Lấy giá trị Q từ mô hình chính

        available = self.get_available_moves() # Lấy các ô có thể di chuyển

        if random.random() < 0.1:
            return random.choice(available) # Chọn nước đi ngẫu nhiên (exploration)

        # Chuyển các ô có thể di chuyển thành các chỉ số phẳng
        flat_available = [i * self.size + j for i, j in available]
        q_values_np = q_values.cpu().numpy().flatten() # Chuyển giá trị Q về mảng numpy
        best_flat = max(flat_available, key=lambda idx: q_values_np[idx]) # Chọn ô có giá trị Q cao nhất
        
        return divmod(best_flat, self.size) # Chuyển lại chỉ số phẳng thành tọa độ 2D

    def replay(self):
        '''Hàm huấn luyện (replay) mô hình từ bộ nhớ'''
        if len(self.memory) < self.batch_size: # Nếu bộ nhớ không đủ thì không huấn luyện
            return
        
        batch = random.sample(self.memory, self.batch_size) # Lấy ngẫu nhiên batch dữ liệu
        state, action, reward, next_state, done = zip(*batch)

        # Chuyển batch thành tensor
        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(state) # Lấy giá trị Q cho batch từ mô hình chính
        next_q_values = self.target_net(next_state) # Lấy giá trị Q cho next_state từ mô hình mục tiêu
        target = q_values.clone() # Sao chép giá trị Q hiện tại

        for i in range(self.batch_size):
            if done[i]: # Nếu kết thúc trò chơi, chỉ cần phần thưởng
                target[i, action[i]] = reward[i]
            else: # Nếu chưa kết thúc, tính giá trị Q mục tiêu từ mô hình tiếp theo
                best_action = torch.argmax(self.policy_net(next_state[i:i+1])).item() 
                target[i, action[i]] = reward[i] + self.gamma * next_q_values[i, best_action]

        # Tính và tối ưu hóa loss
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_experience(self, state, action, reward, next_state, done):
        '''Hàm lưu trữ kinh nghiệm vào bộ nhớ'''
        self.memory.append((state, action, reward, next_state, done))

    def update_target_network(self):
        '''Cập nhật mô hình mục tiêu (target network)'''
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Cập nhật mô hình mục tiêu từ mô hình chính

    def save_model(self, board_size):
        '''Hàm lưu mô hình đã huấn luyện'''
        torch.save(self.policy_net.state_dict(), f"dqn_{board_size}x{board_size}.pth") # Lưu mô hình vào file

class Algorithm:
    '''Class Algorithm: Xử lý trận đấu'''

    @staticmethod
    def play_game(size, mode):
        game = TicTacToeAI(size=size) # Khởi tạo AI chơi trò chơi
        current_player = 1 # Người chơi bắt đầu với số 1 (X)

        print(f"\nStarting game on {size}x{size} board in {mode} mode...\n")
        game.print_board() # In bàn cờ ban đầu

        while True:
            # Người chơi thực hiện nước đi
            if current_player == 1:
                try:
                    move = input("Your move (row col): ") # Nhập nước đi
                    x, y = map(int, move.strip().split()) # Chuyển đổi thành tọa độ
                    if not game.make_move(x, y, 1): # Kiểm tra nước đi hợp lệ
                        print("Invalid move. Try again.")
                        continue
                except:
                    print("Invalid input format. Use: row col")
                    continue
            else:
                # AI thực hiện nước đi theo chế độ
                if mode == "easy":
                    move = game.easy_move()
                elif mode == "medium":
                    move = game.medium_move(-1)
                elif mode == "hard":
                    move = game.hard_move(-1) # Chế độ khó (AI sử dụng DQN)
                game.make_move(move[0], move[1], -1)  # AI đi nước
                print(f"AI moves at: {move}")

            game.print_board() # In lại bàn cờ sau khi mỗi nước đi

            # Kiểm tra nếu ai thắng hoặc hòa
            if game.is_winner(current_player):
                print("You win!" if current_player == 1 else "AI wins!")
                break
            elif game.is_draw():
                print("It's a draw!")
                break

            current_player *= -1 # Chuyển lượt cho người chơi khác

    @staticmethod
    def train_game(size, target_update=10):
        print(f"Training Double DQN on {size}x{size} board continuously. Press Ctrl+C to stop.\n")

        ai1 = TicTacToeAI(size=size) # Khởi tạo AI 1
        ai2 = TicTacToeAI(size=size) # Khởi tạo AI 2
        episode = 0 # Số lượng episode

        win_ai1 = 0 # Số chiến thắng của AI 1
        win_ai2 = 0 # Số chiến thắng của AI 2
        draws = 0 # Số trận hòa

        try:
            while True:
                game = TicTacToe(size) # Khởi tạo trò chơi mới
                current_player = 1 # Bắt đầu với người chơi 1
                state = game.board.flatten().astype(np.float32) # Chuyển trạng thái bàn cờ thành mảng 1 chiều

                while True:
                    flat_state = state.copy() # Sao chép trạng thái hiện tại

                    # AI 1 hoặc AI 2 thực hiện nước đi
                    if current_player == 1:
                        move = ai1.dqn_move() # AI 1 thực hiện nước đi theo DQN
                    else:
                        move = ai2.dqn_move() # AI 2 thực hiện nước đi theo DQN

                    x, y = move
                    game.make_move(x, y, current_player) # Thực hiện nước đi
                    reward = 0 # Khởi tạo phần thưởng

                    # Kiểm tra nếu có người thắng hoặc hòa
                    if game.is_winner(current_player):
                        reward = 1 # Người chơi thắng
                        done = True # Trò chơi kết thúc
                    elif game.is_draw():
                        reward = 0.5 # Trận đấu hòa
                        done = True # Trò chơi vẫn tiếp tục
                    else:
                        done = False

                    next_state = game.board.flatten().astype(np.float32) # Trạng thái sau khi di chuyển
                    action_index = x * size + y # Chỉ số hành động tương ứng
 
                    # Lưu trữ kinh nghiệm cho cả hai AI
                    if current_player == 1:
                        ai1.store_experience(flat_state, action_index, reward if done else 0, next_state, done)
                    else:
                        ai2.store_experience(flat_state, action_index, -reward if done else 0, next_state, done)

                    # Nếu trận đấu kết thúc, cập nhật kết quả
                    if done:
                        print(f"\nFinal board at Episode {episode}:")
                        game.print_board() # In bàn cờ cuối cùng
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

                    current_player *= -1 # Chuyển lượt
                    state = next_state # Cập nhật trạng thái

                # Huấn luyện DQN cho cả hai AI
                ai1.replay()
                ai2.replay()

                # Cập nhật mạng mục tiêu (target network) sau mỗi vài episode
                if episode % target_update == 0:
                    ai1.update_target_network()
                    ai2.update_target_network()

                # In thông tin sau mỗi 100 episode
                if episode % 100 == 0:
                    print(f"\n--- Episode {episode} Summary ---")
                    print(f"AI 1 (X) wins: {win_ai1}")
                    print(f"AI 2 (O) wins: {win_ai2}")
                    print(f"Draws        : {draws}")
                    print("-----------------------------------\n")
                    
                    # Lưu mô hình sau mỗi 100 episode
                    ai1.save_model(size)
                    ai2.save_model(size)

                episode += 1 # Tăng số episode

        except KeyboardInterrupt:
            ai1.save_model(size) # Lưu mô hình khi dừng huấn luyện
            ai2.save_model(size)
            print(f"\nTraining stopped. Models saved as dqn_{size}x{size}.pth")

class Main:
    '''Class Main: Lớp chính'''

    @staticmethod
    def main():
        # Chọn chế độ chơi hoặc huấn luyện
        mode_choice = input("Choose mode (1: Play, 2: Training): ")

        if mode_choice == "1": # Chế độ chơi
            # Chọn kích thước bàn cờ
            size = int(input("Choose board size (3, 5, 7): "))
            # Chọn mức độ khó
            difficulty = input("Choose difficulty (1: Easy, 2: Medium, 3: Hard): ")
            mode = {"1": "easy", "2": "medium", "3": "hard"}.get(difficulty) # Ánh xạ chọn lựa khó

            if mode: # Nếu lựa chọn hợp lệ
                Algorithm.play_game(size, mode) # Gọi hàm play_game từ class Algorithm
            else:
                print("Invalid choice!") # In thông báo lỗi nếu mức độ khó không hợp lệ
        elif mode_choice == "2":# Chế độ huấn luyện
            # Chọn kích thước bàn cờ (chỉ hỗ trợ 5x5 và 7x7)
            size = int(input("Choose board size (5, 7): "))
            if size not in [5, 7]: # Kiểm tra xem kích thước có hợp lệ không
                print("Training mode is only available for 5x5 and 7x7 boards.") # Thông báo lỗi nếu chọn kích thước không hợp lệ
            else:
                Algorithm.train_game(size) # Gọi hàm train_game từ class Algorithm để bắt đầu huấn luyện

        else:
            print("Invalid choice!") # Thông báo lỗi nếu lựa chọn chế độ không hợp lệ

if __name__ == "__main__":
    Main.main() # Chạy hàm main khi chương trình bắt đầu
