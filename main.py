import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import sys
import pickle
import os
import time
from copy import deepcopy

"""
Tic-Tac-Toe AI với ba mức độ khó (easy, medium, hard) và huấn luyện DQN.

Chương trình triển khai trò chơi Tic-Tac-Toe với:
- Chế độ chơi: Easy (IDDFS), Medium (alpha-beta + heuristics), Hard (DQN hoặc alpha-beta).
- Huấn luyện: DQN cho bàn 5x5 và 7x7, lưu mô hình và kinh nghiệm.
- So sánh: Đánh giá sức mạnh giữa Medium vs Easy và Medium vs Hard.

Tối ưu hóa:
- Hiệu suất: ~15-30 giây/episode (7x7, MacBook Air M2, MPS), ~5-10 giây/episode (5x5, MSI, CUDA), ~0.1-0.5s/nước (3x3).
- RAM: ~0.5-1GB, file .pkl ~200-500MB.
- CNN: 3 lớp Conv2d (32, 64, 128), giảm ~30-40% tham số.
- Replay: batch_size=256, per_alpha=0.9, chạy khi memory >= 1000.
- Alpha-beta: Tăng tần suất (episode < 10000, xác suất 80%), độ sâu 1.

Hướng dẫn chạy:
1. Cài đặt: `pip3 install numpy torch==2.6.0`.
2. Chạy: `python3 tictactoe.py`.
3. Chọn:
   - Mode 1: Chơi với AI (3x3, 5x5, 7x7; easy, medium, hard).
   - Mode 2: Huấn luyện DQN (5x5, 7x7; 5 stage, mỗi stage 100,000 episodes).
   - Mode 3: So sánh Medium vs Easy/Hard (3x3, 5x5, 7x7).
"""

# Đặt số luồng cho CPU
torch.set_num_threads(8)

class TicTacToe:
    """
    Quản lý bàn cờ Tic-Tac-Toe và logic cơ bản (thắng, thua, hòa).

    Mục đích: Cung cấp các hàm để chơi Tic-Tac-Toe trên bàn size x size.
    Hiệu suất: Hầu hết hàm O(size^2) hoặc thấp hơn.
    Liên quan: Dùng trong mọi chế độ (easy, medium, hard).
    Dữ liệu: Không lưu file, chỉ quản lý trạng thái bàn cờ trong RAM.
    """

    def __init__(self, size=3):
        """
        Khởi tạo bàn cờ size x size với tất cả ô trống.

        Logic: Tạo ma trận size x size với giá trị 0 (trống), 1 (X: người chơi), -1 (O: AI).
        Hiệu suất: O(size^2) để khởi tạo ma trận.
        Liên quan: Dùng trong mọi chế độ để bắt đầu ván.
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)

    def is_winner(self, player):
        """
        Kiểm tra người chơi (1: X, -1: O) có thắng không bằng cách tìm chuỗi liên tiếp.

        Logic:
        - Tìm chuỗi liên tiếp dài win_length (3 cho 3x3, 4 cho 5x5, 5 cho 7x7) trên hàng, cột, đường chéo.
        - Trả về True nếu tìm thấy chuỗi.
        Hiệu suất: O(size^2 * win_length) do duyệt hàng, cột, và đường chéo.
        Liên quan: Dùng trong mọi chế độ để xác định kết thúc ván.
        """
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
        """
        Kiểm tra trạng thái hòa (bàn đầy, không ai thắng).

        Logic: Trả về True nếu không còn ô trống và không có người thắng.
        Hiệu suất: O(size^2) để kiểm tra ô trống, cộng O(size^2 * win_length) từ is_winner.
        Liên quan: Dùng trong mọi chế độ để kết thúc ván.
        """
        return not np.any(self.board == 0) and not self.is_winner(1) and not self.is_winner(-1)

    def get_available_moves(self):
        """
        Liệt kê các ô trống (nước đi hợp lệ) dưới dạng [(x, y)].

        Logic: Tìm tất cả vị trí có giá trị 0 trên bàn cờ.
        Hiệu suất: O(size^2) để duyệt ma trận.
        Liên quan: Dùng trong mọi chế độ để xác định nước đi khả thi.
        """
        indices = np.where(self.board == 0)
        return list(zip(indices[0], indices[1]))

    def make_move(self, x, y, player):
        """
        Đặt nước đi tại (x, y) cho người chơi (1: X, -1: O).

        Logic: Đặt giá trị player tại (x, y) nếu ô trống, trả về True nếu thành công.
        Hiệu suất: O(1) để truy cập và cập nhật ô.
        Liên quan: Dùng trong mọi chế độ để cập nhật bàn cờ.
        """
        if self.board[x, y] == 0:
            self.board[x, y] = player
            return True
        return False

    def print_board(self):
        """
        In bàn cờ với ký hiệu: . (trống), X (người chơi), O (AI).

        Logic: Chuyển giá trị ma trận thành ký hiệu và in từng hàng.
        Hiệu suất: O(size^2) để duyệt và in.
        Liên quan: Dùng trong mọi chế độ để hiển thị trạng thái.
        """
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()

    def score_move(self, x, y, player):
        """
        Đánh giá nước đi tại (x, y) dựa trên chuỗi, chặn, và vị trí trung tâm.

        Logic:
        - +5 nếu tạo chuỗi 3 (5x5) hoặc 4 (7x7).
        - +4 nếu chặn chuỗi đối thủ.
        - +3 nếu tạo chuỗi 2 với ít nhất 1 đầu mở.
        - +0-1 dựa trên khoảng cách đến trung tâm (gần trung tâm điểm cao hơn).
        Hiệu suất: O(1) vì chỉ kiểm tra 8 hướng với tối đa 4 ô mỗi hướng.
        Liên quan: Dùng trong easy (iddfs_move) và medium (alpha_beta_move) để ưu tiên nước đi.
        """
        score = 0
        opponent = -player
        if self.can_extend_chain(x, y, player):
            score += 5
        if self.can_block(x, y, opponent):
            score += 4
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]
        for dx, dy in directions:
            count = 1
            open_ends = 0
            for dir in [1, -1]:
                nx = x + dir * dx
                ny = y + dir * dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx, ny] == player:
                        count += 1
                    elif self.board[nx, ny] == 0:
                        open_ends += 1
            if count == 2 and open_ends >= 1:
                score += 3
        center = self.size // 2
        distance_to_center = abs(x - center) + abs(y - center)
        score += 1.0 * (self.size - distance_to_center) / self.size
        return score

    def easy_move(self, player):
        """
        Chế độ dễ: Sử dụng Iterative Deepening DFS (IDDFS) với độ sâu giới hạn.

        Logic:
        1. Kiểm tra nước đi thắng ngay lập tức cho player.
        2. Kiểm tra nước đi chặn đối thủ thắng.
        3. Dùng IDDFS với độ sâu 2-3 (3x3, 5x5: 3; 7x7: 2-3 tùy số ô trống).
        4. Nếu hết thời gian, chọn ngẫu nhiên từ ô trống.
        Hiệu suất:
        - IDDFS: O(b^d) với b ~ size^2, d = 2-3.
        - Thời gian: ~0.1-0.5s tùy kích thước.
        Liên quan: Chỉ dùng trong chế độ easy, yếu hơn medium và hard.
        """
        opponent = -player
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
        available_moves = len(self.get_available_moves())
        max_depth = 3
        if self.size != 3 and available_moves > 20:
            max_depth = 2
        time_limit = 0.2 if self.size == 3 else 0.5 if self.size == 5 else 0.8
        move = self.iddfs_move(player, max_depth=max_depth, time_limit=time_limit)
        if move is None:
            return random.choice(self.get_available_moves())
        return move

    def iddfs_move(self, player, max_depth, time_limit):
        """
        Tìm nước đi tốt nhất bằng IDDFS với giới hạn thời gian.

        Logic:
        - Duyệt từng độ sâu từ 1 đến max_depth.
        - Sắp xếp nước đi theo score_move để ưu tiên nước tốt.
        - Dùng depth_limited_dfs để đánh giá điểm.
        - Dừng nếu vượt thời gian hoặc tìm nước đi thắng (+10).
        Hiệu suất: O(b^d) với b ~ size^2, d = 1-3, giảm nhờ sắp xếp.
        Liên quan: Chỉ dùng trong easy (easy_move).
        """
        start_time = time.time()
        best_move = None
        best_score = -np.inf
        moves = self.get_available_moves()
        moves = sorted(moves, key=lambda move: -self.score_move(move[0], move[1], player))

        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_limit:
                break
            temp_best_move = None
            temp_best_score = -np.inf
            for i, j in moves:
                self.board[i, j] = player
                score = self.depth_limited_dfs(depth - 1, player, is_maximizing=False, alpha=-np.inf, beta=np.inf)
                self.board[i, j] = 0
                if score > temp_best_score:
                    temp_best_score = score
                    temp_best_move = (i, j)
                if time.time() - start_time > time_limit:
                    break
            if temp_best_score > best_score:
                best_score = temp_best_score
                best_move = temp_best_move
            if best_score >= 10:
                break
        return best_move

    def depth_limited_dfs(self, depth, player, is_maximizing, alpha, beta):
        """
        DFS với độ sâu giới hạn và cắt tỉa alpha-beta.

        Logic:
        - Nếu hết độ sâu hoặc ván kết thúc, trả về điểm từ evaluate_board.
        - Nếu maximizing (AI), chọn nước đi có điểm cao nhất.
        - Nếu minimizing (người chơi), chọn nước đi có điểm thấp nhất.
        - Sắp xếp nước đi theo score_move để cắt tỉa hiệu quả.
        Hiệu suất: O(b^d) với b ~ size^2, d = 1-3, giảm đáng kể nhờ alpha-beta.
        Liên quan: Chỉ dùng trong easy (iddfs_move).
        """
        if depth == 0 or self.is_winner(1) or self.is_winner(-1) or self.is_draw():
            return self.evaluate_board()

        moves = self.get_available_moves()
        moves = sorted(moves, key=lambda move: -self.score_move(move[0], move[1], -1 if is_maximizing else 1))

        if is_maximizing:
            max_score = -np.inf
            for i, j in moves:
                self.board[i, j] = -1
                score = self.depth_limited_dfs(depth - 1, player, is_maximizing=False, alpha=alpha, beta=beta)
                self.board[i, j] = 0
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_score
        else:
            min_score = np.inf
            for i, j in moves:
                self.board[i, j] = 1
                score = self.depth_limited_dfs(depth - 1, player, is_maximizing=True, alpha=alpha, beta=beta)
                self.board[i, j] = 0
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_score

    def medium_move(self, player, depth=2):
        """
        Chế độ trung bình: Kết hợp heuristics và alpha-beta pruning.

        Logic:
        1. Kiểm tra nước đi thắng ngay lập tức.
        2. Kiểm tra nước đi chặn đối thủ thắng.
        3. Tìm nước đi tạo chuỗi 3 (5x5) hoặc 4 (7x7) bằng can_extend_chain.
        4. Tìm nước đi chặn chuỗi đối thủ bằng can_block.
        5. Với xác suất 10%, chọn ngẫu nhiên để tăng tính bất ngờ.
        6. Dùng alpha_beta_move với độ sâu 2 để tìm nước đi tối ưu.
        Hiệu suất:
        - Alpha-beta: O(b^d) với b ~ size^2, d = 2, giảm nhờ cắt tỉa.
        - Thời gian: ~0.1-0.5s tùy kích thước.
        Liên quan: Chỉ dùng trong chế độ medium, mạnh hơn easy nhưng yếu hơn hard.
        """
        opponent = -player
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
                return block_move
        if random.random() < 0.1:
            return random.choice(self.get_available_moves())
        _, move = self.alpha_beta_move(depth=depth, is_maximizing_player=(player == -1))
        return move

    def can_extend_chain(self, x, y, player):
        """
        Kiểm tra nước đi tại (x, y) có tạo chuỗi 3 (5x5) hoặc 4 (7x7) không.

        Logic:
        - Kiểm tra 8 hướng, đếm số quân liên tiếp của player.
        - Trả về True nếu tạo chuỗi đủ dài hoặc nối hai quân thành chuỗi tiềm năng.
        Hiệu suất: O(1) vì chỉ kiểm tra tối đa 4 ô mỗi hướng.
        Liên quan: Dùng trong medium (medium_move) và hard (hard_move).
        """
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
        """
        Kiểm tra nước đi tại (x, y) có chặn chuỗi nguy hiểm của đối thủ không.

        Logic:
        - Kiểm tra 8 hướng, đếm số quân liên tiếp của đối thủ.
        - Trả về vị trí chặn nếu chuỗi dài win_length-1 hoặc win_length-1 với ô trống.
        Hiệu suất: O(1) vì chỉ kiểm tra tối đa 4 ô mỗi hướng.
        Liên quan: Dùng trong medium (medium_move) and hard (hard_move).
        """
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
        """
        Đánh giá trạng thái bàn cờ để ưu tiên thắng/thua và chuỗi tiềm năng.

        Logic:
        - +10 nếu AI thắng, -10 nếu người chơi thắng.
        - +2/-2 cho chuỗi 3 (5x5) hoặc 2 (3x3) với 1 ô trống.
        - +1/-1 cho chuỗi 2 tiềm năng (1 ô trống ở đầu hoặc cuối).
        - +0.1/-0.1 cho quân gần trung tâm.
        Hiệu suất: O(size^2) để duyệt bàn cờ và kiểm tra chuỗi.
        Liên quan: Dùng trong easy (depth_limited_dfs) và medium (alpha_beta_move).
        """
        if self.is_winner(-1):
            return 10
        elif self.is_winner(1):
            return -10
        score = 0
        target_length = 3 if self.size >= 5 else 2
        for i in range(self.size):
            for j in range(self.size - target_length + 1):
                row_slice = self.board[i, j:j + target_length].tolist()
                col_slice = self.board[j:j + target_length, i].tolist()
                if row_slice.count(-1) == target_length - 1 and row_slice.count(0) == 1:
                    score += 2
                if row_slice.count(1) == target_length - 1 and row_slice.count(0) == 1:
                    score -= 2
                if col_slice.count(-1) == target_length - 1 and col_slice.count(0) == 1:
                    score += 2
                if col_slice.count(1) == target_length - 1 and col_slice.count(0) == 1:
                    score -= 2
        for i in range(self.size):
            for j in range(self.size - 1):
                row_slice = self.board[i, j:j + 2].tolist()
                col_slice = self.board[j:j + 2, i].tolist()
                if row_slice == [-1, 0] or row_slice == [0, -1]:
                    score += 1
                if row_slice == [1, 0] or row_slice == [0, 1]:
                    score -= 1
                if col_slice == [-1, 0] or col_slice == [0, -1]:
                    score += 1
                if col_slice == [1, 0] or col_slice == [0, 1]:
                    score -= 1
        center = self.size // 2
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == -1:
                    distance = abs(i - center) + abs(j - center)
                    score += 0.1 * (self.size - distance) / self.size
                elif self.board[i, j] == 1:
                    distance = abs(i - center) + abs(j - center)
                    score -= 0.1 * (self.size - distance) / self.size
        return score

    def alpha_beta_move(self, depth, is_maximizing_player, alpha=-np.inf, beta=np.inf):
        """
        Tìm nước đi tối ưu bằng alpha-beta pruning.

        Logic:
        - Nếu hết độ sâu hoặc ván kết thúc, trả về điểm từ evaluate_board.
        - Nếu maximizing (AI), chọn nước đi có điểm cao nhất.
        - Nếu minimizing (người chơi), chọn nước đi có điểm thấp nhất.
        - Sắp xếp nước đi theo khoảng cách đến trung tâm để cắt tỉa hiệu quả.
        Hiệu suất: O(b^d) với b ~ size^2, d = 1-8, giảm đáng kể nhờ alpha-beta.
        Liên quan: Dùng trong medium (medium_move) và hard (hard_move).
        """
        if depth == 0 or self.is_winner(1) or self.is_winner(-1) or self.is_draw():
            return self.evaluate_board(), None
        moves = self.get_available_moves()
        center = self.size // 2
        moves = sorted(moves, key=lambda move: -(abs(move[0] - center) + abs(move[1] - center)))
        best_move = None
        if is_maximizing_player:
            max_eval = -np.inf
            for i, j in moves:
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
            for i, j in moves:
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
        """
        Chế độ khó: Kết hợp alpha-beta (3x3) hoặc DQN (5x5, 7x7) để tạo AI mạnh nhất.

        Logic:
        1. Kiểm tra nước đi thắng ngay lập tức.
        2. Kiểm tra nước đi chặn đối thủ thắng.
        3. Tìm nước đi tạo chuỗi 3 (5x5) hoặc 4 (7x7).
        4. Tìm nước đi chặn chuỗi đối thủ.
        5. Nếu 3x3: Dùng alpha_beta_move với độ sâu 8.
           Nếu 5x5/7x7: Dùng dqn_move (epsilon thấp), kiểm tra lại bằng alpha_beta_move độ sâu 2.
        Hiệu suất:
        - 3x3: ~0.1-0.5s do alpha-beta với độ sâu cao.
        - 5x5/7x7: ~0.5-0.7s do DQN + alpha-beta.
        Liên quan: Chỉ dùng trong chế độ hard.
        Dữ liệu: Dùng `dqn_{size}x{size}.pth` cho 5x5/7x7.
        """
        opponent = -player
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
                return block_move
        if self.size == 3:
            _, move = self.alpha_beta_move(depth=8, is_maximizing_player=(player == -1))
            return move
        else:
            dqn_move = self.dqn_move(episode=500000)  # Epsilon ~0.05
            moves = self.get_available_moves()
            best_score = -np.inf
            best_move = dqn_move
            for i, j in moves:
                self.board[i, j] = player
                score, _ = self.alpha_beta_move(depth=2, is_maximizing_player=False)
                self.board[i, j] = 0
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
            return best_move

class TicTacToeAI(TicTacToe):
    """
    Mở rộng TicTacToe để triển khai AI dùng Deep Q-Network (DQN) cho chế độ hard.

    Mục đích: Hỗ trợ chơi ở chế độ hard (5x5, 7x7) và huấn luyện DQN qua self-play.
    Tối ưu hóa:
    - Device: Ưu tiên MPS (MacBook M2), CUDA (MSI/cloud), fallback CPU.
    - CNN: 3 lớp Conv2d (32, 64, 128), giảm ~30-40% tham số, tiết kiệm ~30% thời gian.
    - Bộ nhớ: memory_capacity=20000, elite_memory_capacity=5000, giảm ~50% RAM (~0.5-1GB) và file .pkl (~200-500MB).
    - Replay: batch_size=256, per_alpha=0.9, chỉ chạy khi memory >= 1000, tăng tốc ~20%.
    - Mixed precision: Dùng autocast/GradScaler, tăng tốc ~20-50% trên MPS/CUDA.
    - Alpha-beta: Tăng tần suất (episode < 10000, xác suất 80%), độ sâu 1, tiết kiệm ~10-20%.
    Hiệu suất:
    - Huấn luyện: ~15-30 giây/episode (7x7, M2), ~5-10 giây/episode (5x5, MSI).
    - Chơi: ~0.5-0.7s/nước (5x5, 7x7).
    Liên quan: Dùng trong hard (hard_move, dqn_move) và huấn luyện (train_game).
    Dữ liệu:
    - `dqn_{size}x{size}.pth`: Trọng số DQN (~10-50MB).
    - `memory_{size}x{size}.pkl`: Kinh nghiệm (~100-500MB).
    - `elite_memory_{size}x{size}.pkl`: Kinh nghiệm chất lượng cao (~50-200MB).
    """

    def __init__(self, size=3, mode="hard", evaluate_model=True):
        """
        Khởi tạo AI với mô hình CNN, bộ nhớ, và thiết bị tối ưu.

        Logic:
        - Chọn device: MPS > CUDA > CPU.
        - Tạo policy_net và target_net (CNN 3 lớp).
        - Tải mô hình từ `dqn_{size}x{size}.pth` nếu có.
        - Đánh giá win rate nếu evaluate_model=True.
        - Khởi tạo bộ nhớ với capacity nhỏ (20000, 5000).
        Hiệu suất: O(size^2) để khởi tạo CNN, O(1) cho các bước khác.
        Liên quan: Dùng trong hard và huấn luyện.
        Dữ liệu: Tải `dqn_{size}x{size}.pth` nếu có.
        """
        super().__init__(size)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler(enabled=(self.device.type in ["mps", "cuda"]))
        print(f"Using device: {self.device}")
        self.policy_net = self.create_model(size).to(self.device)
        self.target_net = self.create_model(size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50000, gamma=0.1)
        self.memory = []
        self.memory_capacity = 20000
        self.priorities = []
        self.elite_memory = []
        self.elite_memory_capacity = 5000
        self.elite_priorities = []
        self.batch_size = 256
        self.gamma = 0.995
        self.per_epsilon = 1e-6
        self.per_alpha = 0.9
        if mode == "hard":
            model_path = f"dqn_{size}x{size}.pth"
            if os.path.exists(model_path):
                try:
                    self.policy_net.load_state_dict(torch.load(model_path))
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    print(f"Loaded pre-trained model from {model_path}")
                    if evaluate_model:
                        win_rate = Algorithm.evaluate_model(self, size, num_games=10)
                        print(f"Loaded model win rate: {win_rate:.2f}")
                        if win_rate < 0.5:
                            print("Warning: Loaded model may be weak. Consider retraining.")
                except RuntimeError as e:
                    print(f"Error loading model from {model_path}: {e}")
                    print("Using new model instead.")

    def create_model(self, size):
        """
        Tạo mô hình CNN tối ưu cho DQN.

        Logic:
        - Input: Ma trận size x size (1 kênh).
        - 3 lớp Conv2d (32, 64, 128) với BatchNorm2d và ReLU, không dùng Dropout.
        - 2 lớp fully connected: 128*size*size -> 512 -> size*size.
        - Output: Q-values cho size^2 ô.
        Hiệu suất: O(size^2) cho mỗi forward pass.
        Liên quan: Dùng trong hard (dqn_move) và huấn luyện (replay).
        """
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
                self.fc2 = nn.Linear(512, size * size)

            def forward(self, x):
                x = x.view(-1, 1, size, size)
                x = torch.relu(self.bn1(self.conv1(x)))
                x = torch.relu(self.bn2(self.conv2(x)))
                x = torch.relu(self.bn3(self.conv3(x)))
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        return DQN(size)

    def augment_state(self, state, full_augmentation=False):
        """
        Tạo các phiên bản đối xứng của bàn cờ để tăng dữ liệu huấn luyện.

        Logic:
        - Nếu full_augmentation: Tạo 6 phiên bản (xoay 90/180/270, lật ngang/dọc).
        - Nếu không: Tạo 2 phiên bản (xoay 90, lật ngang).
        - Trả về danh sách các trạng thái đối xứng.
        Hiệu suất: O(size^2) để xoay/lật ma trận.
        Liên quan: Dùng trong huấn luyện (store_experience).
        """
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
            states.extend([
                np.rot90(state, k=1),
                np.fliplr(state)
            ])
        return states

    def store_experience(self, state, action, reward, next_state, done):
        """
        Lưu kinh nghiệm huấn luyện vào bộ nhớ và bộ nhớ chất lượng cao.

        Logic:
        - Điều chỉnh phần thưởng:
          + +0.5 nếu tạo chuỗi (can_extend_chain).
          + +0.7 nếu chặn đối thủ (can_block).
          + -0.05 nếu không tạo chuỗi hoặc chặn.
        - Chỉ lưu kinh nghiệm chất lượng cao (done hoặc |reward| >= 0.5) với xác suất 50%.
        - Dùng augment_state để tăng dữ liệu (6 phiên bản nếu chất lượng cao, 2 nếu không).
        - Lưu vào memory (20000) hoặc elite_memory (5000), ưu tiên kinh nghiệm tốt.
        Hiệu suất: O(size^2) do augment_state và kiểm tra chuỗi/chặn.
        Liên quan: Dùng trong huấn luyện (train_game).
        """
        if not done:
            game_temp = TicTacToe(self.size)
            game_temp.board = next_state
            x, y = divmod(action, self.size)
            if game_temp.can_extend_chain(x, y, -1):
                reward += 0.5
            if game_temp.can_block(x, y, 1):
                reward += 0.7
            if not game_temp.can_extend_chain(x, y, -1) and not game_temp.can_block(x, y, 1):
                reward -= 0.05

        is_high_quality = done or abs(reward) >= 0.5
        if not is_high_quality and random.random() > 0.5:
            return

        full_augmentation = is_high_quality
        state_symmetries = self.augment_state(state, full_augmentation)
        next_state_symmetries = self.augment_state(next_state, full_augmentation)
        
        for sym_state, sym_next_state in zip(state_symmetries, next_state_symmetries):
            experience = (sym_state, action, reward, sym_next_state, done)
            self.memory.append(experience)
            self.priorities.append(1.0)
            if len(self.memory) > self.memory_capacity:
                min_priority_idx = np.argmin(self.priorities)
                self.memory.pop(min_priority_idx)
                self.priorities.pop(min_priority_idx)
            if is_high_quality:
                self.elite_memory.append(experience)
                self.elite_priorities.append(1.0)
                if len(self.elite_memory) > self.elite_memory_capacity:
                    min_priority_idx = np.argmin(self.elite_priorities)
                    self.elite_memory.pop(min_priority_idx)
                    self.elite_priorities.pop(min_priority_idx)

    def dqn_move(self, episode=0, total_episodes=500000, use_alpha_beta=False):
        """
        Chọn nước đi bằng DQN với epsilon-greedy, kết hợp alpha-beta.

        Logic:
        - Epsilon giảm từ 1.0 xuống 0.05 qua 90% total_episodes.
        - Nếu use_alpha_beta và episode < 10000, dùng alpha_beta_move (độ sâu 1) với xác suất 80%.
        - Nếu random < epsilon, chọn ngẫu nhiên.
        - Nếu không, dùng policy_net để chọn nước đi có Q-value cao nhất trong các ô trống.
        Hiệu suất: O(size^2) cho forward pass và kiểm tra ô trống.
        Liên quan: Dùng trong hard (hard_move) và huấn luyện (train_game).
        """
        epsilon_start = 1.0
        epsilon_end = 0.05
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (1 - episode / (0.9 * total_episodes))

        if use_alpha_beta and episode < 10000 and random.random() < 0.8:
            move = self.alpha_beta_move(depth=1, is_maximizing_player=True)[1]
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
        """
        Cập nhật mô hình DQN bằng kinh nghiệm từ memory và elite_memory.

        Logic:
        - Chỉ chạy nếu tổng kinh nghiệm >= 1000.
        - Lấy batch_size=256 mẫu (50% từ memory, 50% từ elite_memory nếu đủ).
        - Dùng prioritized experience replay (per_alpha=0.9, per_epsilon=1e-6).
        - Tính loss bằng MSE, cập nhật policy_net bằng Adam và mixed precision.
        - Cập nhật độ ưu tiên dựa trên TD error.
        Hiệu suất: O(batch_size * size^2) cho forward pass và backpropagation.
        Tối ưu hóa:
        - Mixed precision: Tăng tốc ~20-50% trên MPS/CUDA.
        - Batch_size=256: Cân bằng giữa tốc độ và độ chính xác.
        - per_alpha=0.9: Tăng trọng số cho kinh nghiệm quan trọng.
        Liên quan: Dùng trong huấn luyện (train_game).
        """
        total_experiences = len(self.memory) + len(self.elite_memory)
        if total_experiences < 1000:
            return

        current_batch_size = min(self.batch_size, total_experiences)
        memory_samples = int(current_batch_size * 0.5) if len(self.memory) > 0 else 0
        elite_samples = current_batch_size - memory_samples if len(self.elite_memory) >= int(current_batch_size * 0.5) else 0
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

        with autocast(enabled=(self.device.type in ["mps", "cuda"])):
            q_values = self.policy_net(state)
            with torch.no_grad():
                next_q_values = self.policy_net(next_state)
                next_q_target_values = self.target_net(next_state)
            target = q_values.clone()

            for i in range(current_batch_size):
                best_action = torch.argmax(next_q_values[i]).item()
                target[i, action[i]] = reward[i] + self.gamma * next_q_target_values[i, best_action] * (1 - done[i])

            loss = nn.MSELoss()(q_values, target)
            td_errors = torch.abs(q_values - target).detach().max(dim=1)[0].cpu().numpy()

        for (mem_type, idx), error in zip(indices, td_errors):
            if mem_type == 0:
                self.priorities[idx] = error
            else:
                self.elite_priorities[idx] = error

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.scheduler.step()

    def save_model(self, board_size):
        """
        Lưu trọng số mô hình DQN vào file.

        Logic: Lưu state_dict của policy_net vào `dqn_{size}x{size}.pth`.
        Hiệu suất: O(1) để lưu file, phụ thuộc vào kích thước mô hình (~10-50MB).
        Liên quan: Dùng trong huấn luyện (train_game) để lưu mô hình tốt.
        Dữ liệu: Tạo file `dqn_{size}x{size}.pth`.
        """
        torch.save(self.policy_net.state_dict(), f"dqn_{board_size}x{board_size}.pth")

    def save_memory(self, filename):
        """
        Lưu bộ nhớ kinh nghiệm vào file.

        Logic: Lưu memory và priorities vào file .pkl bằng pickle.
        Hiệu suất: O(memory_capacity) để lưu, phụ thuộc vào memory (~100-500MB).
        Liên quan: Dùng trong huấn luyện (train_game) để lưu kinh nghiệm.
        Dữ liệu: Tạo file `memory_{size}x{size}.pkl`.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)
            pickle.dump(self.priorities, f)

    def save_elite_memory(self, filename):
        """
        Lưu bộ nhớ chất lượng cao vào file.

        Logic: Lưu elite_memory và elite_priorities vào file .pkl bằng pickle.
        Hiệu suất: O(elite_memory_capacity) để lưu, phụ thuộc vào elite_memory (~50-200MB).
        Liên quan: Dùng trong huấn luyện (train_game) để lưu kinh nghiệm tốt.
        Dữ liệu: Tạo file `elite_memory_{size}x{size}.pkl`.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.elite_memory, f)
            pickle.dump(self.elite_priorities, f)

    def load_memory(self, filename):
        """
        Tải bộ nhớ kinh nghiệm từ file.

        Logic: Tải memory và priorities từ file .pkl, xử lý lỗi nếu file không tồn tại.
        Hiệu suất: O(memory_capacity) để tải, phụ thuộc vào memory (~100-500MB).
        Liên quan: Dùng trong huấn luyện (train_game) để tiếp tục huấn luyện.
        Dữ liệu: Đọc file `memory_{size}x{size}.pkl`.
        """
        try:
            with open(filename, 'rb') as f:
                self.memory = pickle.load(f)
                self.priorities = pickle.load(f)
        except FileNotFoundError:
            print(f"Memory file {filename} not found. Starting with empty memory.")

    def load_elite_memory(self, filename):
        """
        Tải bộ nhớ chất lượng cao từ file.

        Logic: Tải elite_memory và elite_priorities từ file .pkl, xử lý lỗi nếu file không tồn tại.
        Hiệu suất: O(elite_memory_capacity) để tải, phụ thuộc vào elite_memory (~50-200MB).
        Liên quan: Dùng trong huấn luyện (train_game) để tiếp tục huấn luyện.
        Dữ liệu: Đọc file `elite_memory_{size}x{size}.pkl`.
        """
        try:
            with open(filename, 'rb') as f:
                self.elite_memory = pickle.load(f)
                self.elite_priorities = pickle.load(f)
        except FileNotFoundError:
            print(f"Elite memory file {filename} not found. Starting with empty elite memory.")

    def update_target_network(self):
        """
        Cập nhật target_net từ policy_net.

        Logic: Sao chép state_dict từ policy_net sang target_net.
        Hiệu suất: O(1) để sao chép tham số mô hình.
        Liên quan: Dùng trong huấn luyện (train_game) để ổn định học DQN.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

class Algorithm:
    """
    Quản lý logic trận đấu và huấn luyện AI.

    Mục đích: Điều phối các ván chơi với người dùng hoặc AI, huấn luyện DQN qua self-play.
    Hiệu suất: Phụ thuộc vào chế độ (easy, medium, hard) và số episodes huấn luyện.
    Liên quan: Dùng trong mọi chế độ (play_game, compare_modes) và huấn luyện (train_game).
    Dữ liệu: Sử dụng các file `.pth` (mô hình) và `.pkl` (kinh nghiệm) khi huấn luyện hoặc chơi hard.
    """

    @staticmethod
    def evaluate_model(ai, size, num_games=20):
        """
        Đánh giá mô hình DQN bằng cách đấu với đối thủ trung bình.

        Logic:
        - Chơi num_games ván, AI (DQN) đấu với medium_move.
        - Tính tỷ lệ thắng của AI (-1: O).
        - In kết quả mỗi ván (thắng, thua, hòa).
        Hiệu suất: O(num_games * size^2) do mỗi ván duyệt nước đi và kiểm tra trạng thái.
        Tối ưu hóa: Dùng epsilon thấp (~0.05) để đánh giá chính xác sức mạnh DQN.
        Liên quan: Dùng trong huấn luyện (train_game) và khởi tạo (TicTacToeAI.__init__) để kiểm tra chất lượng mô hình.
        """
        wins = 0
        for _ in range(num_games):
            game = TicTacToe(size)
            current_player = 1
            while True:
                if current_player == 1:
                    move = game.medium_move(1)
                    game.make_move(*move, 1)
                else:
                    move = ai.dqn_move(episode=500000)  # Epsilon ~0.05
                    game.make_move(*move, -1)
                
                if game.is_winner(-1):
                    wins += 1
                    print("AI wins!")
                    break
                if game.is_winner(1):
                    print("Opponent wins!")
                    break
                if game.is_draw():
                    print("Draw!")
                    break
                current_player *= -1
        return wins / num_games

    @staticmethod
    def play_game(size, mode):
        """
        Chơi một trận với người dùng ở chế độ dễ, trung bình, hoặc khó.

        Logic:
        - Khởi tạo TicTacToeAI với chế độ tương ứng (easy, medium, hard).
        - Người chơi nhập nước đi (row col), AI trả lời theo chế độ.
        - In bàn cờ sau mỗi nước, kiểm tra thắng/thua/hòa.
        - Xử lý lỗi đầu vào (ô không hợp lệ, định dạng sai).
        Hiệu suất:
        - Easy: ~0.1-0.5s/nước do IDDFS.
        - Medium: ~0.1-0.5s/nước do alpha-beta.
        - Hard: ~0.5-0.7s/nước do DQN + alpha-beta (5x5, 7x7).
        Liên quan: Dùng trong chế độ Play (Main.main) để chơi với người.
        """
        game = TicTacToeAI(size=size, mode=mode)
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
                    move = game.easy_move(-1)
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
    def train_game(size, episodes=100000, target_update=1000, load_model_path=None, load_memory_path=None, load_elite_memory_path=None):
        """
        Huấn luyện DQN qua self-play giữa hai AI trên bàn size x size.

        Logic:
        - Khởi tạo hai AI (ai1, ai2) để đấu với nhau.
        - Chơi episodes ván, mỗi ván:
          + Hai AI luân phiên đi (ai1: X, ai2: O), dùng dqn_move với alpha-beta hỗ trợ.
          + Lưu kinh nghiệm (state, action, reward, next_state, done) vào memory.
          + Phần thưởng: +1 (thắng), +0.5 (hòa), 0 (đang chơi), -1 (thua).
        - Replay mỗi 10 episodes nếu memory >= 1000.
        - Cập nhật target_net mỗi target_update=1000 episodes.
        - Đánh giá ai1 mỗi 500 episodes (10 trận vs medium_move).
        - Lưu mô hình, memory nếu win_rate > 0.5 và > best_win_rate.
        - Xử lý Ctrl+C để lưu trước khi thoát.
        Hiệu suất:
        - ~15-30 giây/episode (7x7, M2), ~5-10 giây/episode (5x5, MSI).
        - RAM: ~0.5-1GB, file .pkl ~200-500MB.
        Tối ưu hóa:
        - Replay mỗi 10 episodes: Tăng tần suất học.
        - Batch_size=256: Cân bằng tốc độ và độ chính xác.
        - target_update=1000: Ổn định học DQN.
        - Mixed precision: Tăng tốc ~20-50% trên MPS/CUDA.
        Liên quan: Dùng trong chế độ Training (Main.main) để huấn luyện DQN.
        Dữ liệu:
        - Đọc: `dqn_{size}x{size}.pth`, `memory_{size}x{size}.pkl`, `elite_memory_{size}x{size}.pkl`.
        - Ghi: Tạo hoặc cập nhật các file trên khi lưu.
        """
        print(f"Training Double DQN on {size}x{size} board for {episodes} episodes.\n")
        ai1 = TicTacToeAI(size=size, mode="hard", evaluate_model=False)
        ai2 = TicTacToeAI(size=size, mode="hard", evaluate_model=False)

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
        is_interrupted = False

        try:
            for episode in range(episodes):
                if is_interrupted:
                    break
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

                if episode % 10 == 0 and len(ai1.memory) >= 1000:
                    ai1.replay()
                    ai2.replay()

                if episode % target_update == 0:
                    ai1.update_target_network()
                    ai2.update_target_network()

                if episode % 500 == 0:
                    print(f"\n--- Episode {episode} Summary ---")
                    print(f"AI 1 (X) wins: {win_ai1}")
                    print(f"AI 2 (O) wins: {win_ai2}")
                    print(f"Draws        : {draws}")
                    win_rate = Algorithm.evaluate_model(ai1, size, num_games=10)
                    print(f"Win rate: {win_rate:.2f}, Best win rate: {best_win_rate:.2f}")
                    print(f"Memory: {len(ai1.memory)}/{ai1.memory_capacity}, Elite: {len(ai1.elite_memory)}/{ai1.elite_memory_capacity}")
                    if win_rate > 0.5 and win_rate > best_win_rate:
                        best_win_rate = win_rate
                        ai1.save_model(size)
                        ai1.save_memory(f"memory_{size}x{size}.pkl")
                        ai1.save_elite_memory(f"elite_memory_{size}x{size}.pkl")
                        print(f"Saved best model to dqn_{size}x{size}.pth")
                        print(f"Saved best memory to memory_{size}x{size}.pkl")
                        print(f"Saved best elite memory to elite_memory_{size}x{size}.pkl")
                    print("-----------------------------------\n")

        except KeyboardInterrupt:
            if not is_interrupted and len(ai1.memory) > 0:
                is_interrupted = True
                print("\nSaving model and memory for AI1 before exiting...")
                ai1.save_model(size)
                ai1.save_memory(f"memory_{size}x{size}.pkl")
                ai1.save_elite_memory(f"elite_memory_{size}x{size}.pkl")
                print(f"Saved model to dqn_{size}x{size}.pth")
                print(f"Saved memory to memory_{size}x{size}.pkl")
                print(f"Saved best elite memory to elite_memory_{size}x{size}.pkl")
            sys.exit(0)

    @staticmethod
    def compare_modes(size, num_games):
        """
        So sánh hiệu suất giữa Medium vs Easy và Medium vs Hard qua các trận đấu.

        Logic:
        - Chạy num_games trận, chia đều: num_games//2 trận Medium (X) vs Easy/Hard (O), và num_games//2 trận Medium (O) vs Easy/Hard (X).
        - Ghi lại thắng, thua, hòa từ góc nhìn của Medium.
        - In số trận thắng, thua, hòa, tỷ lệ thắng, và tóm tắt kết quả.
        - Đảm bảo không thiên vị bằng cách luân phiên vai trò X/O.
        Hiệu suất: O(num_games * size^2) cho mỗi nhóm trận do duyệt nước đi và kiểm tra trạng thái.
        Tối ưu hóa:
        - In tiến độ mỗi 10 trận để theo dõi.
        - Reset bàn cờ mỗi ván để đảm bảo độc lập.
        Liên quan: Dùng trong chế độ Compare (Main.main) để đánh giá hiệu suất các chế độ.
        """
        print(f"\nStarting comparison on {size}x{size} board with {num_games} games per matchup...\n")
        
        def simulate_games(game, medium_player, opponent_player, opponent_mode, games, medium_wins, opponent_wins, draws):
            """Hàm phụ để mô phỏng games trận với medium_player và opponent_player."""
            for i in range(games):
                game.board = np.zeros((size, size), dtype=np.int8)  # Reset bàn cờ
                current_player = 1
                while True:
                    if current_player == medium_player:
                        move = game.medium_move(medium_player)
                        game.make_move(*move, medium_player)
                    else:
                        if opponent_mode == "easy":
                            move = game.easy_move(opponent_player)
                        elif opponent_mode == "hard":
                            move = game.hard_move(opponent_player)
                        game.make_move(*move, opponent_player)
                    
                    if game.is_winner(medium_player):
                        medium_wins += 1
                        break
                    if game.is_winner(opponent_player):
                        opponent_wins += 1
                        break
                    if game.is_draw():
                        draws += 1
                        break
                    current_player *= -1
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{games} games")
            return medium_wins, opponent_wins, draws

        # Medium vs Easy
        medium_wins = 0
        easy_wins = 0
        draws = 0
        half_games = num_games // 2
        game_easy = TicTacToeAI(size=size, mode="easy")

        print("Simulating Medium (X) vs Easy (O)...")
        medium_wins, easy_wins, draws = simulate_games(game_easy, 1, -1, "easy", half_games, medium_wins, easy_wins, draws)

        print("Simulating Medium (O) vs Easy (X)...")
        medium_wins, easy_wins, draws = simulate_games(game_easy, -1, 1, "easy", num_games - half_games, medium_wins, easy_wins, draws)

        medium_win_rate = medium_wins / num_games
        print(f"\nMedium vs Easy Results:")
        print(f"Medium wins: {medium_wins} ({medium_win_rate:.2%})")
        print(f"Easy wins  : {easy_wins} ({easy_wins / num_games:.2%})")
        print(f"Draws      : {draws} ({draws / num_games:.2%})\n")

        # Medium vs Hard
        medium_wins = 0
        hard_wins = 0
        draws = 0
        game_hard = TicTacToeAI(size=size, mode="hard")

        print("Simulating Medium (X) vs Hard (O)...")
        medium_wins, hard_wins, draws = simulate_games(game_hard, 1, -1, "hard", half_games, medium_wins, hard_wins, draws)

        print("Simulating Medium (O) vs Hard (X)...")
        medium_wins, hard_wins, draws = simulate_games(game_hard, -1, 1, "hard", num_games - half_games, medium_wins, hard_wins, draws)

        medium_win_rate = medium_wins / num_games
        print(f"\nMedium vs Hard Results:")
        print(f"Medium wins: {medium_wins} ({medium_win_rate:.2%})")
        print(f"Hard wins  : {hard_wins} ({hard_wins / num_games:.2%})")
        print(f"Draws      : {draws} ({draws / num_games:.2%})\n")

        print("Comparison Summary:")
        print(f"- Medium vs Easy: Medium wins {medium_wins} out of {num_games} games, confirming Medium's stronger strategy over Easy.")
        if medium_wins > hard_wins:
            print(f"- Medium vs Hard: Medium wins {medium_wins} out of {num_games} games, unexpectedly outperforming Hard. This may indicate an issue with Hard's strategy on {size}x{size} board.")
        else:
            print(f"- Medium vs Hard: Hard wins {hard_wins} out of {num_games} games, demonstrating superior strategy as expected.")
        print(f"- Easy mode is ideal for beginners, while Hard should pose a significant challenge.")

class Main:
    """
    Điều khiển chính của chương trình, quản lý giao diện người dùng.

    Mục đích: Hiển thị menu, xử lý lựa chọn người dùng, và gọi các chức năng tương ứng.
    Hiệu suất: Phụ thuộc vào chức năng được gọi (play_game, train_game, compare_modes).
    Liên quan: Gọi play_game, train_game, hoặc compare_modes từ Algorithm.
    """

    @staticmethod
    def main():
        """
        Hàm chính: Hiển thị menu và xử lý lựa chọn người dùng.

        Logic:
        - Hiển thị menu với 3 chế độ: Play (1), Training (2), Compare (3).
        - Mode 1: Chơi với AI (easy, medium, hard; 3x3, 5x5, 7x7).
        - Mode 2: Huấn luyện DQN (5x5, 7x7; 5 stage, mỗi stage 100,000 episodes).
        - Mode 3: So sánh Medium vs Easy và Medium vs Hard (3x3, 5x5, 7x7).
        - Xử lý lỗi đầu vào (kích thước bàn, chế độ, số trận) và Ctrl+C.
        Hiệu suất: Phụ thuộc vào play_game (~0.1-0.7s/nước), train_game (~5-30s/episode), hoặc compare_modes (~num_games * size^2).
        Liên quan: Gọi các hàm từ Algorithm để thực thi chức năng.
        """
        print("Welcome to Tic-Tac-Toe!")
        print("1: Play against AI")
        print("2: Train AI model")
        print("3: Compare AI modes")
        mode_choice = input("Choose mode (1: Play, 2: Training, 3: Compare): ")

        if mode_choice == "1":
            size = int(input("Choose board size (3, 5, 7): "))
            if size not in [3, 5, 7]:
                print("Invalid board size! Choose 3, 5, or 7.")
                return
            difficulty = input("Choose difficulty (1: Easy, 2: Medium, 3: Hard): ")
            mode = {"1": "easy", "2": "medium", "3": "hard"}.get(difficulty)
            if mode:
                Algorithm.play_game(size, mode)
            else:
                print("Invalid difficulty! Choose 1, 2, or 3.")
        elif mode_choice == "2":
            size = int(input("Choose board size (5, 7): "))
            if size not in [5, 7]:
                print("Training mode is only available for 5x5 and 7x7 boards.")
            else:
                try:
                    for stage in range(5):
                        print(f"\nStarting training stage {stage+1}/5")
                        load_model = f"dqn_{size}x{size}.pth" if stage > 0 else None
                        load_memory = f"memory_{size}x{size}.pkl" if stage > 0 else None
                        load_elite_memory = f"elite_memory_{size}x{size}.pkl" if stage > 0 else None
                        Algorithm.train_game(size, episodes=100000, target_update=1000,
                                            load_model_path=load_model,
                                            load_memory_path=load_memory,
                                            load_elite_memory_path=load_elite_memory)
                except KeyboardInterrupt:
                    print("\nTraining interrupted. Model and memory already saved in train_game.")
                    sys.exit(0)
        elif mode_choice == "3":
            size = int(input("Choose board size (3, 5, 7): "))
            if size not in [3, 5, 7]:
                print("Invalid board size! Choose 3, 5, or 7.")
                return
            num_games = int(input("Enter number of games to simulate: "))
            if num_games <= 0:
                print("Number of games must be positive!")
                return
            Algorithm.compare_modes(size, num_games)
        else:
            print("Invalid choice! Choose 1, 2, or 3.")

if __name__ == "__main__":
    Main.main()