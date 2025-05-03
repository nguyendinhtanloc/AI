import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pickle
import os
import time
from copy import deepcopy

class TicTacToe:
    """
    Lớp quản lý trò chơi Tic-Tac-Toe, xử lý bàn cờ và logic thắng/thua/hòa.
    - Mục đích: Cung cấp các hàm cơ bản để chơi Tic-Tac-Toe trên bàn cờ size x size.
    - Hiệu suất: Các hàm kiểm tra thắng/thua/hòa có độ phức tạp O(size^2) hoặc thấp hơn.
    - Liên quan: Được dùng trong cả easy, medium, và hard.
    - Dữ liệu: Không sử dụng file lưu trữ trực tiếp.
    """
    
    def __init__(self, size=3):
        """
        Khởi tạo bàn cờ size x size với trạng thái rỗng.
        - Logic: Tạo ma trận 0 (ô trống), 1 (X: người chơi), -1 (O: AI).
        - Hiệu suất: O(size^2) để tạo ma trận.
        - Liên quan: Dùng cho mọi chế độ.
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)

    def is_winner(self, player):
        """
        Kiểm tra xem người chơi (1: X, -1: O) có thắng không bằng cách tìm chuỗi liên tiếp.
        - Logic:
          - Kiểm tra hàng, cột, đường chéo chính/phụ với độ dài thắng (3 cho 3x3, 4 cho 5x5, 5 cho 7x7).
          - Trả về True nếu tìm thấy chuỗi.
        - Hiệu suất: O(size^2) do duyệt tất cả hàng, cột, và đường chéo.
        - Liên quan: Dùng trong easy, medium, hard để xác định kết thúc ván.
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
        Kiểm tra trạng thái hòa: bàn cờ đầy và không ai thắng.
        - Logic: Kiểm tra nếu không còn ô trống và không có người thắng.
        - Hiệu suất: O(size^2) để kiểm tra ô trống, cộng với O(size^2) từ is_winner.
        - Liên quan: Dùng trong mọi chế độ để kết thúc ván.
        """
        return not np.any(self.board == 0) and not self.is_winner(1) and not self.is_winner(-1)

    def get_available_moves(self):
        """
        Trả về danh sách các ô trống (nước đi hợp lệ) dưới dạng [(x, y)].
        - Logic: Tìm tất cả vị trí có giá trị 0 trên bàn cờ.
        - Hiệu suất: O(size^2) để duyệt ma trận.
        - Liên quan: Dùng trong mọi chế độ để xác định nước đi có thể.
        """
        indices = np.where(self.board == 0)
        return list(zip(indices[0], indices[1]))

    def make_move(self, x, y, player):
        """
        Thực hiện nước đi tại (x, y) cho người chơi (1: X, -1: O).
        - Logic: Đặt giá trị player tại (x, y) nếu ô trống, trả về True nếu thành công.
        - Hiệu suất: O(1) để truy cập và cập nhật ô.
        - Liên quan: Dùng trong mọi chế độ để cập nhật bàn cờ.
        """
        if self.board[x, y] == 0:
            self.board[x, y] = player
            return True
        return False

    def print_board(self):
        """
        In bàn cờ với ký hiệu: . (trống), X (người chơi), O (AI).
        - Logic: Chuyển ma trận thành ký hiệu và in từng hàng.
        - Hiệu suất: O(size^2) để duyệt và in.
        - Liên quan: Dùng trong mọi chế độ để hiển thị trạng thái.
        """
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()

    def score_move(self, x, y, player):
        """
        Đánh giá nước đi tại (x, y) dựa trên chuỗi, chặn, và vị trí trung tâm.
        - Logic:
          - +5 nếu tạo chuỗi 3 (5x5) hoặc 4 (7x7).
          - +4 nếu chặn chuỗi đối thủ.
          - +3 nếu tạo chuỗi 2 với ít nhất 1 đầu mở.
          - +0-1 dựa trên khoảng cách đến trung tâm.
        - Hiệu suất: O(1) vì chỉ kiểm tra các ô lân cận (tối đa 8 hướng).
        - Liên quan: Dùng trong easy (iddfs_move) và medium (alpha_beta_move) để sắp xếp nước đi.
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
        Chế độ dễ: Sử dụng IDDFS với độ sâu tối đa tùy kích thước bàn cờ.
        - Logic:
          1. Kiểm tra thắng ngay lập tức.
          2. Kiểm tra chặn đối thủ thắng.
          3. Dùng IDDFS với độ sâu 2-3 (3x3, 5x5: 3; 7x7: 2-3 tùy ô trống).
          4. Nếu hết thời gian, chọn ngẫu nhiên.
        - Hiệu suất:
          - IDDFS: O(b^d) với b ~ size^2, d = 2-3.
          - Thời gian: ~0.1-0.5s tùy kích thước.
        - Liên quan: Chỉ dùng trong chế độ easy, yếu hơn medium và hard.
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
        if self.size == 7 and available_moves > 20:
            max_depth = 2
        time_limit = 0.2 if self.size == 3 else 0.5 if self.size == 5 else 0.8
        move = self.iddfs_move(player, max_depth=max_depth, time_limit=time_limit)
        if move is None:
            return random.choice(self.get_available_moves())
        return move

    def iddfs_move(self, player, max_depth, time_limit):
        """
        Tìm nước đi tốt nhất bằng IDDFS với giới hạn thời gian.
        - Logic:
          - Duyệt từng độ sâu từ 1 đến max_depth.
          - Sắp xếp nước đi theo score_move để ưu tiên nước tốt.
          - Dùng depth_limited_dfs để đánh giá.
          - Dừng nếu vượt thời gian.
        - Hiệu suất: O(b^d) với b ~ size^2, d = 1 đến 3.
        - Liên quan: Chỉ dùng trong easy (easy_move).
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
        DFS với độ sâu giới hạn và alpha-beta pruning.
        - Logic:
          - Nếu hết độ sâu hoặc kết thúc ván, trả về điểm từ evaluate_board.
          - Nếu maximizing (AI), chọn nước đi tối đa điểm.
          - Nếu minimizing (người chơi), chọn nước đi tối thiểu điểm.
          - Sắp xếp nước đi theo score_move để cắt tỉa hiệu quả.
        - Hiệu suất: O(b^d) nhưng giảm nhờ alpha-beta.
        - Liên quan: Chỉ dùng trong easy (iddfs_move).
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
        Chế độ trung bình: Kết hợp kiểm tra thắng/chặn, tạo chuỗi, và alpha-beta.
        - Logic:
          1. Kiểm tra thắng ngay lập tức.
          2. Kiểm tra chặn đối thủ thắng.
          3. Tìm nước đi tạo chuỗi 3 (5x5) hoặc 4 (7x7) bằng can_extend_chain.
          4. Tìm nước đi chặn chuỗi đối thủ bằng can_block.
          5. Với xác suất 10%, chọn ngẫu nhiên.
          6. Dùng alpha_beta_move với độ sâu 2.
        - Hiệu suất:
          - Alpha-beta: O(b^d) với b ~ size^2, d = 2.
          - Thời gian: ~0.1-0.5s tùy kích thước.
        - Liên quan: Chỉ dùng trong chế độ medium, mạnh hơn easy nhưng yếu hơn hard.
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
                print(f"Blocking move at {block_move} to stop opponent from winning!")
                return block_move
        if random.random() < 0.1:
            return random.choice(self.get_available_moves())
        _, move = self.alpha_beta_move(depth=depth, is_maximizing_player=(player == -1))
        return move

    def can_extend_chain(self, x, y, player):
        """
        Kiểm tra xem đặt tại (x, y) có tạo chuỗi 3 (5x5) hoặc 4 (7x7) quân không.
        - Logic:
          - Kiểm tra 8 hướng, đếm số quân liên tiếp của player.
          - Trả về True nếu tạo chuỗi đủ dài hoặc nối hai quân thành chuỗi.
        - Hiệu suất: O(1) vì chỉ kiểm tra tối đa 4 ô mỗi hướng.
        - Liên quan: Dùng trong medium (medium_move) và hard (hard_move).
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
        Kiểm tra xem đặt tại (x, y) có chặn được chuỗi nguy hiểm của đối thủ không.
        - Logic:
          - Kiểm tra 8 hướng, đếm số quân liên tiếp của đối thủ.
          - Trả về vị trí chặn nếu chuỗi đạt win_length-1 hoặc win_length-1 với ô trống.
        - Hiệu suất: O(1) vì chỉ kiểm tra tối đa 4 ô mỗi hướng.
        - Liên quan: Dùng trong medium (medium_move) và hard (hard_move).
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
        Đánh giá bàn cờ: ưu tiên thắng/thua, chuỗi 3, và chuỗi 2 tiềm năng.
        - Logic:
          - +10 nếu AI thắng, -10 nếu người chơi thắng.
          - +2/-2 cho chuỗi 3 (5x5) hoặc 2 (3x3) với 1 ô trống.
          - +1/-1 cho chuỗi 2 tiềm năng.
          - +0.1/-0.1 cho quân gần trung tâm.
        - Hiệu suất: O(size^2) để duyệt bàn cờ.
        - Liên quan: Dùng trong easy (depth_limited_dfs) và medium (alpha_beta_move).
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
        Tìm nước đi tối ưu bằng thuật toán alpha-beta pruning.
        - Logic:
          - Nếu hết độ sâu hoặc kết thúc ván, trả về điểm từ evaluate_board.
          - Nếu maximizing (AI), chọn nước đi tối đa điểm.
          - Nếu minimizing (người chơi), chọn nước đi tối thiểu điểm.
          - Sắp xếp nước đi theo khoảng cách đến trung tâm để cắt tỉa hiệu quả.
        - Hiệu suất: O(b^d) với b ~ size^2, d = 1-2, giảm nhờ alpha-beta.
        - Liên quan: Dùng trong medium (medium_move) và hard (hard_move).
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
        Chế độ khó: Kết hợp DQN với kiểm tra chuỗi và alpha-beta nhẹ.
        - Mục đích: Tạo AI mạnh hơn medium và easy, nhưng không quá khó như người chơi chuyên nghiệp.
        - Dữ liệu:
          - Dùng trọng số từ file `dqn_{size}x{size}.pth` (tải trong TicTacToeAI.__init__).
          - File `memory_{size}x{size}.pkl` và `elite_memory_{size}x{size}.pkl` chỉ dùng khi huấn luyện.
        - Logic:
          1. Kiểm tra thắng ngay lập tức (giống easy, medium).
          2. Kiểm tra chặn đối thủ thắng (giống easy, medium).
          3. Tìm nước đi tạo chuỗi 3 (5x5) hoặc 4 (7x7) bằng can_extend_chain.
          4. Tìm nước đi chặn chuỗi đối thủ bằng can_block.
          5. Dùng dqn_move (epsilon thấp, ưu tiên khai thác) để chọn nước đi.
          6. Kiểm tra lại nước đi DQN bằng alpha_beta_move (độ sâu 1) để chọn nước tốt nhất.
        - Hiệu suất:
          - DQN: O(size^2) cho forward pass qua CNN.
          - Alpha-beta (độ sâu 1): O(size^2).
          - Tổng thời gian: ~0.2-0.5s (5x5), ~0.3-0.8s (7x7).
        - Liên quan: Chỉ dùng trong chế độ hard, mạnh hơn medium (alpha-beta độ sâu 2) và easy (IDDFS).
        """
        opponent = -player
        # Kiểm tra thắng ngay
        for i, j in self.get_available_moves():
            self.board[i, j] = player
            if self.is_winner(player):
                self.board[i, j] = 0
                return (i, j)
            self.board[i, j] = 0
        # Kiểm tra chặn đối thủ
        for i, j in self.get_available_moves():
            self.board[i, j] = opponent
            if self.is_winner(opponent):
                self.board[i, j] = 0
                return (i, j)
            self.board[i, j] = 0
        # Kiểm tra tạo chuỗi
        for i, j in self.get_available_moves():
            if self.can_extend_chain(i, j, player):
                return (i, j)
        # Kiểm tra chặn chuỗi đối thủ
        for i, j in self.get_available_moves():
            block_move = self.can_block(i, j, opponent)
            if block_move:
                print(f"Blocking move at {block_move} to stop opponent from winning!")
                return block_move
        # Sử dụng DQN
        dqn_move = self.dqn_move(episode=100000)
        # Kiểm tra lại nước đi DQN bằng alpha-beta (độ sâu 1)
        moves = self.get_available_moves()
        best_score = -np.inf
        best_move = dqn_move
        for i, j in moves:
            self.board[i, j] = player
            score, _ = self.alpha_beta_move(depth=1, is_maximizing_player=False)
            self.board[i, j] = 0
            if score > best_score:
                best_score = score
                best_move = (i, j)
        return best_move

class TicTacToeAI(TicTacToe):
    """
    Lớp mở rộng TicTacToe với DQN, sử dụng CNN và học tăng cường.
    - Mục đích: Triển khai AI thông minh dựa trên DQN cho chế độ hard, hỗ trợ huấn luyện và chơi.
    - Hiệu suất:
      - Khởi tạo: O(1) trừ khi tải mô hình (O(size^2) cho CNN).
      - Huấn luyện: Phụ thuộc số episodes, batch size (128), và kích thước bàn cờ.
    - Liên quan: Chủ yếu dùng trong hard (hard_move, dqn_move) và huấn luyện.
    - Dữ liệu:
      - `dqn_{size}x{size}.pth`: Trọng số mô hình DQN, dùng khi đấu (hard).
      - `memory_{size}x{size}.pkl`, `elite_memory_{size}x{size}.pkl`: Kinh nghiệm, dùng khi huấn luyện.
    """
    
    def __init__(self, size=3, mode="hard"):
        """
        Khởi tạo AI với mô hình CNN, bộ nhớ, và tham số huấn luyện.
        - Logic:
          - Tạo policy_net và target_net (CNN với 4 lớp conv, batch norm, dropout).
          - Chỉ tải trọng số từ `dqn_{size}x{size}.pth` nếu chế độ là hard và file tồn tại.
          - Xử lý lỗi nếu file không tương thích, bỏ qua và dùng mô hình mới.
          - Khởi tạo bộ nhớ (memory, elite_memory) và optimizer.
        - Hiệu suất: O(1) để khởi tạo, O(size^2) nếu tải mô hình.
        - Liên quan: Ảnh hưởng hard (tải mô hình cho hard_move) và huấn luyện.
        - Sửa đổi (KHẮC PHỤC LỖI):
          - Thêm tham số mode để chỉ tải mô hình trong chế độ hard.
          - Thêm try-except để xử lý lỗi tải file không tương thích.
        """
        super().__init__(size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.policy_net = self.create_model(size).to(self.device)
        self.target_net = self.create_model(size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50000, gamma=0.1)
        self.memory = []
        self.memory_capacity = 50000
        self.priorities = []
        self.elite_memory = []
        self.elite_memory_capacity = 10000
        self.elite_priorities = []
        self.batch_size = 128
        self.gamma = 0.995
        self.per_epsilon = 1e-6
        self.per_alpha = 0.8
        # Chỉ tải mô hình nếu ở chế độ hard
        if mode == "hard":
            model_path = f"dqn_{size}x{size}.pth"
            if os.path.exists(model_path):
                try:
                    self.policy_net.load_state_dict(torch.load(model_path))
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    print(f"Loaded pre-trained model from {model_path}")
                except RuntimeError as e:
                    print(f"Error loading model from {model_path}: {e}")
                    print("Using new model instead.")

    def create_model(self, size):
        """
        Tạo mô hình CNN cải tiến với 4 lớp convolution, batch norm, và dropout.
        - Logic:
          - Input: Ma trận size x size (1 kênh).
          - 4 lớp Conv2d: 32, 64, 128, 256 filters, kernel 3x3, padding 1.
          - BatchNorm2d và ReLU sau mỗi lớp conv.
          - 2 lớp fully connected: 256*size*size -> 512 -> size*size.
          - Dropout (0.2) trước lớp cuối.
        - Hiệu suất: O(size^2) cho forward pass.
        - Liên quan: Dùng trong hard (dqn_move) và huấn luyện.
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
                self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                self.bn4 = nn.BatchNorm2d(256)
                self.fc1 = nn.Linear(256 * size * size, 512)
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(512, size * size)

            def forward(self, x):
                x = x.view(-1, 1, size, size)
                x = torch.relu(self.bn1(self.conv1(x)))
                x = torch.relu(self.bn2(self.conv2(x)))
                x = torch.relu(self.bn3(self.conv3(x)))
                x = torch.relu(self.bn4(self.conv4(x)))
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        return DQN(size)

    def augment_state(self, state, full_augmentation=False):
        """
        Tạo các phiên bản đối xứng của trạng thái bàn cờ để tăng dữ liệu huấn luyện.
        - Logic:
          - Nếu full_augmentation: Tạo 6 phiên bản (xoay 90/180/270, lật ngang/dọc).
          - Nếu không: Tạo 2 phiên bản (xoay 90, lật ngang).
        - Hiệu suất: O(size^2) để xoay/lật ma trận.
        - Liên quan: Dùng trong huấn luyện (store_experience).
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
        Lưu kinh nghiệm (state, action, reward, next_state, done) vào memory hoặc elite_memory.
        - Logic:
          - Điều chỉnh reward dựa trên can_extend_chain (+0.2), can_block (+0.3), hoặc không chiến lược (-0.05).
          - Lưu vào memory nếu kinh nghiệm chất lượng cao (done hoặc |reward| >= 0.5).
          - Dùng augment_state để tạo phiên bản đối xứng (6 nếu chất lượng cao, 2 nếu không).
          - Giới hạn memory (50,000), elite_memory (10,000).
        - Hiệu suất: O(size^2) cho augment_state và kiểm tra chuỗi.
        - Liên quan: Dùng trong huấn luyện (train_game).
        """
        if not done:
            game_temp = TicTacToe(self.size)
            game_temp.board = next_state
            x, y = divmod(action, self.size)
            if game_temp.can_extend_chain(x, y, -1):
                reward += 0.2
            if game_temp.can_block(x, y, 1):
                reward += 0.3
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
        Chọn nước đi với epsilon-greedy cải tiến cho DQN.
        - Logic:
          - Nếu episode < 5000 và random < 0.5, dùng alpha_beta_move (độ sâu 2).
          - Tính epsilon giảm dần từ 1.0 đến 0.05 dựa trên episode.
          - Nếu random < epsilon, chọn ngẫu nhiên.
          - Nếu không, dùng policy_net dự đoán Q-values, chọn nước đi tốt nhất trong ô trống.
        - Hiệu suất: O(size^2) cho forward pass qua CNN.
        - Liên quan: Dùng trong hard (hard_move) và huấn luyện (train_game).
        """
        epsilon_start = 1.0
        epsilon_end = 0.05
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (1 - episode / (0.9 * total_episodes))

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
        """
        Huấn luyện mô hình bằng cách lấy mẫu từ memory và elite_memory.
        - Logic:
          - Lấy mẫu batch (128) với 50% từ memory, 50% từ elite_memory.
          - Dùng prioritized experience replay với alpha=0.8, epsilon=1e-6.
          - Tính Q-values và target bằng Double DQN (policy_net và target_net).
          - Cập nhật trọng số bằng Adam, clip gradient norm <= 1.0.
          - Giảm learning rate mỗi 50,000 episodes (gamma=0.1).
        - Hiệu suất: O(batch_size * size^2) cho mỗi lần replay.
        - Liên quan: Dùng trong huấn luyện (train_game).
        """
        total_experiences = len(self.memory) + len(self.elite_memory)
        if total_experiences < 64:
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

    def save_model(self, board_size):
        """
        Lưu trọng số mô hình vào file `dqn_{size}x{size}.pth`.
        - Logic: Lưu state_dict của policy_net.
        - Hiệu suất: O(size^2) để ghi file.
        - Liên quan: Dùng trong huấn luyện (train_game), ảnh hưởng hard (tải trong __init__).
        """
        torch.save(self.policy_net.state_dict(), f"dqn_{board_size}x{board_size}.pth")

    def save_memory(self, filename):
        """
        Lưu bộ nhớ kinh nghiệm vào file `.pkl`.
        - Logic: Lưu memory và priorities.
        - Hiệu suất: O(memory_capacity) để ghi file.
        - Liên quan: Dùng trong huấn luyện (train_game).
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)
            pickle.dump(self.priorities, f)

    def save_elite_memory(self, filename):
        """
        Lưu bộ nhớ chất lượng cao vào file `.pkl`.
        - Logic: Lưu elite_memory và elite_priorities.
        - Hiệu suất: O(elite_memory_capacity) để ghi file.
        - Liên quan: Dùng trong huấn luyện (train_game).
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.elite_memory, f)
            pickle.dump(self.elite_priorities, f)

    def load_memory(self, filename):
        """
        Tải bộ nhớ kinh nghiệm từ file `.pkl`.
        - Logic: Tải memory và priorities, xử lý lỗi nếu file không tồn tại.
        - Hiệu suất: O(memory_capacity) để đọc file.
        - Liên quan: Dùng trong huấn luyện (train_game).
        """
        try:
            with open(filename, 'rb') as f:
                self.memory = pickle.load(f)
                self.priorities = pickle.load(f)
        except FileNotFoundError:
            print(f"Memory file {filename} not found. Starting with empty memory.")

    def load_elite_memory(self, filename):
        """
        Tải bộ nhớ chất lượng cao từ file `.pkl`.
        - Logic: Tải elite_memory và elite_priorities, xử lý lỗi nếu file không tồn tại.
        - Hiệu suất: O(elite_memory_capacity) để đọc file.
        - Liên quan: Dùng trong huấn luyện (train_game).
        """
        try:
            with open(filename, 'rb') as f:
                self.elite_memory = pickle.load(f)
                self.elite_priorities = pickle.load(f)
        except FileNotFoundError:
            print(f"Elite memory file {filename} not found. Starting with empty elite memory.")

    def update_target_network(self):
        """
        Cập nhật trọng số target_net từ policy_net.
        - Logic: Sao chép state_dict từ policy_net sang target_net.
        - Hiệu suất: O(size^2) để sao chép trọng số.
        - Liên quan: Dùng trong huấn luyện (train_game).
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

class Algorithm:
    """
    Lớp xử lý logic trận đấu và huấn luyện AI.
    - Mục đích: Quản lý ván chơi với người dùng và huấn luyện DQN qua self-play.
    - Hiệu suất: Phụ thuộc vào chế độ (easy, medium, hard) và số episodes huấn luyện.
    - Liên quan: Dùng trong mọi chế độ và huấn luyện.
    - Dữ liệu: Sử dụng các file `.pth` và `.pkl` khi huấn luyện hoặc chơi hard.
    """
    
    @staticmethod
    def evaluate_model(ai, size, num_games=50):
        """
        Đánh giá mô hình DQN bằng cách đấu với đối thủ trung bình.
        - Logic:
          - Chơi num_games ván, AI (DQN) đấu với medium_move.
          - Tính tỷ lệ thắng của AI.
        - Hiệu suất: O(num_games * size^2) cho mỗi ván.
        - Liên quan: Dùng trong huấn luyện (train_game) để kiểm tra chất lượng mô hình.
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
                    move = ai.dqn_move(episode=100000)
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
        Chơi một trận với người chơi, hỗ trợ chế độ dễ, trung bình, khó.
        - Logic:
          - Khởi tạo TicTacToeAI với mode tương ứng (easy, medium, hard).
          - Người chơi nhập nước đi (row col), AI trả lời theo chế độ.
          - In bàn cờ sau mỗi nước, kiểm tra thắng/thua/hòa.
        - Hiệu suất: Phụ thuộc chế độ:
          - Easy: ~0.1-0.5s.
          - Medium: ~0.1-0.5s.
          - Hard: ~0.2-0.8s.
        - Liên quan: Dùng trong mọi chế độ khi chơi với người.
        - Sửa đổi (KHẮC PHỤC LỖI):
          - Truyền tham số mode vào TicTacToeAI để chỉ tải mô hình DQN trong chế độ hard.
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
    def train_game(size, episodes=100000, target_update=500, load_model_path=None, load_memory_path=None, load_elite_memory_path=None):
        """
        Huấn luyện hai AI đấu với nhau, lưu mô hình và bộ nhớ tốt nhất.
        - Logic:
          - Tạo hai TicTacToeAI, tải mô hình/memory nếu có.
          - Chơi episodes ván, mỗi ván:
            - Dùng dqn_move (epsilon-greedy) để chọn nước đi.
            - Lưu kinh nghiệm vào memory/elite_memory.
            - Huấn luyện bằng replay nếu đủ kinh nghiệm.
          - Cập nhật target_net mỗi target_update episodes.
          - Lưu mô hình/memory nếu win_rate cải thiện.
          - Xử lý Ctrl+C để lưu trước khi thoát.
        - Hiệu suất: O(episodes * size^2) cho toàn bộ huấn luyện.
        - Liên quan: Dùng để tạo `dqn_{size}x{size}.pth` cho hard.
        - Dữ liệu:
          - Đọc: `dqn_{size}x{size}.pth`, `memory_{size}x{size}.pkl`, `elite_memory_{size}x{size}.pkl`.
          - Ghi: Ghi đè các file trên khi cải thiện hoặc Ctrl+C.
        """
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
                    print(f"Win rate: {win_rate:.2f}, Best win rate: {best_win_rate:.2f}")
                    print(f"Memory: {len(ai1.memory)}/{ai1.memory_capacity}, Elite: {len(ai1.elite_memory)}/{ai1.elite_memory_capacity}")
                    if win_rate > best_win_rate:
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
                print(f"Saved elite memory to elite_memory_{size}x{size}.pkl")
            sys.exit(0)

class Main:
    """
    Lớp chính để chạy chương trình, quản lý giao diện người dùng.
    - Mục đích: Hiển thị menu, xử lý lựa chọn chơi hoặc huấn luyện.
    - Hiệu suất: O(1) cho menu, phụ thuộc vào play_game hoặc train_game.
    - Liên quan: Dùng để khởi động mọi chế độ và huấn luyện.
    """
    
    @staticmethod
    def main():
        """
        Hàm chính: Hiển thị menu và xử lý lựa chọn người dùng.
        - Logic:
          - Mode 1: Chơi với AI (easy, medium, hard) trên kích thước 3, 5, 7.
          - Mode 2: Huấn luyện DQN trên kích thước 5, 7 với 5 stage, mỗi stage 100,000 episodes.
          - Xử lý lỗi đầu vào và Ctrl+C.
        - Hiệu suất: Phụ thuộc vào play_game hoặc train_game.
        - Liên quan: Gọi play_game (easy, medium, hard) hoặc train_game (hard).
        """
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
                try:
                    for stage in range(5):
                        print(f"\nStarting training stage {stage+1}/5")
                        load_model = f"dqn_{size}x{size}.pth" if stage > 0 else None
                        load_memory = f"memory_{size}x{size}.pkl" if stage > 0 else None
                        load_elite_memory = f"elite_memory_{size}x{size}.pkl" if stage > 0 else None
                        Algorithm.train_game(size, episodes=100000, target_update=500,
                                            load_model_path=load_model,
                                            load_memory_path=load_memory,
                                            load_elite_memory_path=load_elite_memory)
                except KeyboardInterrupt:
                    print("\nTraining interrupted. Model and memory already saved in train_game.")
                    sys.exit(0)
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    Main.main()