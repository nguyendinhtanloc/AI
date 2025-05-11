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

    def is_winner(self, player, last_move=None):
        """
        Kiểm tra xem người chơi (1: X, -1: O) có thắng không bằng cách tìm chuỗi liên tiếp.
        - Nếu last_move được cung cấp và hợp lệ, chỉ kiểm tra hàng, cột, đường chéo qua ô đó, độ phức tạp O(win_length).
        - Nếu last_move là None hoặc không hợp lệ, duyệt toàn bộ bàn cờ, độ phức tạp O(size^2).
        - Logic: Tìm chuỗi dài win_length (3 cho 3x3, 4 cho 5x5, 5 cho 7x7).
        - Trường hợp biên: Trả về False nếu bàn cờ rỗng hoặc last_move ngoài biên.
        - Liên quan: Dùng trong mọi chế độ để kiểm tra thắng.
        """
        win_length = 3 if self.size == 3 else 4 if self.size == 5 else 5

        if last_move is not None:
            x, y = last_move
            # Kiểm tra tính hợp lệ của last_move
            if not (0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == player):
                return False
            # Kiểm tra hàng
            start_y = max(0, y - win_length + 1)
            end_y = min(self.size - win_length + 1, y + 1)
            for j in range(start_y, end_y):
                if np.all(self.board[x, j:j + win_length] == player):
                    return True
            # Kiểm tra cột
            start_x = max(0, x - win_length + 1)
            end_x = min(self.size - win_length + 1, x + 1)
            for i in range(start_x, end_x):
                if np.all(self.board[i:i + win_length, y] == player):
                    return True
            # Kiểm tra đường chéo chính
            offset = min(x, y, win_length - 1)
            start_x, start_y = x - offset, y - offset
            if start_x + win_length <= self.size and start_y + win_length <= self.size:
                if np.all(np.diag(self.board[start_x:start_x + win_length, start_y:start_y + win_length]) == player):
                    return True
            # Kiểm tra đường chéo phụ
            offset = min(x, self.size - 1 - y, win_length - 1)
            start_x, start_y = x - offset, y + offset
            if start_x + win_length <= self.size and start_y - win_length + 1 >= 0:
                if np.all(np.diag(np.fliplr(self.board[start_x:start_x + win_length, start_y - win_length + 1:start_y + 1])) == player):
                    return True
            return False

        # Logic gốc nếu không có last_move
        for i in range(self.size):
            for j in range(self.size - win_length + 1):
                if np.all(self.board[i, j:j + win_length] == player): return True
                if np.all(self.board[j:j + win_length, i] == player): return True
        for offset in range(-self.size + win_length, self.size - win_length + 1):
            diag = np.diagonal(self.board, offset)
            if len(diag) >= win_length:
                for k in range(len(diag) - win_length + 1):
                    if np.all(diag[k:k + win_length] == player):
                        return True
            diag_flipped = np.diagonal(np.fliplr(self.board), offset)
            if len(diag_flipped) >= win_length:
                for k in range(len(diag_flipped) - win_length + 1):
                    if np.all(diag_flipped[k:k + win_length] == player):
                        return True
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
        if 0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == 0:
            self.board[x, y] = player
            return True
        return False

    def print_board(self):
        """
        In bàn cờ với số hàng/cột và ký hiệu đơn giản: X (người chơi), O (AI), . (trống).
        - Hiệu suất: O(size^2) để duyệt và in.
        """
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("\n  ", end=" ")
        # In số cột
        for j in range(self.size):
            print(f" {j:^3}", end="  ")
        print()
        print("  " + "-" * (self.size * 7 - (self.size - 1)))
        # In từng hàng với số hàng
        for i in range(self.size):
            row = [f"{symbols[self.board[i, j]]:^3}" for j in range(self.size)]
            print(f"{i:<2}| {' | '.join(row)} |")
            if i < self.size - 1:
                print("  " + "-" * (self.size * 7 - (self.size - 1)))
        print("  " + "-" * (self.size * 7 - (self.size - 1)) + "\n")

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
        if self.size != 3 and available_moves > 20:
            max_depth = 2
        time_limit = 0.2 if self.size == 3 else 0.5 if self.size == 5 else 0.8
        move = self.iddfs_move(player, max_depth=max_depth, time_limit=time_limit)
        if move is None:
            return random.choice(self.get_available_moves())
        return move

    def iddfs_move(self, player, max_depth, time_limit):
        """
        Tìm nước đi tốt nhất bằng IDDFS với sắp xếp nước đi tối ưu.
        - Hiệu suất: O(b^d), giảm nhờ tính score_move một lần.
        """
        start_time = time.time()
        best_move = None
        best_score = -np.inf
        moves = self.get_available_moves()
        # Tính điểm số một lần và sắp xếp
        scored_moves = [(self.score_move(i, j, player), (i, j)) for i, j in moves]
        scored_moves.sort(reverse=True)
        sorted_moves = [move for _, move in scored_moves]

        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_limit:
                break
            temp_best_move = None
            temp_best_score = -np.inf
            for i, j in sorted_moves:
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
        DFS với độ sâu giới hạn và alpha-beta pruning để tìm điểm số tối ưu.
        - Logic:
          - Nếu hết độ sâu hoặc ván kết thúc, trả về điểm từ evaluate_board.
          - Nếu is_maximizing (AI), chọn nước đi tối đa điểm.
          - Nếu không (người chơi), chọn nước đi tối thiểu điểm.
          - Sắp xếp nước đi theo score_move để cắt tỉa hiệu quả.
        - Hiệu suất: O(b^d) nhưng giảm nhờ alpha-beta.
        - Trả về: Điểm số đánh giá bàn cờ (số thực, cao hơn là tốt cho AI).
        - Liên quan: Hỗ trợ iddfs_move trong chế độ easy.
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
        - Chuỗi nguy hiểm: Chuỗi liên tiếp của đối thủ dài win_length-1 với ít nhất một ô trống để mở rộng.
        - Logic: Kiểm tra 8 hướng, đếm số quân liên tiếp của đối thủ.
        - Trả về: Tuple (x, y) của vị trí chặn hợp lệ hoặc None nếu không có.
        - Hiệu suất: O(1) vì chỉ kiểm tra tối đa 4 ô mỗi hướng.
        - Liên quan: Dùng trong medium (medium_move) và hard (hard_move).
        """
        win_length = 3 if self.size == 5 else 4
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        potential_blocks = []
    
        # Kiểm tra xem ô (x, y) có trống không
        if self.board[x, y] != 0:
            return None
        self.board[x, y] = -player  # Mô phỏng nước đi chặn
    
        # Kiểm tra các hướng
        for dx, dy in directions:
            count = 0
            block_pos = None
            for i in range(-win_length + 1, win_length):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx, ny] == player:
                        count += 1
                    elif self.board[nx, ny] == 0:
                        block_pos = (nx, ny)
                    else:
                        continue
                else:
                    break
                if count >= win_length - 1 and block_pos:
                    potential_blocks.append(block_pos)
    
        self.board[x, y] = 0  # Hoàn tác nước đi
    
        # Nếu (x, y) chặn được chuỗi nguy hiểm, trả về nó
        if potential_blocks:
            return (x, y)
        return None

    def evaluate_board(self, last_move=None):
        """
        Đánh giá bàn cờ dựa trên nước đi cuối cùng (nếu có).
        - Hiệu suất: O(win_length) nếu có last_move, O(size^2) nếu không.
        """
        if self.is_winner(-1, last_move):
            return 10
        elif self.is_winner(1, last_move):
            return -10
        score = 0
        target_length = 3 if self.size >= 5 else 2

        if last_move is not None:
            x, y = last_move
            center = self.size // 2
            # Đánh giá chuỗi ngang
            start_y = max(0, y - target_length + 1)
            end_y = min(self.size - target_length + 1, y + 1)
            for j in range(start_y, end_y):
                slice = self.board[x, j:j + target_length]
                if np.count_nonzero(slice == -1) == target_length - 1 and np.count_nonzero(slice == 0) == 1:
                    score += 2
                if np.count_nonzero(slice == 1) == target_length - 1 and np.count_nonzero(slice == 0) == 1:
                    score -= 2
            # Đánh giá chuỗi dọc
            start_x = max(0, x - target_length + 1)
            end_x = min(self.size - target_length + 1, x + 1)
            for i in range(start_x, end_x):
                slice = self.board[i:i + target_length, y]
                if np.count_nonzero(slice == -1) == target_length - 1 and np.count_nonzero(slice == 0) == 1:
                    score += 2
                if np.count_nonzero(slice == 1) == target_length - 1 and np.count_nonzero(slice == 0) == 1:
                    score -= 2
            # Đánh giá vị trí trung tâm
            distance = abs(x - center) + abs(y - center)
            if self.board[x, y] == -1:
                score += 0.1 * (self.size - distance) / self.size
            elif self.board[x, y] == 1:
                score -= 0.1 * (self.size - distance) / self.size
            return score

        # Logic gốc nếu không có last_move
        for i in range(self.size):
            for j in range(self.size - target_length + 1):
                row_slice = self.board[i, j:j + target_length]
                col_slice = self.board[j:j + target_length, i]
                if np.count_nonzero(row_slice == -1) == target_length - 1 and np.count_nonzero(row_slice == 0) == 1:
                    score += 2
                if np.count_nonzero(row_slice == 1) == target_length - 1 and np.count_nonzero(row_slice == 0) == 1:
                    score -= 2
                if np.count_nonzero(col_slice == -1) == target_length - 1 and np.count_nonzero(col_slice == 0) == 1:
                    score += 2
                if np.count_nonzero(col_slice == 1) == target_length - 1 and np.count_nonzero(col_slice == 0) == 1:
                    score -= 2
        for i in range(self.size):
            for j in range(self.size - 1):
                row_slice = self.board[i, j:j + 2]
                col_slice = self.board[j:j + 2, i]
                if np.any(row_slice == [-1, 0]) or np.any(row_slice == [0, -1]):
                    score += 1
                if np.any(row_slice == [1, 0]) or np.any(row_slice == [0, 1]):
                    score -= 1
                if np.any(col_slice == [-1, 0]) or np.any(col_slice == [0, -1]):
                    score += 1
                if np.any(col_slice == [1, 0]) or np.any(col_slice == [0, 1]):
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
        Tìm nước đi tối ưu bằng alpha-beta pruning với sắp xếp nước đi tối ưu.
        - Hiệu suất: O(b^d), giảm nhờ tính score_move một lần.
        """
        if depth == 0 or self.is_winner(1) or self.is_winner(-1) or self.is_draw():
            return self.evaluate_board(), None
        moves = self.get_available_moves()
        # Tính điểm số một lần và sắp xếp
        scored_moves = [(self.score_move(i, j, -1 if is_maximizing_player else 1), (i, j)) for i, j in moves]
        scored_moves.sort(reverse=True)
        sorted_moves = [move for _, move in scored_moves]
        best_move = None
        if is_maximizing_player:
            max_eval = -np.inf
            for i, j in sorted_moves:
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
            for i, j in sorted_moves:
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
        Chế độ khó: Dùng alpha-beta với độ sâu cao cho 3x3, DQN cho 5x5 và 7x7.
        - Mục đích: Tạo AI mạnh hơn medium và easy, phù hợp với kích thước bàn cờ.
        - Dữ liệu:
          - 3x3: Không dùng DQN, chỉ dùng alpha-beta độ sâu 8.
          - 5x5, 7x7: Dùng trọng số từ `dqn_{size}x{size}.pth`.
        - Logic:
          1. Kiểm tra thắng ngay lập tức.
          2. Kiểm tra chặn đối thủ thắng.
          3. Tìm nước đi tạo chuỗi 3 (5x5) hoặc 4 (7x7).
          4. Tìm nước đi chặn chuỗi đối thủ.
          5. 3x3: Dùng alpha_beta_move độ sâu 8.
             5x5, 7x7: Dùng dqn_move với epsilon thấp, kiểm tra lại bằng alpha_beta_move độ sâu 3.
        - Hiệu suất:
          - 3x3: ~0.1-0.5s.
          - 5x5, 7x7: ~0.5-0.7s.
        - Trả về: Tuple (x, y) hoặc None nếu không có nước đi hợp lệ (bàn cờ đầy).
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
        if not self.get_available_moves():
            return None
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
                score, _ = self.alpha_beta_move(depth=3, is_maximizing_player=False)
                self.board[i, j] = 0
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
            return best_move

class TicTacToeAI(TicTacToe):
    """
    Lớp mở rộng TicTacToe với DQN, sử dụng CNN và học tăng cường.
    - Mục đích: Triển khai AI thông minh cho chế độ Hard và hỗ trợ huấn luyện DQN.
    - Hiệu suất: Khởi tạo O(1), huấn luyện phụ thuộc episodes và batch_size (128).
    - Liên quan: Chủ yếu dùng trong Hard (dqn_move, hard_move) và huấn luyện (train_game).
    - Dữ liệu:
        - `dqn_{size}x{size}.pth`: Trọng số mô hình DQN, dùng khi chơi Hard.
        - `memory_{size}x{size}.pkl`: Kinh nghiệm, dùng khi huấn luyện.
        - `elite_memory_{size}x{size}.pkl`: Kinh nghiệm chất lượng cao, dùng khi huấn luyện.
    """
    
    def __init__(self, size=3, mode="hard", evaluate_model=True):
        """
        Khởi tạo AI với kích thước bộ nhớ tùy thuộc vào size.
        """
        super().__init__(size)
        if size == 7:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.policy_net = self.create_model(size).to(self.device)
        self.target_net = self.create_model(size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50000, gamma=0.1)
        self.memory = []
        self.memory_capacity = 50000 if size == 5 else 20000  # 50,000 cho 5x5, 20,000 cho 7x7
        self.priorities = []
        self.elite_memory = []
        self.elite_memory_capacity = 10000 if size == 5 else 5000  # 10,000 cho 5x5, 5,000 cho 7x7
        self.elite_priorities = []
        self.batch_size = 128
        self.gamma = 0.995
        self.per_epsilon = 1e-6
        self.per_alpha = 0.8
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type in ["cuda", "mps"]))
        if mode == "hard":
            model_path = f"dqn_{size}x{size}.pth"
            if os.path.exists(model_path):
                try:
                    self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    print(f"Loaded pre-trained model from {model_path}")
                    if evaluate_model:
                        win_rate = Algorithm.evaluate_model(self, size, num_games=20)
                        print(f"Loaded model win rate: {win_rate:.2f}")
                        if win_rate < 0.5:
                            print("Warning: Loaded model may be weak. Consider retraining.")
                except (RuntimeError, pickle.UnpicklingError) as e:
                    print(f"Error loading model from {model_path}: {e}")
                    print("Using new model instead.")

    def create_model(self, size):
        """
        Tạo mô hình CNN tùy thuộc vào kích thước bàn cờ.
        - Logic:
          - 5x5: 4 lớp Conv2d (32, 64, 128, 256), tương thích với mô hình cũ.
          - 3x3, 7x7: 3 lớp Conv2d (32, 64, 128), mô hình mới.
          - BatchNorm2d và ReLU sau mỗi lớp conv.
          - 2 lớp fully connected: (128 hoặc 256)*size*size -> 512 -> size*size.
          - Dropout (0.2) trước lớp cuối.
        - Hiệu suất: O(size^2) cho forward pass.
        - Liên quan: Dùng trong hard (dqn_move) và huấn luyện.
        """
        class DQN(nn.Module):
            def __init__(self, size):
                super(DQN, self).__init__()
                if size == 5:  # Mô hình cũ cho 5x5
                    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                    self.bn1 = nn.BatchNorm2d(32)
                    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.bn2 = nn.BatchNorm2d(64)
                    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    self.bn3 = nn.BatchNorm2d(128)
                    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                    self.bn4 = nn.BatchNorm2d(256)
                    self.fc1 = nn.Linear(256 * size * size, 512)
                else:  # Mô hình mới cho 3x3 và 7x7
                    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                    self.bn1 = nn.BatchNorm2d(32)
                    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.bn2 = nn.BatchNorm2d(64)
                    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    self.bn3 = nn.BatchNorm2d(128)
                    self.fc1 = nn.Linear(128 * size * size, 512)
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(512, size * size)

            def forward(self, x):
                x = x.view(-1, 1, size, size)
                x = torch.relu(self.bn1(self.conv1(x)))
                x = torch.relu(self.bn2(self.conv2(x)))
                x = torch.relu(self.bn3(self.conv3(x)))
                if size == 5:  # Thêm tầng conv4 cho 5x5
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
          - Nếu full_augmentation: Tạo 6 phiên bản (xoay 90/180/270, lật ngang/dọc) với action tương ứng.
          - Nếu không: Tạo 2 phiên bản (xoay 90, lật ngang) với action tương ứng.
        - Hiệu suất: O(size^2) để xoay/lật ma trận.
        - Liên quan: Dùng trong huấn luyện (store_experience).
        """
        states = [state]
        actions = [None]
        size = self.size
        if full_augmentation:
            states.extend([
                np.rot90(state, k=1),
                np.rot90(state, k=2),
                np.rot90(state, k=3),
                np.fliplr(state),
                np.flipud(state)
            ])
            actions.extend([
                lambda x, y: (y, size - 1 - x),  # Xoay 90
                lambda x, y: (size - 1 - x, size - 1 - y),  # Xoay 180
                lambda x, y: (size - 1 - y, x),  # Xoay 270
                lambda x, y: (x, size - 1 - y),  # Lật ngang
                lambda x, y: (size - 1 - x, y)  # Lật dọc
            ])
        else:
            states.extend([
                np.rot90(state, k=1),
                np.fliplr(state)
            ])
            actions.extend([
                lambda x, y: (y, size - 1 - x),  # Xoay 90
                lambda x, y: (x, size - 1 - y)  # Lật ngang
            ])
        return list(zip(states, actions))
                    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Lưu kinh nghiệm (state, action, reward, next_state, done) vào memory hoặc elite_memory.
        """
        if not (state.shape == (self.size, self.size) and next_state.shape == (self.size, self.size)):
            print(f"Warning: Invalid state shape, skipping experience.")
            return
        if action < 0 or action >= self.size * self.size:
            print(f"Warning: Invalid action {action}, skipping experience.")
            return

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
        x, y = divmod(action, self.size)
        augmented = self.augment_state(state, full_augmentation)
        next_augmented = self.augment_state(next_state, full_augmentation)

        for (sym_state, action_transform), (sym_next_state, _) in zip(augmented, next_augmented):
            if action_transform:
                new_x, new_y = action_transform(x, y)
                new_action = new_x * self.size + new_y
                if new_action < 0 or new_action >= self.size * self.size:
                    print(f"Warning: Invalid transformed action {new_action}, skipping.")
                    continue
            else:
                new_action = action
            experience = (sym_state, new_action, reward, sym_next_state, done)
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

        available = self.get_available_moves()
        if not available:
            return None

        if random.random() < epsilon:
            return random.choice(available)

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state)
        self.policy_net.train()
        flat_available = [i * self.size + j for i, j in available]
        q_values_np = q_values.cpu().numpy().flatten()
        best_flat = max(flat_available, key=lambda idx: q_values_np[idx])
        return divmod(best_flat, self.size)

    def replay(self):
        """
        Huấn luyện mô hình với batch từ memory và elite_memory, đảm bảo tính ổn định.
        - Logic: Lấy mẫu ưu tiên từ memory và elite_memory, tính loss MSE, cập nhật policy_net.
        - Sử dụng prioritized experience replay với per_epsilon (tránh ưu tiên 0) và per_alpha (điều chỉnh trọng số mẫu).
        - Trường hợp đặc biệt: Bỏ qua nếu batch rỗng hoặc gặp lỗi tensor, in cảnh báo.
        - Hiệu suất: O(batch_size * size^2) cho forward/backward pass.
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
            print("Warning: Empty batch in replay, skipping.")
            return

        try:
            state, action, reward, next_state, done = zip(*batch)
            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
            action = torch.tensor(action, dtype=torch.long, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
            done = torch.tensor(done, dtype=torch.float32, device=self.device)

            # Kiểm tra tính hợp lệ của action
            if any(a >= self.size * self.size or a < 0 for a in action):
                print(f"Warning: Invalid action indices detected: {action.tolist()}, skipping batch.")
                return

            self.policy_net.eval()
            self.target_net.eval()
            with torch.no_grad():
                next_q_values = self.policy_net(next_state)
                next_q_target_values = self.target_net(next_state)
            self.policy_net.train()
            self.target_net.train()

            q_values = self.policy_net(state)
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
            if self.device.type in ["cuda", "mps"]:
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    scaled_loss = self.scaler.scale(loss)
                    scaled_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                    if grad_norm > 1.0:
                        print(f"Warning: Large gradient norm detected: {grad_norm:.2f}")
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                if grad_norm > 1.0:
                    print(f"Warning: Large gradient norm detected: {grad_norm:.2f}")
                self.optimizer.step()
                self.scaler.update()

        except Exception as e:
            print(f"Error processing batch in replay: {e}")
            return

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
            pickle.dump(self.memory, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.priorities, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_elite_memory(self, filename):
        """
        Lưu bộ nhớ chất lượng cao vào file `.pkl`.
        - Logic: Lưu elite_memory và elite_priorities.
        - Hiệu suất: O(elite_memory_capacity) để ghi file.
        - Liên quan: Dùng trong huấn luyện (train_game).
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.elite_memory, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.elite_priorities, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_memory(self, filename):
        """
        Tải bộ nhớ kinh nghiệm từ file `.pkl`.
        - Logic: Tải memory và priorities, xử lý lỗi nếu file không tồn tại hoặc hỏng.
        - Hiệu suất: O(memory_capacity) để đọc file.
        - Liên quan: Dùng trong huấn luyện (train_game).
        """
        try:
            with open(filename, 'rb') as f:
                self.memory = pickle.load(f)
                self.priorities = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
            print(f"Failed to load memory from {filename}: {e}. Starting with empty memory.")
            self.memory = []
            self.priorities = []

    def load_elite_memory(self, filename):
        """
        Tải bộ nhớ chất lượng cao từ file `.pkl`.
        - Logic: Tải elite_memory và elite_priorities, xử lý lỗi nếu file không tồn tại hoặc hỏng.
        - Hiệu suất: O(elite_memory_capacity) để đọc file.
        - Liên quan: Dùng trong huấn luyện (train_game).
        """
        try:
            with open(filename, 'rb') as f:
                self.elite_memory = pickle.load(f)
                self.elite_priorities = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
            print(f"Failed to load elite memory from {filename}: {e}. Starting with empty elite memory.")
            self.elite_memory = []
            self.elite_priorities = []

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
    def evaluate_model(ai, size, num_games=20):
        """
        Đánh giá mô hình DQN bằng cách đấu với đối thủ trung bình.
        - Logic:
          - Chơi num_games ván, AI (DQN) đấu với medium_move.
          - Tính tỷ lệ thắng của AI.
        - Hiệu suất: O(num_games * size^2) cho mỗi ván.
        - Liên quan: Dùng trong huấn luyện (train_game) và khởi tạo (TicTacToeAI.__init__) để kiểm tra chất lượng mô hình.
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
                    move = ai.dqn_move(episode=500000)  # Epsilon thấp (~0.05) để đánh giá
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
        """
        Chơi một trận với người chơi, nhập tọa độ hàng/cột (row, col).
        """
        game = TicTacToeAI(size=size, mode=mode)
        current_player = 1
        print(f"\nStarting game on {size}x{size} board in {mode} mode...")
        print(f"Enter your move as 'row col' (e.g., '1 2' for row 1, column 2).")
        print(f"Rows and columns are numbered from 0 to {size-1}. Type 'quit' to exit.")
        game.print_board()

        max_attempts = 5
        while True:
            if current_player == 1:
                attempts = 0
                while attempts < max_attempts:
                    try:
                        move_input = input("Your move (row col, or 'quit' to exit): ").strip()
                        if move_input.lower() == 'quit':
                            print("Game exited.")
                            return
                        row, col = map(int, move_input.split())
                        if not (0 <= row < size and 0 <= col < size):
                            print(f"Invalid coordinates! Row and column must be between 0 and {size-1}.")
                            attempts += 1
                            continue
                        if not game.make_move(row, col, 1):
                            print("Invalid move! That cell is already occupied.")
                            attempts += 1
                            continue
                        break
                    except ValueError:
                        print("Invalid input! Enter two numbers separated by a space (e.g., '1 2') or 'quit'.")
                        attempts += 1
                        continue
                if attempts >= max_attempts:
                    print("Too many invalid attempts. Game over.")
                    return
            else:
                if mode == "easy":
                    move = game.easy_move(-1)
                elif mode == "medium":
                    move = game.medium_move(-1)
                elif mode == "hard":
                    move = game.hard_move(-1)
                game.make_move(move[0], move[1], -1)
                print(f"AI moves at: {move[0]} {move[1]}")

            game.print_board()

            # Kiểm tra kết thúc trò chơi
            last_move = (row, col) if current_player == 1 else move
            if game.is_winner(current_player, last_move):
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
        ai1 = TicTacToeAI(size=size, mode="hard", evaluate_model=False)
        ai2 = TicTacToeAI(size=size, mode="hard", evaluate_model=False)

        if load_model_path:
            try:
                ai1.policy_net.load_state_dict(torch.load(load_model_path, map_location=ai1.device))
                ai1.target_net.load_state_dict(ai1.policy_net.state_dict())
                ai2.policy_net.load_state_dict(torch.load(load_model_path, map_location=ai2.device))
                ai2.target_net.load_state_dict(ai2.policy_net.state_dict())
                print(f"Loaded pre-trained model from {load_model_path}")
            except (FileNotFoundError, RuntimeError, pickle.UnpicklingError) as e:
                print(f"Failed to load model from {load_model_path}: {e}. Starting with new model.")

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

    @staticmethod
    def compare_modes(size, num_games):
        """
        Mô phỏng các trận đấu giữa Medium vs Easy và Medium vs Hard, đảm bảo công bằng.
        - Logic:
          - Chạy num_games trận, chia đều: num_games//2 trận Medium (X) vs Easy/Hard (O), và ngược lại.
          - Luân phiên vai trò X/O để tránh thiên vị do thứ tự đi.
          - Ghi lại thắng, thua, hòa từ góc nhìn của Medium.
          - In bàn cờ khi Hard thua để hỗ trợ debug chiến lược.
        - Hiệu suất: O(num_games * size^2) cho mỗi nhóm trận.
        - Liên quan: Dùng trong chế độ Compare để so sánh hiệu suất.
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
    @staticmethod
    def main():
        """
        Hàm chính: Hiển thị menu và xử lý lựa chọn người dùng, hỗ trợ chế độ Compare.
        - Logic:
          - Mode 1: Chơi với AI (easy, medium, hard).
          - Mode 2: Huấn luyện DQN (5x5, 7x7), 5 giai đoạn, mỗi giai đoạn 100,000 episodes.
          - Mode 3: So sánh Medium vs Easy và Medium vs Hard.
          - Xử lý lỗi đầu vào và Ctrl+C.
        - Lưu trữ: Ghi đè `dqn_{size}x{size}.pth`, `memory_{size}x{size}.pkl`, `elite_memory_{size}x{size}.pkl` khi huấn luyện cải thiện.
        - Hiệu suất: Phụ thuộc vào play_game, train_game, hoặc compare_modes.
        - Liên quan: Gọi play_game, train_game, hoặc compare_modes.
        """
        print("Welcome to Tic-Tac-Toe!")
        print("1: Play against AI")
        print("2: Train AI model")
        print("3: Compare AI modes")
        try:
            mode_choice = input("Choose mode (1: Play, 2: Training, 3: Compare): ").strip()
            if mode_choice not in ["1", "2", "3"]:
                print("Invalid choice! Choose 1, 2, or 3.")
                return

            if mode_choice == "1":
                size = input("Choose board size (3, 5, 7): ").strip()
                try:
                    size = int(size)
                    if size not in [3, 5, 7]:
                        print("Invalid board size! Choose 3, 5, or 7.")
                        return
                except ValueError:
                    print("Invalid input! Please enter a number (3, 5, or 7).")
                    return
                difficulty = input("Choose difficulty (1: Easy, 2: Medium, 3: Hard): ").strip()
                mode = {"1": "easy", "2": "medium", "3": "hard"}.get(difficulty)
                if mode:
                    Algorithm.play_game(size, mode)
                else:
                    print("Invalid difficulty! Choose 1, 2, or 3.")
            elif mode_choice == "2":
                size = input("Choose board size (5, 7): ").strip()
                try:
                    size = int(size)
                    if size not in [5, 7]:
                        print("Training mode is only available for 5x5 and 7x7 boards.")
                        return
                except ValueError:
                    print("Invalid input! Please enter a number (5 or 7).")
                    return
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
            elif mode_choice == "3":
                size = input("Choose board size (3, 5, 7): ").strip()
                try:
                    size = int(size)
                    if size not in [3, 5, 7]:
                        print("Invalid board size! Choose 3, 5, or 7.")
                        return
                except ValueError:
                    print("Invalid input! Please enter a number (3, 5, or 7).")
                    return
                num_games = input("Enter number of games to simulate: ").strip()
                try:
                    num_games = int(num_games)
                    if num_games <= 0:
                        print("Number of games must be positive!")
                        return
                except ValueError:
                    print("Invalid input! Please enter a positive number.")
                    return
                Algorithm.compare_modes(size, num_games)
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
            sys.exit(0)

if __name__ == "__main__":
    Main.main()