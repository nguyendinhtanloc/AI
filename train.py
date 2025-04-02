import numpy as np
import random

class TicTacToe:
    def __init__(self, size=3):
        self.size = size  
        self.board = np.zeros((size, size), dtype=int)  

    def is_winner(self, player):
        """Kiểm tra xem người chơi có chiến thắng không"""
        win_length = 3 if self.size == 3 else 4 if self.size == 5 else 5  

        # Kiểm tra hàng ngang & cột dọc
        for i in range(self.size):
            for j in range(self.size - win_length + 1):
                if np.all(self.board[i, j:j + win_length] == player): return True
                if np.all(self.board[j:j + win_length, i] == player): return True

        # Kiểm tra chéo chính & chéo phụ
        for i in range(self.size - win_length + 1):
            for j in range(self.size - win_length + 1):
                if np.all(np.diag(self.board[i:i + win_length, j:j + win_length]) == player): return True
                if np.all(np.diag(np.fliplr(self.board[i:i + win_length, j:j + win_length])) == player): return True

        return False

    def is_draw(self):
        """Kiểm tra hòa"""
        return not np.any(self.board == 0) and not self.is_winner(1) and not self.is_winner(-1)

    def get_available_moves(self):
        """Lấy nước đi hợp lệ"""
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]

    def make_move(self, x, y, player):
        """Đánh dấu nước đi"""
        if self.board[x, y] == 0:
            self.board[x, y] = player
            return True
        return False

    def print_board(self):
        """In bàn cờ"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()

    def easy_move(self):
        """Chế độ dễ: nước đi ngẫu nhiên"""
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

        # Chặn nếu đối thủ có 2 quân liên tiếp (5x5) hoặc 3 quân liên tiếp (7x7)
        for i, j in self.get_available_moves():
            if self.can_block(i, j, opponent):
                print(f"Blocking move at {i},{j} to stop opponent from winning!")
                return (i, j)

        # Chọn nước đi có lợi nhất bằng heuristic đơn giản
        best_score = -np.inf
        for i, j in self.get_available_moves():
            self.board[i, j] = player
            score = self.evaluate_board()
            self.board[i, j] = 0
            if score > best_score:
                best_score = score
                best_move = (i, j)

        return best_move if best_move else self.easy_move()

    def can_block(self, x, y, player):
        """Kiểm tra xem nếu AI (hoặc đối thủ) có thể tạo thành một chuỗi quân đủ dài (2 quân cho 5x5, 3 quân cho 7x7)"""
        # Điều kiện số quân liên tiếp cần chặn
        if self.size == 5:
            win_length = 2  # Đối với bàn cờ 5x5, đối thủ có thể thắng với 2 quân liên tiếp
        elif self.size == 7:
            win_length = 3  # Đối với bàn cờ 7x7, đối thủ có thể thắng với 3 quân liên tiếp
        else:
            return None  # Chế độ không có hiệu lực đối với các bàn cờ khác

        # Các hướng kiểm tra (hàng, cột, chéo)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

        for dx, dy in directions:
            count = 0  # Đếm quân của đối thủ
            potential_block = None  # Vị trí cần chặn

            # Kiểm tra phía trước
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

            # Kiểm tra phía sau
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

            # Nếu số quân liên tiếp đủ để chiến thắng và có vị trí cần chặn
            if count == win_length and potential_block:
                return potential_block

        return None  # Không tìm thấy vị trí cần chặn

    def hard_move(self):
        """Chế độ khó: AI chọn nước đi tối ưu"""
        return self.best_move()  # Minimax đầy đủ

class TicTacToeAI(TicTacToe):
    def __init__(self, size=3):
        super().__init__(size)
        self.memo = {}  # Bộ nhớ đệm để lưu kết quả đã tính toán

    def best_move(self, max_depth=None):
        """Tìm nước đi tốt nhất bằng thuật toán Minimax có cắt tỉa Alpha-Beta"""
        best_score = -np.inf
        move = None

        for i, j in self.get_available_moves():
            self.board[i, j] = 1
            score = self.alpha_beta(0, -np.inf, np.inf, False, max_depth)
            self.board[i, j] = 0

            if score > best_score:
                best_score = score
                move = (i, j)

        return move

    def alpha_beta(self, depth, alpha, beta, is_maximizing, max_depth):
        """Thuật toán Minimax với cắt tỉa Alpha-Beta và bộ nhớ đệm"""
        board_tuple = tuple(map(tuple, self.board))
        if board_tuple in self.memo:
            return self.memo[board_tuple]

        if self.is_winner(1): return 1
        if self.is_winner(-1): return -1
        if self.is_draw(): return 0

        if max_depth is not None and depth == max_depth:
            score = self.evaluate_board()
            self.memo[board_tuple] = score
            return score

        best_score = -np.inf if is_maximizing else np.inf
        player = 1 if is_maximizing else -1

        for i, j in self.get_available_moves():
            self.board[i, j] = player
            score = self.alpha_beta(depth + 1, alpha, beta, not is_maximizing, max_depth)
            self.board[i, j] = 0

            if is_maximizing:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)

            if beta <= alpha:
                break  

        self.memo[board_tuple] = best_score
        return best_score

    def evaluate_board(self):
        """Hàm đánh giá bàn cờ dựa trên số quân cờ"""
        score = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 1:
                    score += 10  # Thưởng cho AI
                elif self.board[i, j] == -1:
                    score -= 10  # Phạt cho đối thủ
        return score

class Algorithm:
    @staticmethod
    def play_game(size, mode):
        game = TicTacToeAI(size=size)

        while True:
            game.print_board()

            try:
                x, y = map(int, input("Enter your move (x y): ").split())
                if not (0 <= x < size and 0 <= y < size) or not game.make_move(x, y, -1):
                    print("Invalid move. Try again!")
                    continue
            except ValueError:
                print("Invalid input. Enter two numbers separated by space.")
                continue

            if game.is_winner(-1):
                game.print_board()
                print("You win!")
                break

            if game.is_draw():
                game.print_board()
                print("It's a draw!")
                break

            if mode == "medium":
                move = game.medium_move(1)
            elif mode == "hard":
                move = game.hard_move()
            else:
                move = game.easy_move()

            game.make_move(*move, 1)
            print(f"AI chose move: {move}")

            if game.is_winner(1):
                game.print_board()
                print("AI wins!")
                break
            if game.is_draw():
                game.print_board()
                print("It's a draw!")
                break

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
            print("Training mode not available.")
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    Main.main()
