import tkinter as tk
from tkinter import messagebox
import os
import random
import subprocess

class TicTacToe:
    def back_to_menu(self):
        self.root.destroy()
        subprocess.run(["python3", "manhinh2.py"])

    def __init__(self):
        try:
            with open("game_settings.txt", "r") as f:
                settings = f.read().splitlines()
                self.difficulty = settings[0]
                self.board_size = settings[1]
                self.game_mode = settings[2] if len(settings) > 2 else "AI"
        except Exception as e:
            print(f"Lỗi đọc file: {e}")
            self.difficulty = "Dễ"
            self.board_size = "3x3"
            self.game_mode = "AI"

        self.size = int(self.board_size[0])
        self.current_player = "X"
        self.board = [["" for _ in range(self.size)] for _ in range(self.size)]
        self.game_over = False

        self.root = tk.Tk()
        self.root.attributes('-modified', True)
        self.root.title(f"Tic-Tac-Toe - {self.board_size} | {self.difficulty} | {self.game_mode}")
        self.root.configure(bg="#ffe6f0")

        self.cell_size = 120
        self.canvas_size = self.cell_size * self.size

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size,
                                bg="#fff0f5", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.draw_grid()

        self.info_frame = tk.Frame(self.root, bg="#ffe6f0")
        self.info_frame.pack(pady=5)

        self.turn_label = tk.Label(
            self.info_frame,
            text=f"Lượt: {self.current_player}",
            font=("Arial", 40, "bold"),
            fg="#ff4d88",
            bg="#ffe6f0"
        )
        self.turn_label.pack(side=tk.LEFT, padx=10)

        self.control_frame = tk.Frame(self.root, bg="#ffe6f0")
        self.control_frame.pack(pady=10)

        tk.Button(
            self.control_frame,
            text="🔁 Chơi lại",
            font=("Arial", 30),
            width=20,
            bg="#ffb3d9",
            command=self.reset_game
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            self.control_frame,
            text="⏪ Thoát",
            font=("Arial", 30),
            width=20,
            bg="#ffb3d9",
            command=self.back_to_menu
        ).pack(side=tk.LEFT, padx=10)

        self.root.mainloop()

    def draw_grid(self):
        for i in range(self.size + 1):
            self.canvas.create_line(0, i * self.cell_size, self.canvas_size, i * self.cell_size, fill="#999", width=1)
            self.canvas.create_line(i * self.cell_size, 0, i * self.cell_size, self.canvas_size, fill="#999", width=1)

    def on_canvas_click(self, event):
        if self.game_over:
            return

        row = event.y // self.cell_size
        col = event.x // self.cell_size

        if row < self.size and col < self.size:
            self.make_move(row, col)

    def draw_symbol(self, row, col, player):
        x0 = col * self.cell_size + self.cell_size // 2
        y0 = row * self.cell_size + self.cell_size // 2
        color = "red" if player == "X" else "purple"
        symbol = "X" if player == "X" else "♡"
        self.canvas.create_text(x0, y0, text=symbol, font=("Arial", 48, "bold"), fill=color)

    def make_move(self, row, col):
        if self.game_over or self.board[row][col] != "":
            return

        self.board[row][col] = self.current_player
        self.draw_symbol(row, col, self.current_player)

        if self.check_winner(self.current_player):
            self.end_game(f"Người chơi {self.current_player} thắng!")
            return

        if self.check_draw():
            self.end_game("Hòa!")
            return

        self.switch_player()

        if self.game_mode == "AI" and self.current_player == "♡":
            self.root.after(500, self.ai_move)

    def ai_move(self):
        if self.game_over:
            return

        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == ""]

        if not empty_cells:
            return

        if self.difficulty == "Dễ":
            row, col = random.choice(empty_cells)
        elif self.difficulty == "Trung Bình":
            row, col = self.smart_ai_move(level=1)
        else:
            row, col = self.smart_ai_move(level=2)

        self.make_move(row, col)

    def smart_ai_move(self, level):
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == ""]

        for r, c in empty_cells:
            self.board[r][c] = "♡"
            if self.check_winner("♡"):
                self.board[r][c] = ""
                return r, c
            self.board[r][c] = ""

        for r, c in empty_cells:
            self.board[r][c] = "X"
            if self.check_winner("X"):
                self.board[r][c] = ""
                return r, c
            self.board[r][c] = ""

        if level >= 2 and self.size == 3:
            center = (1, 1)
            if center in empty_cells:
                return center
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            available_corners = [c for c in corners if c in empty_cells]
            if available_corners:
                return random.choice(available_corners)

        return random.choice(empty_cells)

    def check_winner(self, player):
        for i in range(self.size):
            if all(self.board[i][j] == player for j in range(self.size)) or \
               all(self.board[j][i] == player for j in range(self.size)):
                return True

        if all(self.board[i][i] == player for i in range(self.size)) or \
           all(self.board[i][self.size - 1 - i] == player for i in range(self.size)):
            return True

        return False

    def check_draw(self):
        return all(self.board[i][j] != "" for i in range(self.size) for j in range(self.size))

    def switch_player(self):
        self.current_player = "♡" if self.current_player == "X" else "X"
        self.turn_label.config(text=f"Lượt: {self.current_player}")

    def end_game(self, message):
        self.game_over = True
        messagebox.showinfo("Kết thúc", message)
        self.highlight_winning_cells()

    def highlight_winning_cells(self):
        pass

    def reset_game(self):
        self.current_player = "X"
        self.game_over = False
        self.board = [["" for _ in range(self.size)] for _ in range(self.size)]
        self.canvas.delete("all")
        self.draw_grid()
        self.turn_label.config(text=f"Lượt: {self.current_player}")


def display_board():
    TicTacToe()


if __name__ == "__main__":
    display_board()
