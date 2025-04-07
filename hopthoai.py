from tkinter import messagebox, simpledialog
import random
import tkinter as tk
import os

# Đường dẫn file lưu trữ dữ liệu
HISTORY_FILE = "history.txt"
LEADERBOARD_FILE = "leaderboard.txt"

# Khởi tạo giá trị mặc định
selected_difficulty = "Dễ"
selected_board_size = "3x3"
first_player = "Người chơi 1"

def save_to_file(filename, data):
    """Ghi dữ liệu vào file"""
    with open(filename, "w") as file:
        for item in data:
            file.write(" ".join(map(str, item)) + "\n")

def load_from_file(filename):
    """Đọc dữ liệu từ file, nếu không có thì trả về danh sách trống"""
    if not os.path.exists(filename):
        return []
    with open(filename, "r") as file:
        return [[parts[0], int(parts[1])] for parts in (line.strip().split(" ") for line in file) if len(parts) == 2]

def update_leaderboard(winner):
    """Cập nhật bảng xếp hạng khi có người thắng"""
    leaderboard = load_from_file(LEADERBOARD_FILE)
    leaderboard_dict = {player: score for player, score in leaderboard} if leaderboard else {}

    leaderboard_dict[winner] = leaderboard_dict.get(winner, 0) + 1

    sorted_leaderboard = sorted(leaderboard_dict.items(), key=lambda x: x[1], reverse=True)
    save_to_file(LEADERBOARD_FILE, sorted_leaderboard)

def save_match_history(player1, player2, winner):
    """Lưu lịch sử trận đấu"""
    history = load_from_file(HISTORY_FILE)
    history.append([player1, player2, winner])
    save_to_file(HISTORY_FILE, history)

def show_leaderboard():
    """Hiển thị bảng xếp hạng"""
    leaderboard = load_from_file(LEADERBOARD_FILE)
    
    leaderboard_window = tk.Toplevel()
    leaderboard_window.title("Bảng Xếp Hạng")
    leaderboard_window.geometry("300x300")

    tk.Label(leaderboard_window, text="Bảng Xếp Hạng", font=("Arial", 14, "bold")).pack(pady=10)

    if not leaderboard:
        tk.Label(leaderboard_window, text="Chưa có dữ liệu.", font=("Arial", 12)).pack(pady=10)
    else:
        for rank, (player, score) in enumerate(leaderboard, start=1):
            tk.Label(leaderboard_window, text=f"{rank}. {player} - {score} điểm", font=("Arial", 12)).pack(pady=5)

    tk.Button(leaderboard_window, text="Đóng", font=("Arial", 12), command=leaderboard_window.destroy).pack(pady=10)

def show_history():
    """Hiển thị lịch sử trận đấu"""
    history = load_from_file(HISTORY_FILE)

    history_str = "\n".join(f"{p1} vs {p2} - Thắng: {winner}" for p1, p2, winner in history) if history else "Chưa có trận đấu nào."

    messagebox.showinfo("Lịch sử đấu", history_str)

def show_result(winner):
    """Hiển thị thông báo kết quả trận đấu"""
    messagebox.showinfo("Kết quả", f"Người thắng cuộc: {winner}")

def toss_coin():
    """Tung đồng xu để quyết định ai đi trước"""
    global first_player
    first_player = random.choice(["Người chơi 1", "Người chơi 2"])
    messagebox.showinfo("Tung đồng xu", f"{first_player} sẽ đi trước!")
    return first_player

def get_first_player():
    """Lấy thông tin người đi trước"""
    return first_player

def choose_server_mode():
    """Giao diện chọn chế độ chơi: Local hoặc Server"""
    root = tk.Tk()
    root.title("Chế độ chơi")
    root.geometry("300x200")

    tk.Label(root, text="Chọn chế độ chơi", font=("Arial", 14)).pack(pady=10)

    def select_mode(mode):
        root.destroy()
        if mode == "Local":
            choose_difficulty()
        else:
            messagebox.showinfo("Phòng chờ", "Đang chuyển đến màn hình phòng chờ...")
            # Thêm xử lý cho chế độ Server ở đây

    tk.Button(root, text="Chơi Local", font=("Arial", 12), command=lambda: select_mode("Local")).pack(pady=5)


def choose_board_size(difficulty):
    """Mở màn hình chọn bàn cờ"""
    global selected_difficulty, selected_board_size
    selected_difficulty = difficulty

    board_window = tk.Toplevel()
    board_window.title("Chọn bàn cờ")
    
    tk.Label(board_window, text=f"Chọn kích thước bàn cờ ({difficulty}):", 
            font=("Arial", 12)).pack(pady=10)

    def select_size(size):
        global selected_board_size
        selected_board_size = size
        board_window.destroy()
        messagebox.showinfo("Thông báo", f"Đã chọn bàn cờ {size} ({difficulty})")

    tk.Button(board_window, text="3x3", command=lambda: select_size("3x3")).pack(pady=5)
    tk.Button(board_window, text="5x5", command=lambda: select_size("5x5")).pack(pady=5)
    tk.Button(board_window, text="7x7", command=lambda: select_size("7x7")).pack(pady=5)
    
def choose_difficulty():
    """Mở màn hình chọn độ khó"""
    root = tk.Tk()
    root.title("Chọn độ khó")
    root.geometry("300x200")

    tk.Label(root, text="Chọn độ khó:", font=("Arial", 14)).pack(pady=10)

    tk.Button(root, text="Khó", font=("Arial", 12), command=lambda: [choose_board_size("Khó"), root.destroy()]).pack(pady=5)
    tk.Button(root, text="Trung Bình", font=("Arial", 12), command=lambda: [choose_board_size("Trung Bình"), root.destroy()]).pack(pady=5)
    tk.Button(root, text="Dễ", font=("Arial", 12), command=lambda: [choose_board_size("Dễ"), root.destroy()]).pack(pady=5)
    root.mainloop()

def get_selected_options():
    """Trả về lựa chọn độ khó và kích thước bàn cờ"""
    return selected_difficulty, selected_board_size