import tkinter as tk
from tkinter import messagebox
from hopthoai import (
    choose_difficulty, 
    get_selected_options,
    choose_server_mode,
    choose_board_size,
    toss_coin,
    show_leaderboard, 
    show_history,
    get_first_player
)
import time
import subprocess
import os

# Kiểm tra có phải khách không
try:
    with open("guest.txt", "r") as f:
        guest_name = f.read()
    is_guest = True
    # Xóa file guest.txt sau khi đọc
    import os
    os.remove("guest.txt")
except:
    is_guest = False
    guest_name = None

if is_guest:
    # Hiển thị thông báo chế độ khách
    messagebox.showinfo("Chế độ khách", f"Bạn đang chơi với tư cách: {guest_name}")

def start_game():
    """Chạy sau khi đã chọn xong độ khó và kích thước bàn cờ"""
    try:
        # Lấy thông tin từ hopthoai.py thay vì đọc file
        difficulty, board_size = get_selected_options()
        
        if not board_size:
            messagebox.showerror("Lỗi", "Vui lòng chọn bàn cờ trước!")
            return

        # Ghi đầy đủ thông tin vào file
        with open("game_settings.txt", "w") as f:
            f.write(f"{difficulty}\n{board_size}\n{'AI' if difficulty in ['Dễ', 'Trung Bình', 'Khó'] else 'Friend'}")

        print(f"Bắt đầu game: {board_size} ({difficulty})")
        
        # Mở màn hình game
        subprocess.Popen(["python3", "manhinh3.py"])
        root.destroy()

    except Exception as e:
        messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {str(e)}")

def play_with_ai():
    choose_difficulty()
    # Sau khi chọn độ khó, bàn cờ sẽ tự động hiện trong hopthoai.py

def play_with_friend():
    # Tạo cửa sổ chọn chế độ
        choose_board_size("Đấu với bạn")

def ranked_match():
    """Kết nối server đấu hạng"""
    if messagebox.askyesno("Xác nhận", "Vào chế độ đấu hạng?\nĐiểm rank của bạn sẽ thay đổi sau trận đấu"):
        # Đóng màn hình hiện tại
        root.destroy()
        
        # Kết nối server đấu hạng
        try:
            subprocess.Popen(["python3", "manhinh4.py"])
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể kết nối: {str(e)}")
            subprocess.Popen(["python3", "manhinh2.py"])  # Quay lại màn hình chính

def show_match_found():
    first_player = toss_coin()
    messagebox.showinfo("Bắt đầu", f"{first_player} sẽ đi trước!")
    # Lưu cài đặt cho đấu hạng
    with open("game_settings.txt", "w") as f:
        f.write("Ranked\n3x3")  # Mặc định đấu hạng 3x3
    start_game()

def view_leaderboard():
    show_leaderboard()

def view_history():
    show_history()

def logout():
    root.destroy()
    subprocess.Popen(["python3", "dangnhap.py"])

# ==================== GIAO DIỆN CHÍNH ====================
root = tk.Tk()
root.title("Tic-Tac-Toe - Màn hình chính")
root.geometry("500x400")

# Tạo menu bar
menubar = tk.Menu(root)
root.config(menu=menubar)

# Menu trợ giúp
help_menu = tk.Menu(menubar, tearoff=0)
help_menu.add_command(label="Hướng dẫn", command=lambda: messagebox.showinfo("Hướng dẫn", "Chọn chế độ chơi và bắt đầu!"))
help_menu.add_separator()
help_menu.add_command(label="Giới thiệu", command=lambda: messagebox.showinfo("Giới thiệu", "Tic-Tac-Toe by Python"))
menubar.add_cascade(label="Trợ giúp", menu=help_menu)

# Phần chính
tk.Label(root, text="Chọn chế độ chơi", font=("Arial", 16, "bold")).pack(pady=20)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

tk.Button(button_frame, text="Đấu với máy", font=("Arial", 12), width=20, command=play_with_ai).pack(pady=5)
tk.Button(button_frame, text="Đấu với bạn", font=("Arial", 12), width=20, command=play_with_friend).pack(pady=5)
tk.Button(button_frame, text="Đấu hạng", font=("Arial", 12), width=20, command=ranked_match).pack(pady=5)
tk.Button(button_frame, text="Xếp hạng", font=("Arial", 12), width=20, command=view_leaderboard).pack(pady=5)
tk.Button(button_frame, text="Lịch sử", font=("Arial", 12), width=20, command=view_history).pack(pady=5)

# Footer
footer_frame = tk.Frame(root)
footer_frame.pack(side=tk.BOTTOM, pady=10)
tk.Button(footer_frame, text="Bắt đầu game", font=("Arial", 12), command=start_game).pack(side=tk.LEFT, padx=5)
tk.Button(footer_frame, text="Đăng xuất", font=("Arial", 12), fg="red", command=logout).pack(side=tk.RIGHT, padx=5)

root.mainloop()
