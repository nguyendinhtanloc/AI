import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import random
import subprocess
from hopthoai import toss_coin, choose_board_size

class RankedWaitingRoom:
    def __init__(self, master):
        self.master = master
        self.master.title("Tic-Tac-Toe - Phòng chờ Đấu hạng")
        self.master.geometry("500x400")
        self.master.resizable(False, False)
        
        # Thông tin người chơi và rank
        self.player_rank = 1200  # Giả lập ELO rating
        self.opponent_found = False
        self.search_time = 0
        
        # Tạo giao diện
        self.setup_ui()
        
        # Bắt đầu tìm đối thủ
        self.start_matchmaking()

    def setup_ui(self):
        """Xây dựng giao diện phòng chờ đấu hạng"""
        # Background
        self.bg_color = "#f0f2f5"
        self.master.configure(bg=self.bg_color)
        
        # Header
        header_frame = tk.Frame(self.master, bg="#3f51b5", height=80)
        header_frame.pack(fill=tk.X)
        
        tk.Label(header_frame, 
                text="PHÒNG CHỜ ĐẤU HẠNG", 
                font=("Arial", 18, "bold"), 
                bg="#3f51b5", fg="white").pack(pady=20)
        
        # Main content
        content_frame = tk.Frame(self.master, bg=self.bg_color, padx=20, pady=20)
        content_frame.pack(expand=True, fill=tk.BOTH)
        
        # Player info
        info_frame = tk.Frame(content_frame, bg=self.bg_color)
        info_frame.pack(fill=tk.X, pady=10)
        
        # Rank info
        rank_frame = tk.Frame(content_frame, bg="black", bd=2, relief=tk.GROOVE)
        rank_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(rank_frame, 
                text="THÔNG TIN XẾP HẠNG", 
                font=("Arial", 12, "bold"), 
                bg="black").pack(pady=5)
        
        tk.Label(rank_frame, 
                text=f"Điểm rank hiện tại: {self.player_rank}", 
                font=("Arial", 11), 
                bg="black").pack()
        
        # Search status
        self.status_label = tk.Label(content_frame, 
                                    text="Đang tìm đối thủ phù hợp...", 
                                    font=("Arial", 12), 
                                    bg=self.bg_color)
        self.status_label.pack(pady=10)
        
        # Time counter
        self.time_label = tk.Label(content_frame, 
                                 text="Thời gian tìm kiếm: 0 giây", 
                                 font=("Arial", 10), 
                                 bg=self.bg_color)
        self.time_label.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(content_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=15)
        
        # Estimated time
        self.estimate_label = tk.Label(content_frame, 
                                     text="Ước tính thời gian: 15-30 giây", 
                                     font=("Arial", 9), 
                                     fg="gray",
                                     bg=self.bg_color)
        self.estimate_label.pack()
        
        # Cancel button
        tk.Button(content_frame, 
                 text="Hủy tìm kiếm", 
                 command=self.cancel_search,
                 bg="#f44336",
                 fg="white",
                 font=("Arial", 12)).pack(pady=10)

    def start_matchmaking(self):
        """Bắt đầu quá trình tìm đối thủ"""
        self.progress.start()
        self.search_start_time = time.time()
        
        # Cập nhật thời gian tìm kiếm
        self.update_time()
        
        # Giả lập kết nối server trong thread riêng
        threading.Thread(target=self.simulate_server_connection, daemon=True).start()

    def update_time(self):
        """Cập nhật thời gian tìm kiếm"""
        if not self.opponent_found:
            self.search_time = int(time.time() - self.search_start_time)
            self.time_label.config(text=f"Thời gian tìm kiếm: {self.search_time} giây")
            self.master.after(1000, self.update_time)

    def simulate_server_connection(self):
        """Giả lập kết nối server tìm đối thủ"""
        # Giả lập thời gian tìm đối thủ (3-8 giây)
        time.sleep(random.uniform(3, 8))
        
        if not self.opponent_found:
            self.opponent_found = True
            self.master.after(0, self.opponent_matched)

    def opponent_matched(self):
        """Xử lý khi tìm thấy đối thủ"""
        self.progress.stop()
        self.status_label.config(text="Đã tìm thấy đối thủ!", fg="green")
        self.time_label.pack_forget()
        self.estimate_label.pack_forget()
        
        # Hiển thị thông tin đối thủ
        opponent_frame = tk.Frame(self.master, bg="black", bd=2, relief=tk.GROOVE)
        opponent_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(opponent_frame, 
                text="THÔNG TIN ĐỐI THỦ", 
                font=("Arial", 12, "bold"), 
                bg="black").pack(pady=5)
        
        tk.Label(opponent_frame, 
                text="Tên: Player_" + str(random.randint(1000, 9999)), 
                font=("Arial", 11), 
                bg="white").pack()
        
        tk.Label(opponent_frame, 
                text=f"Điểm rank: {self.player_rank + random.randint(-50, 50)}", 
                font=("Arial", 11), 
                bg="white").pack()
        
        # Nút bắt đầu
        tk.Button(self.master, 
                 text="BẮT ĐẦU TRẬN", 
                 command=self.start_match,
                 bg="#4CAF50",
                 fg="white",
                 font=("Arial", 12, "bold")).pack(pady=15)

    def start_match(self):
        """Bắt đầu trận đấu"""
        # Đóng phòng chờ
        self.master.destroy()
        
        # Tung đồng xu chọn người đi trước
        first_player = toss_coin()
        
        # Chọn bàn cờ (luôn 3x3 cho đấu hạng)
        choose_board_size("Ranked")
        
        # Lưu cài đặt trận đấu
        with open("game_settings.txt", "w") as f:
            f.write("Ranked\n3x3\nMultiplayer")
        
        # Khởi chạy màn hình game
        subprocess.Popen(["python3", "manhinh3.py"])

    def cancel_search(self):
        """Hủy tìm kiếm đối thủ"""
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn dừng tìm kiếm?"):
            self.master.destroy()
            # Quay lại màn hình chính
            subprocess.Popen(["python3", "manhinh2.py"])

def main():
    root = tk.Tk()
    app = RankedWaitingRoom(root)
    root.mainloop()

if __name__ == "__main__":
    main()