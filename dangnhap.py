import tkinter as tk
from tkinter import messagebox, PhotoImage
from PIL import Image, ImageTk
from tkinter import ttk
import subprocess
import re

def validate_input(text):
    # Chỉ cho phép chữ cái, số và không có khoảng trắng
    return bool(re.match("^[a-zA-Z0-9]*$", text))
        
def start_as_guest():
    """Bắt đầu chơi mà không cần đăng nhập"""
    if messagebox.askyesno("Xác nhận", "Bạn sẽ chơi với tư cách khách.\nLịch sử và điểm số sẽ không được lưu lại.\nTiếp tục?"):
        root.destroy()
        import random
        guest_name = "Khách_" + str(random.randint(1000, 9999))
        with open("guest.txt", "w") as f:
            f.write(guest_name)
        subprocess.Popen(["python3", "manhinh2.py"])

# Danh sách lưu tài khoản (chưa có database nên tạm thời dùng dictionary)
accounts = {}

# Hàm đăng nhập
def login():
    def toggle_password(entry, btn):
      if entry.cget('show') == '*':
        entry.config(show='')
        btn.config(text='👁')
      else:
        entry.config(show='*')
        btn.config(text='🔒')
    def check_login():
        username = entry_username.get().strip()
        password = entry_password.get().strip()
        
        if username in accounts and accounts[username] == password:
            messagebox.showinfo("Thành công", f"Chào mừng {username}!")
            login_window.destroy()
            import subprocess
            subprocess.Popen(["python3", "manhinh2.py"])
        else:
            messagebox.showerror("Lỗi", "Tài khoản không tồn tại, vui lòng đăng ký.")

    # Tạo cửa sổ
    login_window = tk.Tk()
    login_window.title("Đăng Nhập")
    login_window.geometry("500x400")
    login_window.configure(bg="pink")

    vcmd = (login_window.register(validate_input), '%P')

    # Frame chứa form login
    login_frame = tk.Frame(login_window, bg="pink")
    login_frame.pack(expand=True, fill="both", padx=50, pady=50)

    # Tiêu đề
    tk.Label(login_frame, text="ĐĂNG NHẬP", font=("Comic Sans MS", 23, "bold"), bg="pink").pack(pady=10)

    # Ô nhập tên đăng nhập
    tk.Label(login_frame, text="Tên đăng nhập:", bg="pink", font=("Comic Sans MS", 18)).pack()
    entry_username = ttk.Entry(login_frame, font=("Comic Sans MS", 18),validate="key",validatecommand=vcmd)
    entry_username.pack(pady=10, fill="x")

    # Ô nhập mật khẩu
    tk.Label(login_frame, text="Mật khẩu:", bg="pink", font=("Comic Sans MS", 18)).pack()

    entry_password = ttk.Entry(login_frame, show="*", font=("Comic Sans MS", 18),validate="key",validatecommand=vcmd)
    entry_password.pack(pady=10, fill="x")

    btn_show = tk.Button(entry_password, text="🔒", 
                    command=lambda: toggle_password(entry_password, btn_show))
    btn_show.pack(side="right")

    # Nút đăng nhập
    btn_login = ttk.Button(login_frame, text="ĐĂNG NHẬP", command=check_login)
    btn_login.pack(pady=20, ipadx=10, ipady=10)

def register():
    def save_account():
        username = entry_new_username.get().strip()
        password = entry_new_password.get().strip()
        confirm_password = entry_confirm_password.get().strip()

        if not username or not password:
            messagebox.showerror("Lỗi", "Tên đăng nhập và mật khẩu không được để trống!")
            return
        if username in accounts:
            messagebox.showerror("Lỗi", "Tài khoản đã tồn tại!")
        elif password != confirm_password:
            messagebox.showerror("Lỗi", "Mật khẩu nhập lại không khớp!")
        else:
            accounts[username] = password
            messagebox.showinfo("Thành công", "Đăng ký thành công! Bạn có thể đăng nhập ngay.")
            register_window.destroy()
    def toggle_password(entry, button):
        if entry['show'] == '*':
            entry.config(show='')
            button.config(text="🔒")
        else:
            entry.config(show='*')
            button.config(text="👁️")

    # Tạo cửa sổ đăng ký
    register_window = tk.Tk()
    register_window.title("Đăng Ký")
    register_window.geometry("600x500")
    register_window.configure(bg="pink")

    vcmd = (register_window.register(validate_input), '%P')

    # Frame chứa form đăng ký
    register_frame = tk.Frame(register_window, bg="pink")
    register_frame.pack(expand=True, fill="both", padx=50, pady=50)

# Tiêu đề
    tk.Label(register_frame, text="ĐĂNG KÝ", font=("Comic Sans MS", 23, "bold"), bg="pink").pack(pady=10)

    # Ô nhập tên đăng nhập mới
    tk.Label(register_frame, text="Tên đăng nhập:", bg="pink", font=("Comic Sans MS", 18)).pack()
    entry_new_username = ttk.Entry(register_frame, font=("Comic Sans MS", 18), validate="key", validatecommand=vcmd)
    entry_new_username.pack(pady=10, fill="x")

    # Ô nhập mật khẩu mới
    tk.Label(register_frame, text="Mật khẩu:", bg="pink", font=("Comic Sans MS", 18)).pack()

    frame_password = tk.Frame(register_frame, bg="pink")
    frame_password.pack(fill="x", pady=10,ipady=5)
    entry_new_password = ttk.Entry(register_frame, show="*", font=("Comic Sans MS", 18), validate="key", validatecommand=vcmd)
    entry_new_password.pack(ipady=5,pady=0, fill="x")
    
    btn_show_new = tk.Button(entry_new_password, text="🔒", 
                    command=lambda: toggle_password(entry_new_password, btn_show_new))
    btn_show_new.pack(side="right", padx=5)

    # Ô nhập lại mật khẩu
    tk.Label(register_frame, text="Nhập lại mật khẩu:", bg="pink", font=("Comic Sans MS", 18)).pack()
    
    entry_confirm_password = ttk.Entry(register_frame, show="*", font=("Comic Sans MS", 18), validate="key", validatecommand=vcmd)
    entry_confirm_password.pack(ipady=5,pady=10, fill="x")
    
    btn_show_confirm = tk.Button(entry_confirm_password, text="🔒", 
                    command=lambda: toggle_password(entry_confirm_password, btn_show_confirm))
    btn_show_confirm.pack(side="right")

    # Nút đăng ký
    btn_register = ttk.Button(register_frame, text="ĐĂNG KÝ", command=save_account)
    btn_register.pack(pady=20, ipadx=10, ipady=10)

# Hàm thoát game
def exit_game():
    root.destroy()

# Giao diện màn hình chính
root = tk.Tk()
root.title("Tic-Tac-Toe")
root.geometry("1440x900")

# Thêm ảnh nền
try:
    bg_img = Image.open("modau.jpg")
    bg_img = bg_img.resize((1440, 900), Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_img)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)
except Exception as e:
    messagebox.showerror("Lỗi", f"Không thể tải ảnh nền: {e}")

# Tiêu đề
title_label = tk.Label(root, text="Chào mừng đến với Tic-Tac-Toe", font=("Comic Sans MS", 30, "bold"), bg="#FFC107")
title_label.pack(pady=50)

# Hàm tải ảnh an toàn
def load_image(filename, size):
    try:
        img = Image.open(filename)
        img = img.resize(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể tải ảnh {filename}: {e}")
        return None

# Tải hình ảnh nút
login_icon = load_image("O.jpg", (60, 60))
register_icon = load_image("O.jpg", (60, 60))
exit_icon = load_image("X.jpg", (60, 60))
play_icon = load_image("O.jpg", (60, 60))  # Cần file play.jpg trong thư mục
btn_play = tk.Button(root, 
                    text="CHƠI NGAY", 
                    image=play_icon, 
                    compound="left", 
                    font=("Comic Sans MS", 30, "bold"), 
                    bg="#FFD700",
                    fg="pink", 
                    width=500,
                    height =80, 
                    command=start_as_guest)
btn_play.pack(pady=30, anchor= "center")

# Các nút chức năng
btn1 = tk.Button(root, text="ĐĂNG NHẬP", image=login_icon, compound="left", font=("Comic Sans MS", 30, "bold"), bg="#FFD700", fg="pink", width=500,height=80, command=login)
btn1.pack(pady=20, anchor = "center")

btn2 = tk.Button(root, text="ĐĂNG KÝ", image=register_icon, compound="left", font=("Comic Sans MS", 30, "bold"), bg="#FFD700", fg="pink", width=500,height=80, command=register)
btn2.pack(pady=20, anchor = "center")

btn3 = tk.Button(root, text="THOÁT TRÒ CHƠI", image=exit_icon, compound="left", font=("Comic Sans MS", 30, "bold"), bg="#FFD700", fg="pink", width=500,height=80, command=exit_game)
btn3.pack(pady=20, anchor = "center")

root.mainloop()


