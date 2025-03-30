import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# Danh sách lưu tài khoản (chưa có database nên tạm thời dùng dictionary)
accounts = {}

# Hàm đăng nhập
def login():
    def check_login():
        username = entry_username.get()
        password = entry_password.get()
        
        if username in accounts and accounts[username] == password:
            messagebox.showinfo("Thành công", f"Chào mừng {username}!")
            login_window.destroy()
        else:
            messagebox.showerror("Lỗi", "Tài khoản không tồn tại, vui lòng đăng ký.")

    # Tạo cửa sổ đăng nhập
    login_window = tk.Toplevel(root)
    login_window.title("Đăng nhập")
    login_window.geometry("300x200")

    tk.Label(login_window, text="Tên đăng nhập:").pack()
    entry_username = tk.Entry(login_window)
    entry_username.pack()

    tk.Label(login_window, text="Mật khẩu:").pack()
    entry_password = tk.Entry(login_window, show="*")  # Ẩn mật khẩu
    entry_password.pack()

    tk.Button(login_window, text="Đăng nhập", command=check_login).pack(pady=10)

def toggle_password(entry, btn):
    if entry.cget('show') == '*':
        entry.config(show='')
        btn.config(text='🙈')
    else:
        entry.config(show='*')
        btn.config(text='👁')

# Hàm đăng ký
def register():
    def save_account():
        username = entry_new_username.get()
        password = entry_new_password.get()
        confirm_password = entry_confirm_password.get()

        if username in accounts:
            messagebox.showerror("Lỗi", "Tài khoản đã tồn tại!")
        elif password != confirm_password:
            messagebox.showerror("Lỗi", "Mật khẩu nhập lại không khớp!")
        else:
            accounts[username] = password
            messagebox.showinfo("Thành công", "Đăng ký thành công! Bạn có thể đăng nhập ngay.")
            register_window.destroy()

    # Tạo cửa sổ đăng ký
    register_window = tk.Toplevel(root)
    register_window.title("Đăng ký")
    register_window.geometry("350x250")

    tk.Label(register_window, text="Tên đăng ký:").pack()
    entry_new_username = tk.Entry(register_window)
    entry_new_username.pack()

    tk.Label(register_window, text="Mật khẩu:").pack()
    frame_password = tk.Frame(register_window)
    entry_new_password = tk.Entry(frame_password, show="*")
    entry_new_password.pack(side=tk.LEFT, fill=tk.X, expand=True)
    btn_toggle1 = tk.Button(frame_password, text="👁", command=lambda: toggle_password(entry_new_password, btn_toggle1))
    btn_toggle1.pack(side=tk.RIGHT)
    frame_password.pack(fill=tk.X)

    tk.Label(register_window, text="Nhập lại mật khẩu:").pack()
    frame_confirm = tk.Frame(register_window)
    entry_confirm_password = tk.Entry(frame_confirm, show="*")
    entry_confirm_password.pack(side=tk.LEFT, fill=tk.X, expand=True)
    btn_toggle2 = tk.Button(frame_confirm, text="👁", command=lambda: toggle_password(entry_confirm_password, btn_toggle2))
    btn_toggle2.pack(side=tk.RIGHT)
    frame_confirm.pack(fill=tk.X)

    tk.Button(register_window, text="Đăng ký", command=save_account).pack(pady=10)
# Hàm thoát game
def exit_game():
    root.destroy()

# Giao diện màn hình chính
root = tk.Tk()
root.title("Tic-Tac-Toe")
root.geometry("400x300")

tk.Label(root, text="Chào mừng đến với Tic-Tac-Toe", font=("Arial", 14, "bold")).pack(pady=20)

tk.Button(root, text="Đăng nhập", font=("Arial", 12), width=15, command=login).pack(pady=5)
tk.Button(root, text="Đăng ký", font=("Arial", 12), width=15, command=register).pack(pady=5)
tk.Button(root, text="Thoát trò chơi", font=("Arial", 12), width=15, command=exit_game).pack(pady=20)

root.mainloop()
