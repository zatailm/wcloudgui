import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import os

# Global variables
file_path = ""
text_data = ""
additional_stopwords = set()

# Fungsi untuk menampilkan informasi tentang aplikasi
def show_about():
    messagebox.showinfo("Tentang Aplikasi", "WordCloud Generator\nVersi 1.0\n\n(c) 2025 MAZ ILMAM\n\nAplikasi ini digunakan untuk membuat visualisasi WordCloud dari teks.\n\nDibuat dengan Python dengan modul tkinter, wordcloud, dan matplotlib.")

# Fungsi untuk memilih file teks
def pilih_file():
    global file_path, text_data
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if not file_path:
        return
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
        file_label.config(text=f"File: {os.path.basename(file_path)}", fg="blue", wraplength=450)
        buat_wordcloud_button.config(state="normal")
        simpan_button.config(state="normal")
        statistik_button.config(state="normal")
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan: {e}")

# Fungsi untuk mendapatkan stopwords
def ambil_stopwords():
    global additional_stopwords
    custom_words = stopword_entry.get().strip().lower()
    if custom_words:
        additional_stopwords = set(custom_words.split(";"))
    return STOPWORDS.union(additional_stopwords)

# Fungsi untuk membuat statistik kata
def tampilkan_statistik():
    if not text_data:
        messagebox.showerror("Error", "Silakan pilih file terlebih dahulu.")
        return
    stopwords = ambil_stopwords()
    words = [word.lower() for word in text_data.split() if word.lower() not in stopwords]
    word_counts = Counter(words)
    stat_window = tk.Toplevel(root)
    stat_window.title("Statistik Kata")
    stat_window.geometry("250x400")
    stat_window.resizable(False, True)
    text_area = tk.Text(stat_window, wrap="word", font=("Arial", 10))
    text_area.pack(expand=True, fill="both")
    for word, count in word_counts.most_common(30):
        text_area.insert("end", f"{word}: {count}\n")

# Fungsi untuk membuat WordCloud
def buat_wordcloud():
    global text_data
    if not text_data:
        messagebox.showerror("Error", "Silakan pilih file terlebih dahulu.")
        return
    stopwords = ambil_stopwords()
    wc = WordCloud(
        width=800, height=400,
        background_color=bg_color.get(),
        stopwords=stopwords,
        colormap=color_theme.get(),
        max_words=int(max_words_entry.get()),
        min_font_size=int(min_font_size_entry.get()),
        font_path=None if font_choice.get() == "Default" else font_choice.get()
    ).generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Fungsi untuk menyimpan WordCloud
def simpan_wordcloud():
    filetypes = [("PNG file", "*.png"), ("JPG file", "*.jpg")]
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=filetypes)
    if not save_path:
        return
    stopwords = ambil_stopwords()
    wc = WordCloud(
        width=800, height=400,
        background_color=bg_color.get(),
        stopwords=stopwords,
        colormap=color_theme.get(),
        max_words=int(max_words_entry.get()),
        min_font_size=int(min_font_size_entry.get()),
        font_path=None if font_choice.get() == "Default" else font_choice.get()
    ).generate(text_data)
    wc.to_file(save_path)
    messagebox.showinfo("Sukses", "WordCloud berhasil disimpan!")

# GUI Setup
root = tk.Tk()
root.title("WordCloud Generator")
root.geometry("500x500")
root.configure(bg="#f4f4f4")
root.resizable(False, False)

# Frame Utama
main_frame = tk.Frame(root, bg="white", padx=20, pady=20)
main_frame.pack(pady=10, padx=10, fill="both", expand=True)

# File Picker
upload_button = tk.Button(main_frame, text="Pilih File", command=pilih_file, bg="#3498db", fg="white", width=30)
upload_button.pack()
file_label = tk.Label(main_frame, text="File .txt: (Belum dipilih)", fg="red", bg="white")
file_label.pack()

# Stopwords
stopword_label = tk.Label(main_frame, text="Stopwords (pisahkan dengan ';'):", bg="white")
stopword_label.pack()
stopword_entry = tk.Entry(main_frame, width=50, bd=1, relief="solid")
stopword_entry.pack()

# Opsi Warna
color_theme = tk.StringVar(value="viridis")
tk.Label(main_frame, text="Pilih Warna Tema:", bg="white").pack()
color_dropdown = ttk.Combobox(main_frame, textvariable=color_theme, values=plt.colormaps(), state="readonly")
color_dropdown.pack()

# Opsi Font
font_choice = tk.StringVar(value="Default")
tk.Label(main_frame, text="Pilih Font:", bg="white").pack()
font_dropdown = ttk.Combobox(main_frame, textvariable=font_choice, values=["Default", "arial.ttf", "times.ttf", "verdana.ttf"], state="readonly")
font_dropdown.pack()

# Ukuran Kata Minimum
min_font_size_entry = tk.Entry(main_frame, width=5)
min_font_size_entry.insert(0, "10")
tk.Label(main_frame, text="Ukuran Kata Minimum:", bg="white").pack()
min_font_size_entry.pack()

# Jumlah Kata Maksimum
max_words_entry = tk.Entry(main_frame, width=5)
max_words_entry.insert(0, "200")
tk.Label(main_frame, text="Jumlah Kata Maksimum:", bg="white").pack()
max_words_entry.pack()

# Pilih Background Color
bg_color = tk.StringVar(value="white")
tk.Label(main_frame, text="Pilih Warna Background:", bg="white").pack()
bg_color_dropdown = ttk.Combobox(main_frame, textvariable=bg_color, values=["white", "black", "gray", "blue", "red", "yellow"], state="readonly")
bg_color_dropdown.pack()

# Tombol
button_frame = tk.Frame(main_frame, bg="white")
button_frame.pack(pady=10)
buat_wordcloud_button = tk.Button(button_frame, text="Buat WordCloud", command=buat_wordcloud, bg="#2ecc71", fg="white", state="disabled")
buat_wordcloud_button.grid(row=0, column=0, padx=5)
statistik_button = tk.Button(button_frame, text="Statistik Kata", command=tampilkan_statistik, bg="#e74c3c", fg="white", state="disabled")
statistik_button.grid(row=0, column=1, padx=5)
simpan_button = tk.Button(button_frame, text="Simpan Gambar", command=simpan_wordcloud, bg="#e67e22", fg="white", state="disabled")
simpan_button.grid(row=0, column=2, padx=5)

# Tombol About
about_button = tk.Button(root, text="About", command=show_about, bg="#34495e", fg="white")
about_button.pack(pady=5)

root.mainloop()
