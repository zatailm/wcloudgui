import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel, QPushButton, QVBoxLayout, QGridLayout, QWidget, QLineEdit, QComboBox, QSpinBox, QDialog, QTextEdit, QProgressBar, QFrame, QColorDialog, QInputDialog, QTableWidget, QTableWidgetItem
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QIcon
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import os
import PyPDF2
import docx
from PIL import Image
import numpy as np
import json

class FileLoaderThread(QThread):
    file_loaded = Signal(str, str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            if self.file_path.endswith(".txt"):
                with open(self.file_path, "r", encoding="utf-8") as file:
                    text_data = file.read()
            elif self.file_path.endswith(".pdf"):
                text_data = self.extract_text_from_pdf(self.file_path)
            elif self.file_path.endswith(".doc") or self.file_path.endswith(".docx"):
                text_data = self.extract_text_from_word(self.file_path)
            self.file_loaded.emit(self.file_path, text_data)
        except Exception as e:
            self.file_loaded.emit("", f"There is an error: {e}")

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def extract_text_from_word(self, word_path):
        doc = docx.Document(word_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

class WordCloudGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.file_path = ""
        self.text_data = ""
        self.mask_path = ""
        self.additional_stopwords = set()
        self.custom_color_palettes = {}
        self.current_figure = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('WCGen')
        self.setFixedSize(500, 600)
        self.setWindowIcon(QIcon('D:/python_proj/gs.ico'))

        layout = QGridLayout()

        self.file_label = QLineEdit('..... no file selected ..... :(', self)
        self.file_label.setFrame(True)
        self.file_label.setFixedHeight(30)
        self.file_label.setReadOnly(True)
        self.file_label.setToolTip('Selected file name')
        self.file_label.setStyleSheet("""
            QLineEdit {
                border: 1px solid #9a9a9a;  /* Green border */
                border-radius: 5px;  /* Rounded corners */
                padding: 0px;  /* Padding inside the text box */
                background-color: #f0f0f0;  /* Light grey background */
            }
            QLineEdit:focus {
                border: 1px solid red;  /* Darker green border when focused */
            }
        """)
        layout.addWidget(self.file_label, 0, 0, 1, 3)

        self.upload_button = QPushButton('Select File', self)
        self.upload_button.setFixedHeight(30)
        self.upload_button.setToolTip('Select a text, PDF, or Word file to generate a word cloud')
        self.upload_button.clicked.connect(self.pilih_file)
        layout.addWidget(self.upload_button, 1, 0, 1, 3)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0) 
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border-radius: 1px;
                background-color: white;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: red;
                width: 1px;  /* Ukuran kecil agar animasi terlihat lebih baik */
                margin: 0px; /* Pastikan tidak ada margin */
            }
        """)
        self.progress_bar.setContentsMargins(10, 0, 10, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar, 2, 0, 1, 3)

        self.view_text_button = QPushButton('View Full Text', self)
        self.view_text_button.setFixedHeight(30)
        self.view_text_button.setToolTip('View the full text in a separate window')
        self.view_text_button.clicked.connect(self.view_full_text)
        self.view_text_button.setEnabled(False)
        layout.addWidget(self.view_text_button, 3, 0, 1, 3)

        self.stopword_label = QLabel('Enter stopwords --separate each with semicolons (;):', self)
        layout.addWidget(self.stopword_label, 4, 0, 1, 3)
        self.stopword_entry = QLineEdit(self)
        self.stopword_entry.setFixedHeight(30)
        self.stopword_entry.setToolTip('Enter custom stopwords separated by semicolons (;)')
        layout.addWidget(self.stopword_entry, 5, 0, 1, 3)

        self.color_theme_label = QLabel('Color Theme:', self)
        layout.addWidget(self.color_theme_label, 6, 0, 1, 1)
        self.color_theme = QComboBox(self)
        self.color_theme.setFixedWidth(240)
        self.color_theme.addItems(plt.colormaps())
        self.color_theme.setToolTip('Select a color theme for the word cloud')
        layout.addWidget(self.color_theme, 6, 1, 1, 2)

        self.custom_palette_button = QPushButton('Custom', self)
        self.custom_palette_button.setFixedSize(75, 30)
        self.custom_palette_button.setToolTip('Create a custom color palette')
        self.custom_palette_button.clicked.connect(self.create_custom_palette)
        layout.addWidget(self.custom_palette_button, 6, 2, 1, 1, Qt.AlignRight)

        self.font_choice_label = QLabel('Font Choice:', self)
        layout.addWidget(self.font_choice_label, 8, 0, 1, 1)
        self.font_choice = QComboBox(self)
        self.font_choice.addItems(["Default", "arial.ttf", "times.ttf", "verdana.ttf"])
        self.font_choice.setToolTip('Select a font for the word cloud')
        layout.addWidget(self.font_choice, 8, 1, 1, 2)

        self.min_font_size_label = QLabel('Minimum Font Size:', self)
        layout.addWidget(self.min_font_size_label, 9, 0, 1, 1)
        self.min_font_size_entry = QSpinBox(self)
        self.min_font_size_entry.setValue(10)
        layout.addWidget(self.min_font_size_entry, 9, 1, 1, 2)

        self.max_words_label = QLabel('Maximum Words:', self)
        layout.addWidget(self.max_words_label, 10, 0, 1, 1)
        self.max_words_entry = QSpinBox(self)
        self.max_words_entry.setMaximum(10000)
        self.max_words_entry.setValue(200) 
        layout.addWidget(self.max_words_entry, 10, 1, 1, 2)

        self.bg_color_label = QLabel('Background Color:', self)
        layout.addWidget(self.bg_color_label, 11, 0, 1, 1)
        self.bg_color = QComboBox(self)
        self.bg_color.setFixedWidth(240)
        self.bg_color.addItems(["white", "black", "gray", "blue", "red", "yellow"])
        self.bg_color.setToolTip('Select a background color for the word cloud')
        layout.addWidget(self.bg_color, 11, 1, 1, 2)

        self.custom_bg_color_button = QPushButton('Custom', self)
        self.custom_bg_color_button.setFixedSize(75, 30)
        self.custom_bg_color_button.setToolTip('Select a custom background color')
        self.custom_bg_color_button.clicked.connect(self.select_custom_bg_color)
        layout.addWidget(self.custom_bg_color_button, 11, 2, 1, 1, Qt.AlignRight)

        self.mask_label = QLabel('Mask Image:', self)
        layout.addWidget(self.mask_label, 12, 0, 1, 1)
        self.mask_button = QPushButton('Mask Image', self)
        self.mask_button.setFixedHeight(30) 
        self.mask_button.setToolTip('Select an image to use as a mask for the word cloud')
        self.mask_button.clicked.connect(self.pilih_mask)
        layout.addWidget(self.mask_button, 12, 1, 1, 1)

        self.reset_mask_button = QPushButton('Remove Mask', self)
        self.reset_mask_button.setFixedHeight(30)
        self.reset_mask_button.setToolTip('Remove the selected mask image')
        self.reset_mask_button.clicked.connect(self.reset_mask)
        layout.addWidget(self.reset_mask_button, 12, 2, 1, 1)

        self.mask_path_label = QLabel('Mask: default (rectangle)', self)
        layout.addWidget(self.mask_path_label, 13, 0, 1, 3)

        self.buat_wordcloud_button = QPushButton('Generate WordCloud', self)
        self.buat_wordcloud_button.setFixedHeight(30) 
        self.buat_wordcloud_button.setToolTip('Generate the word cloud')
        self.buat_wordcloud_button.clicked.connect(self.buat_wordcloud)
        self.buat_wordcloud_button.setEnabled(False)
        layout.addWidget(self.buat_wordcloud_button, 14, 0, 1, 3)

        self.wordcloud_progress_bar = QProgressBar(self)
        self.wordcloud_progress_bar.setRange(0, 0) 
        self.wordcloud_progress_bar.setFixedHeight(4)
        self.wordcloud_progress_bar.setStyleSheet("""
            QProgressBar {
                border-radius: 1px;
                background-color: white;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: blue;
                width: 1px;  /* Small size for better animation */
                margin: 0px; /* Ensure no margin */
            }
        """)
        self.wordcloud_progress_bar.setVisible(False)
        layout.addWidget(self.wordcloud_progress_bar, 15, 0, 1, 3)

        self.statistik_button = QPushButton('Word Count', self)
        self.statistik_button.setFixedHeight(30)
        self.statistik_button.setToolTip('Show word frequency statistics')
        self.statistik_button.clicked.connect(self.tampilkan_statistik)
        self.statistik_button.setEnabled(False)
        layout.addWidget(self.statistik_button, 16, 0, 1, 3)

        self.simpan_button = QPushButton('Save WordCloud', self)
        self.simpan_button.setFixedHeight(30) 
        self.simpan_button.setToolTip('Save the generated word cloud')
        self.simpan_button.clicked.connect(self.simpan_wordcloud)
        self.simpan_button.setEnabled(False)
        layout.addWidget(self.simpan_button, 17, 0, 1, 3)

        self.about_button = QPushButton('About', self)
        self.about_button.setFixedSize(150, 30)
        self.about_button.setToolTip('Show information about the application')
        self.about_button.clicked.connect(self.show_about)
        layout.addWidget(self.about_button, 18, 0, 1, 3)

        self.quit_button = QPushButton('Quit', self)
        self.quit_button.setFixedHeight(30) 
        self.quit_button.setToolTip('Quit WCGen :(')
        self.quit_button.clicked.connect(self.close)
        layout.addWidget(self.quit_button, 18, 2, 1, 1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, ':(', "Are you sure you want to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            for widget in QApplication.topLevelWidgets():
                if widget is not self:
                    widget.close()
            event.accept()
        else:
            event.ignore()

    def show_about(self):
        about_text = (
            "WCGen\nVersion: 1.3\n\n(c) 2025 MAZ Ilmam\nhttps://github.com/zatailm/wcloudgui\n\n"
            "WordCloud is a text analysis technique used to visualize the frequency of words in a text, "
            "where words that appear more frequently are displayed in a larger size. This technique helps "
            "in quickly and intuitively understanding patterns and trends in textual data. The WCGen application, built with Python and a GUI, simplifies this process by providing an "
            "interactive interface for users. It supports various text formats, including txt, PDF and "
            "DOC/DOCX files, thanks to the use of wordcloud, matplotlib, tkinter, PyPDF2, and docx modules. "
            "With this application, users can easily input text, customize visualization parameters, and "
            "generate and save WordClouds for data analysis, research, or presentations.\n\n"
            "License: free for personal and educational purposes."
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("About")

        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setPlainText(about_text)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.show()

    def pilih_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Supported Files (*.txt *.pdf *.doc *.docx)", options=options)
        if not file_path:
            return

        self.progress_bar.setVisible(True)

        try:
            self.file_loader_thread = FileLoaderThread(file_path)
            self.file_loader_thread.file_loaded.connect(self.on_file_loaded)
            self.file_loader_thread.start()
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def on_file_loaded(self, file_path, text_data):
        self.progress_bar.setVisible(False)
        if file_path:
            self.text_data = text_data
            self.file_label.setText(os.path.basename(file_path))
            self.buat_wordcloud_button.setEnabled(True)
            self.simpan_button.setEnabled(True)
            self.statistik_button.setEnabled(True)
            self.view_text_button.setEnabled(True)
        else:
            QMessageBox.critical(self, "Error", text_data)

    def pilih_mask(self):
        options = QFileDialog.Options()
        mask_path, _ = QFileDialog.getOpenFileName(self, "Select Mask Image", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if mask_path:
            self.mask_path = mask_path
            self.mask_path_label.setText(f"Mask: {mask_path}")

    def reset_mask(self):
        self.mask_path = ""
        self.mask_path_label.setText("Mask: default (rectangle)")

    def ambil_stopwords(self):
        custom_words = self.stopword_entry.text().strip().lower()
        if custom_words:
            self.additional_stopwords = set(custom_words.split(";"))
        return STOPWORDS.union(self.additional_stopwords)

    def tampilkan_statistik(self):
        if not self.text_data:
            QMessageBox.critical(self, "Error", "Select file first!")
            return
        stopwords = self.ambil_stopwords()
        words = [word.lower() for word in self.text_data.split() if word.lower() not in stopwords]
        word_counts = Counter(words)
        
        sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Word Count")
 
        layout = QVBoxLayout()

        table = QTableWidget()
        table.setRowCount(len(sorted_word_counts))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Word", "Count"])
        for row, (word, count) in enumerate(sorted_word_counts):
            table.setItem(row, 0, QTableWidgetItem(word))
            table.setItem(row, 1, QTableWidgetItem(str(count)))
        table.resizeColumnsToContents()
        layout.addWidget(table)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.show()

    def buat_wordcloud(self):
        if not self.text_data:
            QMessageBox.critical(self, "Error", "Select file first!")
            return
        stopwords = self.ambil_stopwords()
        mask = None
        if self.mask_path:
            try:
                mask = np.array(Image.open(self.mask_path))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load mask image: {e}")
                return
        try:
            if self.current_figure:  
                plt.close(self.current_figure)
            
            self.wordcloud_progress_bar.setVisible(True) 
            QApplication.processEvents() 

            wc = WordCloud(
                width=800, height=400,
                background_color=self.bg_color.currentText(),
                stopwords=stopwords,
                colormap=self.color_theme.currentText(),
                max_words=self.max_words_entry.value(),
                min_font_size=self.min_font_size_entry.value(),
                mask=mask,
                font_path=None if self.font_choice.currentText() == "Default" else self.font_choice.currentText()
            ).generate(self.text_data)
            
            self.current_figure = plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.show()

            self.wordcloud_progress_bar.setVisible(False) 
        except Exception as e:
            self.wordcloud_progress_bar.setVisible(False) 
            QMessageBox.critical(self, "Error", f"Failed to generate word cloud: {e}")

    def simpan_wordcloud(self):
        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(self, "Save WordCloud", "", "PNG file (*.png);;JPG file (*.jpg)", options=options)
        if not save_path:
            return
        stopwords = self.ambil_stopwords()
        mask = None
        if self.mask_path:
            try:
                mask = np.array(Image.open(self.mask_path))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load mask image: {e}")
                return
        try:
            wc = WordCloud(
                width=800, height=400,
                background_color=self.bg_color.currentText(),
                stopwords=stopwords,
                colormap=self.color_theme.currentText(),
                max_words=self.max_words_entry.value(),
                min_font_size=self.min_font_size_entry.value(),
                mask=mask,
                font_path=None if self.font_choice.currentText() == "Default" else self.font_choice.currentText()
                            ).generate(self.text_data)
            wc.to_file(save_path)
            QMessageBox.information(self, "Succeed", "WordCloud saved!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save word cloud: {e}")

    def create_custom_palette(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Custom Palette")
        dialog.setFixedSize(400, 150)

        layout = QVBoxLayout()
        color_list = []

        def add_color():
            color = QColorDialog.getColor()
            if color.isValid():
                color_list.append(color.name())
                color_label.setText(", ".join(color_list))

        color_label = QLabel("", self)
        layout.addWidget(color_label)

        add_color_button = QPushButton("Add Color", self)
        add_color_button.clicked.connect(add_color)
        layout.addWidget(add_color_button)

        save_palette_button = QPushButton("Save Palette", self)
        save_palette_button.clicked.connect(lambda: self.save_custom_palette(color_list, dialog))
        layout.addWidget(save_palette_button)

        dialog.setLayout(layout)
        dialog.exec()

    def save_custom_palette(self, color_list, dialog):
        palette_name, ok = QInputDialog.getText(self, "Save Palette", "Enter palette name:")
        if ok and palette_name:
            self.custom_color_palettes[palette_name] = color_list
            self.color_theme.addItem(palette_name)
            dialog.accept()

    def select_custom_bg_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color.addItem(color.name())
            self.bg_color.setCurrentText(color.name())

    def view_full_text(self):
        if not self.text_data:
            QMessageBox.critical(self, "Error", "No text data available to display.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Full Text")

        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setPlainText(self.text_data)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WordCloudGenerator()
    ex.show()
    sys.exit(app.exec())