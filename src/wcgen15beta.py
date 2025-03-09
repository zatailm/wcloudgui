# wcgen.py
import sys
from collections import Counter  # Add this line
import os
import asyncio
from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox,
                              QLabel, QPushButton, QVBoxLayout, QGridLayout, QWidget,
                              QLineEdit, QComboBox, QSpinBox, QDialog, QTextEdit,
                              QProgressBar, QFrame, QColorDialog, QInputDialog,
                              QTableWidget, QTableWidgetItem, QTextBrowser)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QIcon
import matplotlib
matplotlib.use('QtAgg')
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
import socket

class StartupThread(QThread):          # ✅ Thread untuk memuat dependensi berat tetapi umum di latar belakang
    def run(self):
        try:
            import pypdf
            import docx
            import pandas as pd
        except Exception as e:
            pass

class FileLoaderThread(QThread):        # ✅ Thread untuk memuat file teks dari berbagai format
    file_loaded = Signal(str, str)

    def __init__(self, file_path):      # ✅ Inisialisasi path file
        super().__init__()
        self.file_path = file_path

    def run(self):                      # ✅ Fungsi untuk memuat file
        try:
            if self.file_path.endswith(".txt"):
                with open(self.file_path, "r", encoding="utf-8") as file:
                    text_data = file.read()
            elif self.file_path.endswith(".pdf"):
                text_data = self.extract_text_from_pdf(self.file_path)
            elif self.file_path.endswith((".doc", ".docx")):
                text_data = self.extract_text_from_word(self.file_path)
            elif self.file_path.endswith((".xlsx", ".xls")):
                text_data = self.extract_text_from_excel(self.file_path)
            elif self.file_path.endswith(".csv"):
                text_data = self.extract_text_from_csv(self.file_path)
            else:
                text_data = "Error: Unsupported file format"

            self.file_loaded.emit(self.file_path, text_data)
        except Exception as e:
            self.file_loaded.emit("", f"Error loading file: {e}")

    def extract_text_from_pdf(self, pdf_path):          # ✅ Fungsi untuk mengekstrak teks dari PDF
        import pypdf  # Local import
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            return f"PDF Error: {str(e)}"
        return text

    def extract_text_from_word(self, word_path):        # ✅ Fungsi untuk mengekstrak teks dari Word
        import docx  # Local import
        try:
            doc = docx.Document(word_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            return f"Word Error: {str(e)}"

    def extract_text_from_excel(self, excel_path):      # ✅ Fungsi untuk mengekstrak teks dari Excel
        import pandas as pd  # Local import
        try:
            df = pd.read_excel(excel_path, sheet_name=None)
            text_data = ""
            for sheet_name, sheet_df in df.items():
                text_data += f"\n--- {sheet_name} ---\n"
                text_data += sheet_df.to_string(index=False, header=True)
            return text_data
        except Exception as e:
            return f"Excel Error: {e}"

    def extract_text_from_csv(self, csv_path):          # ✅ Fungsi untuk mengekstrak teks dari CSV
        import pandas as pd  # Local import
        try:
            df = pd.read_csv(csv_path)
            return df.to_string(index=False, header=True)
        except Exception as e:
            return f"CSV Error: {e}"

class CustomFileLoaderThread(QThread):                  # ✅ Thread untuk memuat file khusus seperti lexicon dan model
    file_loaded = Signal(object, bool)

    def __init__(self, file_path, file_type):           # ✅ Inisialisasi path file dan jenis file
        super().__init__()
        self.file_path = file_path
        self.file_type = file_type

    def run(self):                                      # ✅ Fungsi untuk memuat file
        try:
            if self.file_type == 'lexicon':
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.vader_analyzer = CustomVaderSentimentIntensityAnalyzer(custom_lexicon_file=self.file_path)
                self.file_loaded.emit(self.file_path, True)
            elif self.file_type == 'model':
                from flair.models import TextClassifier
                model = TextClassifier.load(self.file_path)
                self.file_loaded.emit(model, True)
        except Exception as e:
            self.file_loaded.emit(str(e), False)

class CustomVaderSentimentIntensityAnalyzer:            # ✅ Kelas untuk menganalisis sentimen dengan lexicon kustom
    def __init__(self, lexicon_file="vader_lexicon.txt", custom_lexicon_file=None):     # ✅ Inisialisasi lexicon
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()
        if custom_lexicon_file:
            self.load_custom_lexicon(custom_lexicon_file)

    def load_custom_lexicon(self, custom_lexicon_file):         # ✅ Fungsi untuk memuat lexicon kustom
        try:
            with open(custom_lexicon_file, 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        word, measure = parts
                        try:
                            self.analyzer.lexicon[word] = float(measure)
                        except ValueError:
                            pass
        except Exception:
            pass

    def polarity_scores(self, text):                            # ✅ Fungsi untuk menghitung skor sentimen
        return self.analyzer.polarity_scores(text)

class SentimentAnalysisThread(QThread):                 # ✅ Thread untuk menganalisis sentimen teks
    sentiment_analyzed = Signal(dict)
    offline_warning = Signal(str)

    def __init__(self, text_data, sentiment_mode, vader_analyzer, flair_classifier, flair_classifier_cuslang):      # ✅ Inisialisasi data teks dan mode sentimen
        super().__init__()
        self.text_data = text_data
        self.sentiment_mode = sentiment_mode
        self.vader_analyzer = vader_analyzer
        self.flair_classifier = flair_classifier
        self.flair_classifier_cuslang = flair_classifier_cuslang

    async def _async_translate(self, text):             # ✅ Fungsi untuk menerjemahkan teks asinkron
        try:
            from googletrans import Translator
            translator = Translator()
            translated = await translator.translate(text, dest='en')
            return translated.text
        except Exception as e:
            return None

    def translate_text(self, text):                     # ✅ Fungsi untuk menerjemahkan teks
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            translated = []
            for sentence in sentences:
                result = loop.run_until_complete(self._async_translate(sentence))
                if result:
                    translated.append(result)
            return '. '.join(translated)
        except Exception as e:
            return None
        finally:
            loop.close()

    def analyze_textblob(self, text):                   # ✅ Fungsi untuk menganalisis sentimen menggunakan TextBlob
        from textblob import TextBlob
        blob = TextBlob(text)
        sentiment = blob.sentiment
        return {
            "positive_score": max(sentiment.polarity, 0),
            "negative_score": max(-sentiment.polarity, 0),
            "neutral_score": 1 - abs(sentiment.polarity),
            "compound_score": sentiment.polarity,
            "sentiment_label": "POSITIVE" if sentiment.polarity > 0 else "NEGATIVE" if sentiment.polarity < 0 else "NEUTRAL",
            "subjectivity": sentiment.subjectivity
        }

    def run(self):                                      # ✅ Fungsi untuk menjalankan thread
        result = {
            "positive_score": 0,
            "neutral_score": 0,
            "negative_score": 0,
            "compound_score": 0,
            "sentiment_label": "N/A",
            "subjectivity": "N/A"
        }

        try:
            text_to_analyze = self.text_data
            needs_translation = False

            if self.sentiment_mode in ["VADER", "Flair", "TextBlob (Non-English)"]:
                try:
                    from langdetect import detect
                    detected_lang = detect(self.text_data)
                    needs_translation = detected_lang != "en"
                except:
                    needs_translation = False

            if needs_translation:
                if not is_connected():
                    self.offline_warning.emit("Translation required but no internet connection")
                    return
                
                translated_text = self.translate_text(self.text_data)
                if translated_text:
                    text_to_analyze = translated_text

            if self.sentiment_mode in ["TextBlob", "TextBlob (Non-English)"]:
                result.update(self.analyze_textblob(text_to_analyze))

            # ✅ Handle VADER analysis safely
            elif self.sentiment_mode == "VADER":
                if self.vader_analyzer:
                    sentiment = self.vader_analyzer.polarity_scores(text_to_analyze)
                    result.update({
                        "positive_score": sentiment['pos'],
                        "negative_score": sentiment['neg'],
                        "neutral_score": sentiment['neu'],
                        "compound_score": sentiment['compound'],
                        "sentiment_label": "POSITIVE" if sentiment['compound'] >= 0.05 else "NEGATIVE" if sentiment['compound'] <= -0.05 else "NEUTRAL"
                    })
                else:
                    result["sentiment_label"] = "Error: VADER not initialized"

            elif self.sentiment_mode == "VADER (Custom Lexicon)":
                if self.vader_analyzer:
                    sentiment = self.vader_analyzer.polarity_scores(self.text_data)
                    result.update({
                        "positive_score": sentiment['pos'],
                        "negative_score": sentiment['neg'],
                        "neutral_score": sentiment['neu'],
                        "compound_score": sentiment['compound'],
                        "sentiment_label": "POSITIVE" if sentiment['compound'] >= 0.05 else "NEGATIVE" if sentiment['compound'] <= -0.05 else "NEUTRAL"
                    })
                else:
                    result["sentiment_label"] = "Error: Custom VADER not initialized"

            elif self.sentiment_mode == "Flair":
                from flair.data import Sentence
                sentence = Sentence(text_to_analyze)
                self.flair_classifier.predict(sentence)
                sentiment = sentence.labels[0]
                result.update({
                    "compound_score": sentiment.score,
                    "sentiment_label": sentiment.value,
                    "positive_score": sentiment.score if sentiment.value == 'POSITIVE' else 0,
                    "negative_score": sentiment.score if sentiment.value == 'NEGATIVE' else 0,
                    "neutral_score": 1 - sentiment.score
                })

            elif self.sentiment_mode == "Flair (Custom Model)":
                from flair.data import Sentence
                sentence = Sentence(self.text_data)
                self.flair_classifier_cuslang.predict(sentence)
                sentiment = sentence.labels[0]
                result.update({
                    "compound_score": sentiment.score,
                    "sentiment_label": sentiment.value,
                    "positive_score": sentiment.score if sentiment.value == 'POSITIVE' else 0,
                    "negative_score": sentiment.score if sentiment.value == 'NEGATIVE' else 0,
                    "neutral_score": 1 - sentiment.score
                })

            self.sentiment_analyzed.emit(result)
        except Exception as e:
            result["sentiment_label"] = f"Error: {str(e)}"
            self.sentiment_analyzed.emit(result)

class FlairModelLoaderThread(QThread):              # ✅ Thread untuk memuat model Flair
    model_loaded = Signal(object)
    error_occurred = Signal(str)
    
    _cached_model = None  # Class-level cache

    def __init__(self, model_path='sentiment'):     # ✅ Inisialisasi path model
        super().__init__()
        self.model_path = model_path

    def run(self):                                  # ✅ Fungsi untuk memuat model
        try:
            import time
            from flair.models import TextClassifier
            if FlairModelLoaderThread._cached_model is None:
                time.sleep(0.1)
                model = TextClassifier.load(self.model_path)
                FlairModelLoaderThread._cached_model = model
            else:
                model = FlairModelLoaderThread._cached_model
            self.model_loaded.emit(model)
        except Exception as e:
            self.error_occurred.emit(str(e))


class WordCloudGenerator(QMainWindow):              # ✅ Kelas utama untuk GUI
    def __init__(self):                             # ✅ Inisialisasi variabel
        super().__init__()
        self.file_path = ""
        self.text_data = ""
        self.mask_path = ""
        self.additional_stopwords = set()
        self.custom_color_palettes = {}
        self.current_figure = None

        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.sentiment_mode = "TextBlob"
        self.custom_lexicon_path = None

        self.custom_model_path = None
        self.flair_classifier = None
        self.flair_classifier_cuslang = None
        self.active_threads = []

        self.file_loader_thread = None
        self.startup_thread = StartupThread()
        self.startup_thread.start()
        
        self.initUI()

    def initUI(self):                                   # ✅ Fungsi untuk membuat GUI
        self.setWindowTitle('WCGen')
        self.setFixedSize(550, 750)
        self.setWindowIcon(QIcon('D:/python_proj/wcloudgui/res/gs.ico'))

        layout = QGridLayout()

        # File selection widgets
        self.file_label = QLineEdit(self)                       # ✅ Label untuk menampilkan nama file  
        self.file_label.setReadOnly(True)
        self.file_label.setFixedHeight(30)
        self.file_label.setPlaceholderText("No file selected")
        layout.addWidget(self.file_label, 0, 0, 1, 6)

        self.upload_button = QPushButton('Load Text File', self)   # ✅ Tombol untuk memilih file
        self.upload_button.clicked.connect(self.pilih_file)
        self.upload_button.setFixedHeight(30)
        layout.addWidget(self.upload_button, 1, 0, 1, 4)

        self.view_text_button = QPushButton('View Full Text', self)  # ✅ Tombol untuk melihat teks penuh
        self.view_text_button.setFixedHeight(30)
        self.view_text_button.setToolTip('View the full text in a separate window')
        self.view_text_button.clicked.connect(self.view_full_text)
        self.view_text_button.setEnabled(False)
        layout.addWidget(self.view_text_button, 1, 4, 1, 2)

        self.progress_bar = QProgressBar(self)  # ✅ Progress bar untuk menunjukkan proses loading file
        self.progress_bar.setRange(0, 0) 
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border-radius: 1px;
                background-color: white;
                text-align: center;
                margin-left: 5px;  
                margin-right: 5px;                        
            }
            QProgressBar::chunk {
                background-color: red;
                width: 1px;  /* Ukuran kecil agar animasi terlihat lebih baik */
                margin: 0px; /* Pastikan tidak ada margin */
            }
        """)
        self.progress_bar.setContentsMargins(10, 0, 10, 0) 
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar, 2, 0, 1, 6)
        
        self.stopword_entry = QTextEdit(self)   # ✅ Input untuk stopwords
        self.stopword_entry.setFixedHeight(75)
        self.stopword_entry.setPlaceholderText("Enter stopwords, separated by spaces or new lines (optional)")
        self.stopword_entry.setToolTip('Stopwords')
        layout.addWidget(self.stopword_entry, 3, 0, 1, 6)

        # --------------------------------------------------------------------------------

        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        layout.addWidget(hline, 4, 0, 1, 6) 

        self.color_theme_label = QLabel('Color Theme:', self)   # ✅ Pilihan tema warna
        layout.addWidget(self.color_theme_label, 5, 0, 1, 2)

        self.color_theme = QComboBox(self)  # ✅ Pilihan tema warna
        self.color_theme.setFixedHeight(30)
        QTimer.singleShot(100, self.load_colormaps) 
        self.color_theme.setToolTip('Select a color theme for the word cloud')
        layout.addWidget(self.color_theme, 5, 2, 1, 3)

        self.custom_palette_button = QPushButton('Custom', self)    # ✅ Tombol untuk membuat palet warna kustom
        self.custom_palette_button.setFixedHeight(30)
        self.custom_palette_button.setToolTip('Create a custom color palette')
        self.custom_palette_button.clicked.connect(self.create_custom_palette)
        layout.addWidget(self.custom_palette_button, 5, 5, 1, 1)        

        self.bg_color_label = QLabel('Background Color:', self) # ✅ Pilihan warna latar belakang
        layout.addWidget(self.bg_color_label, 7, 0, 1, 2)

        self.bg_color = QComboBox(self) # ✅ Pilihan warna latar belakang
        self.bg_color.setFixedHeight(30)
        self.bg_color.addItems(["white", "black", "gray", "blue", "red", "yellow"])
        self.bg_color.setToolTip('Select a background color for the word cloud')
        layout.addWidget(self.bg_color, 7, 2, 1, 3)

        self.custom_bg_color_button = QPushButton('Custom', self)   # ✅ Tombol untuk memilih warna latar belakang kustom
        self.custom_bg_color_button.setFixedHeight(30)
        self.custom_bg_color_button.setToolTip('Select a custom background color')
        self.custom_bg_color_button.clicked.connect(self.select_custom_bg_color)
        layout.addWidget(self.custom_bg_color_button, 7, 5, 1, 1)

        # --------------------------------------------------------------------------------

        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        layout.addWidget(hline, 8, 0, 1, 6) 

        self.title_label = QLabel('WordCloud Title:', self) # ✅ Input untuk Judul WordCloud
        layout.addWidget(self.title_label, 9, 0, 1, 2)

        self.title_entry = QLineEdit(self)          # ✅ Input untuk Judul WordCloud
        self.title_entry.setFixedHeight(30)
        self.title_entry.setPlaceholderText("Enter title (optional)")
        layout.addWidget(self.title_entry, 9, 2, 1, 2)

        self.title_font_size = QSpinBox(self)       # ✅ Input untuk ukuran font judul
        self.title_font_size.setRange(8, 72)    
        self.title_font_size.setValue(14)
        self.title_font_size.setFixedHeight(30)
        layout.addWidget(self.title_font_size, 9, 4, 1, 1)

        self.title_position = QComboBox(self)       # ✅ Pilihan posisi judul
        self.title_position.addItems(['Left', 'Center', 'Right'])
        self.title_position.setCurrentText('Center')
        self.title_position.setFixedHeight(30)
        layout.addWidget(self.title_position, 9, 5, 1, 1)        

        self.font_choice_label = QLabel('Font Choice:', self)   # ✅ Pilihan font
        layout.addWidget(self.font_choice_label, 10, 0, 1, 2)

        self.font_choice = QComboBox(self)  # ✅ Pilihan font
        self.font_choice.setFixedHeight(30)
        self.font_choice.addItem("Default") 
        self.font_choice.setToolTip('Select a font family for the word cloud')
        layout.addWidget(self.font_choice, 10, 2, 1, 4)
        QTimer.singleShot(100, self.load_matplotlib_fonts) 

        # --------------------------------------------------------------------------------

        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        layout.addWidget(hline, 11, 0, 1, 6)         

        self.min_font_size_label = QLabel('Minimum Font Size:', self)      # ✅ Input untuk ukuran font minimum
        layout.addWidget(self.min_font_size_label, 12, 0, 1, 2)

        self.min_font_size_entry = QSpinBox(self)       # ✅ Input untuk ukuran font minimum
        self.min_font_size_entry.setFixedHeight(30)
        self.min_font_size_entry.setValue(11)
        layout.addWidget(self.min_font_size_entry, 12, 2, 1, 1)

        self.max_words_label = QLabel('Maximum Words:', self)       # ✅ Input untuk jumlah kata maksimum
        layout.addWidget(self.max_words_label, 12, 3, 1, 2, Qt.AlignRight)

        self.max_words_entry = QSpinBox(self)       # ✅ Input untuk jumlah kata maksimum
        self.max_words_entry.setFixedHeight(30)
        self.max_words_entry.setMaximum(10000)
        self.max_words_entry.setValue(200) 
        layout.addWidget(self.max_words_entry, 12, 5, 1, 1)

        # --------------------------------------------------------------------------------

        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        layout.addWidget(hline, 13, 0, 1, 6) 

        self.mask_label = QLabel('Mask Image:', self)       # ✅ Input untuk mask image
        layout.addWidget(self.mask_label, 14, 0, 1, 2)

        self.mask_path_label = QLineEdit('default (rectangle)', self)       # ✅ Input untuk mask image
        self.mask_path_label.setReadOnly(True)
        self.mask_path_label.setFixedHeight(30)
        layout.addWidget(self.mask_path_label, 14, 2, 1, 4)

        self.mask_button = QPushButton('Load Mask Image', self)     # ✅ Tombol untuk memilih mask image
        self.mask_button.setFixedHeight(30) 
        self.mask_button.setToolTip('Select an image to use as a mask for the word cloud')
        self.mask_button.clicked.connect(self.pilih_mask)
        layout.addWidget(self.mask_button, 15, 2, 1, 2)

        self.reset_mask_button = QPushButton('Remove Mask Image', self)     # ✅ Tombol untuk menghapus mask image
        self.reset_mask_button.setFixedHeight(30)
        self.reset_mask_button.setToolTip('Remove the selected mask image')
        self.reset_mask_button.clicked.connect(self.reset_mask)
        layout.addWidget(self.reset_mask_button, 15, 4, 1, 2)

        # --------------------------------------------------------------------------------

        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        layout.addWidget(hline, 16, 0, 1, 6) 

        self.buat_wordcloud_button = QPushButton('Generate WordCloud', self)        # ✅ Tombol untuk membuat wordcloud
        self.buat_wordcloud_button.setFixedHeight(50) 
        self.buat_wordcloud_button.setToolTip('Generate the word cloud')
        self.buat_wordcloud_button.clicked.connect(self.buat_wordcloud)
        self.buat_wordcloud_button.setEnabled(False)
        layout.addWidget(self.buat_wordcloud_button, 17, 0, 1, 6)

        self.wordcloud_progress_bar = QProgressBar(self)        # ✅ Progress bar untuk menunjukkan proses pembuatan wordcloud
        self.wordcloud_progress_bar.setRange(0, 0) 
        self.wordcloud_progress_bar.setFixedHeight(4)
        self.wordcloud_progress_bar.setStyleSheet("""
            QProgressBar {
                border-radius: 1px;
                background-color: white;
                text-align: center;
                margin-left: 5px;  
                margin-right: 5px;
            }
            QProgressBar::chunk {
                background-color: blue;
                width: 1px;  /* Small size for better animation */
                margin: 0px; /* Ensure no margin */
            }
        """)
        self.wordcloud_progress_bar.setVisible(False)
        layout.addWidget(self.wordcloud_progress_bar, 18, 0, 1, 6)

        self.statistik_button = QPushButton('Word Count', self)         # ✅ Tombol untuk menampilkan statistik kata
        self.statistik_button.setFixedHeight(30)
        self.statistik_button.setToolTip('Show word frequency statistics')
        self.statistik_button.clicked.connect(self.tampilkan_statistik)
        self.statistik_button.setEnabled(False)
        layout.addWidget(self.statistik_button, 19, 0, 1, 3)

        self.simpan_button = QPushButton('Save WordCloud', self)        # ✅ Tombol untuk menyimpan wordcloud
        self.simpan_button.setFixedHeight(30) 
        self.simpan_button.setToolTip('Save the generated word cloud')
        self.simpan_button.clicked.connect(self.simpan_wordcloud)
        self.simpan_button.setEnabled(False)
        layout.addWidget(self.simpan_button, 19, 3, 1, 3)

        # --------------------------------------------------------------------------------

        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        layout.addWidget(hline, 20, 0, 1, 6) 

        self.sentiment_mode_label = QLabel('Sentiment Analysis Mode:', self)        # ✅ Pilihan mode analisis sentimen
        layout.addWidget(self.sentiment_mode_label, 21, 0, 1, 2)

        self.sentiment_mode_combo = QComboBox(self)             # ✅ Pilihan mode analisis sentimen
        self.sentiment_mode_combo.setFixedHeight(30)
        self.sentiment_mode_combo.addItems(["TextBlob", "TextBlob (Non-English)", "VADER", "VADER (Custom Lexicon)", "Flair", "Flair (Custom Model)"])
        self.sentiment_mode_combo.setToolTip('Select sentiment analysis mode')
        self.sentiment_mode_combo.setCurrentText("TextBlob")
        self.sentiment_mode_combo.currentTextChanged.connect(self.change_sentiment_mode)
        layout.addWidget(self.sentiment_mode_combo, 21, 2, 1, 3)        

        self.sentiment_mode_info_button = QPushButton('Info', self)         # ✅ Tombol untuk menampilkan informasi mode analisis sentimen
        self.sentiment_mode_info_button.setFixedHeight(30)
        self.sentiment_mode_info_button.setToolTip('Show description for each sentiment analysis mode')
        self.sentiment_mode_info_button.clicked.connect(self.show_sentiment_mode_info)
        layout.addWidget(self.sentiment_mode_info_button, 21, 5, 1, 1)

        self.custom_lexicon_button = QPushButton('Load Lexicon', self)          # ✅ Tombol untuk memuat lexicon kustom
        self.custom_lexicon_button.setFixedHeight(30)
        self.custom_lexicon_button.setToolTip('Load a custom lexicon file for VADER')
        self.custom_lexicon_button.clicked.connect(self.load_custom_lexicon)
        self.custom_lexicon_button.setEnabled(False)
        layout.addWidget(self.custom_lexicon_button, 22, 2, 1, 2)

        self.custom_model_button = QPushButton('Load Model', self)          # ✅ Tombol untuk memuat model kustom
        self.custom_model_button.setFixedHeight(30)
        self.custom_model_button.setToolTip('Load a custom model file for Flair')
        self.custom_model_button.clicked.connect(self.load_custom_model)
        self.custom_model_button.setEnabled(False)
        layout.addWidget(self.custom_model_button, 22, 4, 1, 2)        

        self.model_progress_bar = QProgressBar(self)            # ✅ Progress bar untuk menunjukkan proses memuat model
        self.model_progress_bar.setRange(0, 0) 
        self.model_progress_bar.setFixedHeight(4)
        self.model_progress_bar.setStyleSheet("""
            QProgressBar {
                border-radius: 1px;
                background-color: white;
                text-align: center;
                margin-left: 5px;  
                margin-right: 5px;
            }
            QProgressBar::chunk {
                background-color: orange;
                width: 1px; 
                margin: 0px; 
            }
        """)
        self.model_progress_bar.setVisible(False)
        layout.addWidget(self.model_progress_bar, 23, 2, 1, 4)

        self.sentiment_button = QPushButton('Analyze Sentiment', self)          # ✅ Tombol untuk menganalisis sentimen
        self.sentiment_button.setFixedHeight(50)
        self.sentiment_button.setToolTip('Analyze the sentiment of the text')
        self.sentiment_button.clicked.connect(self.analyze_sentiment)
        self.sentiment_button.setEnabled(False)
        layout.addWidget(self.sentiment_button, 24, 0, 1, 6)

        # --------------------------------------------------------------------------------

        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        layout.addWidget(hline, 25, 0, 1, 6) 

        self.about_button = QPushButton('About', self)          # ✅ Tombol untuk menampilkan informasi aplikasi
        self.about_button.setFixedHeight(30)
        self.about_button.setToolTip('Show information about the application')
        self.about_button.clicked.connect(self.show_about)
        layout.addWidget(self.about_button, 26, 0, 1, 2)

        self.panic_button = QPushButton('STOP', self)               # ✅ Tombol untuk menghentikan semua proses
        self.panic_button.setFixedHeight(30)
        self.panic_button.setToolTip('Stop all ongoing processes')
        self.panic_button.clicked.connect(self.stop_all_processes)
        layout.addWidget(self.panic_button, 26, 2, 1, 2) 

        self.quit_button = QPushButton('Quit', self)            # ✅ Tombol untuk keluar dari aplikasi
        self.quit_button.setFixedHeight(30) 
        self.quit_button.setToolTip('Quit WCGen :(')
        self.quit_button.clicked.connect(self.close)
        layout.addWidget(self.quit_button, 26, 4, 1, 2)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # Add new method to load fonts
    def load_matplotlib_fonts(self):        # ✅ Fungsi untuk memuat font matplotlib
        try:
            from matplotlib import font_manager
            
            self.font_map = {}  # Format: {"Display Name": "font_path"}
            
            weight_conversion = {
                100: 'Thin',
                200: 'Extra Light',
                300: 'Light',
                400: 'Regular',
                500: 'Medium',
                600: 'Semi Bold',
                700: 'Bold',
                800: 'Extra Bold',
                900: 'Black'
            }

            for font in font_manager.fontManager.ttflist:
                try:
                    # Ambil nama family dan style sebenarnya
                    family = font.family_name if hasattr(font, 'family_name') else font.name
                    style = font.style_name.lower() if hasattr(font, 'style_name') else font.style.lower()
                    
                    # Convert weight numerik ke nama
                    weight = weight_conversion.get(font.weight, str(font.weight))
                    
                    # Format nama tampilan
                    display_parts = [family]
                    
                    # Handle style khusus
                    if 'italic' in style or 'oblique' in style:
                        display_parts.append('Italic')
                    elif 'normal' not in style:
                        display_parts.append(style.title())
                    
                    # Handle weight
                    if weight != 'Regular':
                        display_parts.append(weight)
                    
                    display_name = ' '.join(display_parts)
                    
                    # Simpan path font (pastikan unik)
                    if display_name not in self.font_map:
                        self.font_map[display_name] = font.fname
                        
                except Exception as e:
                    continue

            # Urutkan dan tambahkan ke ComboBox
            self.font_choice.clear()
            self.font_choice.addItem("Default")
            self.font_choice.addItems(sorted(self.font_map.keys()))
            
        except Exception as e:
            QMessageBox.warning(self, "Font Error", 
                            f"Failed to load fonts: {str(e)}")
            self.font_choice.addItems(["Arial", "Times New Roman", "Verdana"])

    def load_colormaps(self):
        """Load matplotlib colormaps with deferred import"""
        import matplotlib.pyplot as plt
        try:
            import matplotlib.pyplot as plt
            self.color_theme.addItems(plt.colormaps())
        except ImportError as e:
            self.color_theme.clear()
            self.color_theme.addItem("Default")
            QMessageBox.warning(self, "Dependency Error", 
                              f"Failed to load color maps: {str(e)}")

    def analyze_sentiment(self):
        if not self.text_data:
            QMessageBox.critical(self, "Error", "No text data available for sentiment analysis./nPlease load a text file first.")
            return
        
        # Validasi mode sentimen
        if self.sentiment_mode == "Flair" and not self.flair_classifier:
            QMessageBox.warning(self, "Model Loading", 
                            "Default Flair model is still loading. Please wait...")
            return
        
        if self.sentiment_mode == "Flair (Custom Model)":
            if not self.flair_classifier_cuslang:
                QMessageBox.warning(self, "Model Required", 
                                "Please load a custom Flair model first!")
                return
        
        # Use custom model for analysis
        classifier = self.flair_classifier_cuslang if self.sentiment_mode == "Flair (Custom Model)" else self.flair_classifier

        # Proceed with analysis
        self.progress_bar.show()
        self.sentiment_thread = SentimentAnalysisThread(
            self.text_data, self.sentiment_mode, self.vader_analyzer, 
            classifier, self.flair_classifier_cuslang
        )
        self.sentiment_thread.offline_warning.connect(self.handle_offline_warning)
        self.sentiment_thread.sentiment_analyzed.connect(self.on_sentiment_analyzed)
        self.active_threads.append(self.sentiment_thread)
        self.sentiment_thread.start()


    # Keep all other methods identical to original below this point
    def stop_all_processes(self):
            for thread in self.active_threads:
                if thread.isRunning():
                    thread.terminate() 
                    thread.wait()  
            self.progress_bar.setVisible(False)
            self.model_progress_bar.setVisible(False)
            self.model_progress_bar.setVisible(False)
            self.active_threads.clear() 
            QMessageBox.information(self, "Processes Stopped", "All ongoing processes have been stopped.")

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
            "<h2>WCGen - WordCloud Generator</h2>"
            "<p><b>Version:</b> 1.5</p>"
            "<p>&copy; 2025 MAZ Ilmam</p>"
            '<p><a href="https://github.com/zatailm/wcloudgui">GitHub Repository</a></p>'
            "<h3>About WCGen</h3>"
            "<p>WCGen is a powerful and user-friendly application designed for generating word clouds from text data. "
            "It helps users quickly visualize word frequency, making it useful for text analysis, research, and presentations.</p>"
            "<h3>Features</h3>"
            "<ul>"
            "<li>Supports multiple file formats: <b>TXT, PDF, DOC/DOCX, CSV, XLSX</b></li>"
            "<li>Customizable visualization: <b>colors, fonts, mask images, and themes</b></li>"
            "<li>Stopword filtering for cleaner word clouds</li>"
            "<li>Easy export and saving options</li>"
            "<li>Optional sentiment analysis using <b>TextBlob, VADER, and Flair</b></li>"
            "</ul>"
            "<h3>License</h3>"
            "<p>WCGen is free for personal and educational use. For commercial use, please refer to the licensing terms.</p>"
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("About WCGen")
        layout = QVBoxLayout()
        
        text_browser = QTextBrowser()
        text_browser.setHtml(about_text)
        text_browser.setOpenExternalLinks(True)
        text_browser.setReadOnly(True)
        
        layout.addWidget(text_browser)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.setLayout(layout)
        dialog.exec()


    def pilih_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Text File", 
            "", 
            "Supported Files (*.txt *.pdf *.doc *.docx *.csv *.xlsx *.xls);;All Files (*)", 
            options=options
        )
        if not file_path:
            return

        self.progress_bar.setVisible(True)

        try:
            self.file_loader_thread = FileLoaderThread(file_path)
            self.file_loader_thread.file_loaded.connect(self.on_file_loaded)
            self.active_threads.append(self.file_loader_thread)
            self.file_loader_thread.start()
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")


    def on_file_loaded(self, file_path, text_data):
        self.progress_bar.setVisible(False)

        if file_path and text_data and text_data.strip():  # ✅ Pastikan text_data tidak None sebelum strip()
            self.text_data = text_data
            self.file_label.setText(os.path.basename(file_path))

            # ✅ Aktifkan tombol yang diperlukan setelah file dimuat
            self.buat_wordcloud_button.setEnabled(True)
            self.simpan_button.setEnabled(True)
            self.statistik_button.setEnabled(True)
            self.view_text_button.setEnabled(True)
            self.sentiment_button.setEnabled(True)  # ✅ Baru aktif setelah file dimuat
        else:
            QMessageBox.critical(self, "Error", "Failed to load file or file is empty.")
            
            # ✅ Pastikan semua tombol tetap nonaktif jika file gagal dimuat
            self.buat_wordcloud_button.setEnabled(False)
            self.simpan_button.setEnabled(False)
            self.statistik_button.setEnabled(False)
            self.view_text_button.setEnabled(False)
            self.sentiment_button.setEnabled(False)


    def pilih_mask(self):
        options = QFileDialog.Options()
        mask_path, _ = QFileDialog.getOpenFileName(self, "Select Mask Image", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if mask_path:
            self.mask_path = mask_path
            self.mask_path_label.setText(f"{mask_path}")

    def reset_mask(self):
        self.mask_path = ""
        self.mask_path_label.setText("default (rectangle)")

    def ambil_stopwords(self):
        custom_words = self.stopword_entry.toPlainText().strip().lower()
        if custom_words:
            self.additional_stopwords = set(custom_words.split())  # ✅ Split by whitespace
        return STOPWORDS.union(self.additional_stopwords)

    def tampilkan_statistik(self):
        if not self.text_data:
            QMessageBox.critical(self, "Error", "Load text file first!")
            return

        stopwords = self.ambil_stopwords()
        words = [word.lower() for word in self.text_data.split() if word.lower() not in stopwords]
        word_counts = Counter(words)

        sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

        # Tutup dialog sebelumnya jika masih terbuka
        if hasattr(self, "stats_dialog") and self.stats_dialog is not None:
            self.stats_dialog.close()

        self.stats_dialog = QDialog(self)
        self.stats_dialog.setWindowTitle("Word Count")
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
        close_button.clicked.connect(self.stats_dialog.accept)
        layout.addWidget(close_button)

        self.stats_dialog.setLayout(layout)
        self.stats_dialog.show()

    def buat_wordcloud(self):
        import matplotlib.pyplot as plt

        font_path = None
        if self.font_choice.currentText() != "Default":
            selected_font = self.font_choice.currentText()
            font_path = self.font_map.get(selected_font)
            
            # Validasi path
            if not font_path or not os.path.exists(font_path):
                QMessageBox.warning(
                    self, 
                    "Font Error", 
                    f"Font file not found:\n{selected_font}\nPath: {font_path}"
                )
                font_path = None
                return  # Hentikan proses jika font tidak valid

        if not self.text_data:
            QMessageBox.critical(self, "Error", "Load text file first!")
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
            # Get font properties
            selected_font = self.font_choice.currentText()
            font_path = None
            
            if selected_font != "Default":
                try:
                    font_path = self.font_map.get(selected_font)
                    if not font_path or not os.path.exists(font_path):
                        raise FileNotFoundError(f"Font file not found: {font_path}")
                except Exception as e:
                    QMessageBox.warning(self, "Font Error", 
                                    f"Invalid font selection: {str(e)}")
                    font_path = None
                    
            if self.current_figure:
                plt.close(self.current_figure)

            self.wordcloud_progress_bar.setVisible(True)
            QApplication.processEvents()

            # ✅ Buat WordCloud
            wc = WordCloud(
                width=800, height=400,
                background_color=self.bg_color.currentText(),
                stopwords=stopwords,
                colormap=self.color_theme.currentText(),
                max_words=self.max_words_entry.value(),
                min_font_size=self.min_font_size_entry.value(),
                mask=mask,
                font_path=font_path
            ).generate(self.text_data)

            plt.ion()
            self.current_figure = plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")

            # ✅ Tambahkan Judul jika Ada
            title_text = self.title_entry.text().strip()

            if title_text:
                # Ambil path font yang dipilih
                selected_font_name = self.font_choice.currentText()
                if selected_font_name != "Default":
                    font_path = self.font_map.get(selected_font_name)
                    if font_path:
                        # Gunakan FontProperties dengan path font
                        from matplotlib.font_manager import FontProperties
                        title_font = FontProperties(fname=font_path)
                        title_font.set_size(self.title_font_size.value())
                    else:
                        title_font = None
                else:
                    title_font = None

                plt.title(
                    title_text,
                    loc=self.title_position.currentText().lower(),
                    fontproperties=title_font
                )

            plt.axis("off")  # Selalu hilangkan border
            plt.show()

            self.wordcloud_progress_bar.setVisible(False)
        except Exception as e:
            self.wordcloud_progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to generate word cloud: {e}")

    def simpan_wordcloud(self):             # ✅ Fungsi untuk menyimpan wordcloud
        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(self, "Save WordCloud", "", "PNG file (*.png);;JPG file (*.jpg)", options=options)
        if not save_path:
            return
        stopwords = self.ambil_stopwords()
        mask = None

        font_path = None
        if self.font_choice.currentText() != "Default":
            from matplotlib import font_manager

            selected_font = self.font_choice.currentText()
            font_path = self.font_map.get(selected_font)

            try:
                # Validate font exists
                font_manager.findfont(self.font_choice.currentText())
                font_path = self.font_choice.currentText()
            except:
                QMessageBox.warning(self, "Font Error", "Selected font not found, using default")

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

    # ✅ Fungsi untuk membuat palet warna kustom
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

    # ✅ Fungsi untuk menyimpan palet warna kustom
    def save_custom_palette(self, color_list, dialog):              
        palette_name, ok = QInputDialog.getText(self, "Save Palette", "Enter palette name:")
        if ok and palette_name:
            self.custom_color_palettes[palette_name] = color_list
            self.color_theme.addItem(palette_name)
            dialog.accept()

    # ✅ Fungsi untuk memilih warna latar belakang kustom
    def select_custom_bg_color(self):           
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color.addItem(color.name())
            self.bg_color.setCurrentText(color.name())

    # ✅ Fungsi untuk melihat teks penuh
    def view_full_text(self):           
        if not self.text_data:
            QMessageBox.critical(self, "Error", "No text data available to display.")
            return

        # Tutup dialog sebelumnya jika masih terbuka
        if hasattr(self, "text_dialog") and self.text_dialog is not None:
            self.text_dialog.close()

        self.text_dialog = QDialog(self)
        self.text_dialog.setWindowTitle("Full Text")
        layout = QVBoxLayout()

        text_edit = QTextEdit()
        text_edit.setPlainText(self.text_data)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.text_dialog.accept)
        layout.addWidget(close_button)

        self.text_dialog.setLayout(layout)
        self.text_dialog.show()

    # ✅ Stop progressbar dan buka jendela warning
    def handle_offline_warning(self, msg):      
        self.progress_bar.setVisible(False) 
        QMessageBox.warning(self, "Offline Mode", msg) 

    def on_sentiment_analyzed(self, result):
        self.progress_bar.setVisible(False)
        text_length = len(self.text_data)
        word_count = len(self.text_data.split())
        char_count_excl_spaces = len(self.text_data.replace(" ", ""))
        avg_word_length = char_count_excl_spaces / word_count if word_count > 0 else 0
        most_frequent_words = self.get_most_frequent_words(self.text_data, 5)

        self.show_sentiment_analysis(
            result["positive_score"], result["neutral_score"], result["negative_score"],
            result["compound_score"], result["sentiment_label"], text_length, result["subjectivity"],
            word_count, char_count_excl_spaces, avg_word_length, most_frequent_words
        )

    def show_sentiment_analysis(self, positive_score, neutral_score, negative_score, compound_score, sentiment_label, text_length, subjectivity, word_count, char_count_excl_spaces, avg_word_length, most_frequent_words):
        dialog = QDialog(self)
        dialog.setWindowTitle("Sentiment Analysis")
        dialog.setMinimumSize(300, 200)
        dialog.setSizeGripEnabled(True)

        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setPlainText(
            f"Sentiment Label: {sentiment_label}\n"
            f"Positive Sentiment: {positive_score:.2f}\n"
            f"Neutral Sentiment: {neutral_score:.2f}\n"
            f"Negative Sentiment: {negative_score:.2f}\n"
            f"Compound Score: {compound_score:.2f}\n"
            f"Text Length: {text_length} characters\n"
            f"Word Count: {word_count}\n"
            f"Character Count (excluding spaces): {char_count_excl_spaces}\n"
            f"Average Word Length: {avg_word_length:.2f}\n"
            f"Most Frequent Words: {', '.join(most_frequent_words)}\n"
            f"Subjectivity: {subjectivity}"
        )
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.show()

    def get_most_frequent_words(self, text, n):
        stopwords = self.ambil_stopwords().union(STOPWORDS)
        words = [word.lower() for word in text.split() if word.lower() not in stopwords]
        word_counts = Counter(words)
        most_common = word_counts.most_common(n)
        return [word for word, count in most_common]

    def change_sentiment_mode(self, mode):
        self.sentiment_mode = mode

        if "Flair" in mode:
            self.sentiment_button.setEnabled(False)

        # Handle VADER controls
        if mode == "VADER (Custom Lexicon)":
            self.custom_lexicon_button.setEnabled(True)
            if not self.custom_lexicon_path:
                QMessageBox.warning(self, "Lexicon Required", 
                                    "You have selected 'VADER (Custom Lexicon)'. "
                                    "Please load a custom lexicon before analyzing sentiment.")
        else:
            self.custom_lexicon_button.setEnabled(False)

        if mode in ["TextBlob", "TextBlob (Non-English)", "VADER"]:
            self.sentiment_button.setEnabled(True)
            self.custom_model_button.setEnabled(False)    

        # Handle Flair controls
        if mode == "Flair (Custom Model)":
            if self.flair_classifier_cuslang:
                self.custom_model_button.setEnabled(True)
                self.sentiment_button.setEnabled(True)
            else:
                self.custom_model_button.setEnabled(False)
                self.load_flair_model()
        
        # Preload default Flair model for any Flair mode
        elif mode == "Flair":
            if not self.flair_classifier:
                self.load_flair_model()

    def load_flair_model(self):
        # ✅ Ensure flair_loader_thread exists before calling isRunning()
        if hasattr(self, 'flair_loader_thread') and self.flair_loader_thread and self.flair_loader_thread.isRunning():
            self.flair_loader_thread.quit()
            self.flair_loader_thread.wait()

        # ✅ Show progress bar before loading starts
        self.model_progress_bar.setVisible(True)
        self.model_progress_bar.setRange(0, 0)  # Indeterminate progress
        QApplication.processEvents()  # ✅ Forces UI update before loading begins

        # ✅ Start Flair model loading in background thread
        self.flair_loader_thread = FlairModelLoaderThread()
        self.flair_loader_thread.model_loaded.connect(self.on_flair_model_loaded)
        self.flair_loader_thread.error_occurred.connect(self.on_flair_model_error)
        self.flair_loader_thread.finished.connect(self.cleanup_flair_thread)
        self.flair_loader_thread.start()

    def on_flair_model_loaded(self, model):
        if model:
            self.flair_classifier = model
            self.model_progress_bar.setVisible(False)

            if self.flair_loader_thread is not None:
                self.flair_loader_thread.quit()
                self.flair_loader_thread.wait()
                self.flair_loader_thread = None

            # ✅ Jika mode Flair biasa, aktifkan tombol jika ada teks valid
            if self.sentiment_mode == "Flair" and self.text_data and self.text_data.strip():
                self.sentiment_button.setEnabled(True)

            # ✅ Jika mode Flair (Custom Model), aktifkan hanya jika ada teks & model kustom sudah dimuat
            elif self.sentiment_mode == "Flair (Custom Model)" and self.text_data and self.text_data.strip() and self.flair_classifier_cuslang:
                self.sentiment_button.setEnabled(True)

            QMessageBox.information(self, "Library Ready", "Flair model loaded successfully!")

            # ✅ Pastikan pesan hanya muncul jika mode "Flair (Custom Model)" dipilih & model belum dimuat
            if self.sentiment_mode == "Flair (Custom Model)" and not self.custom_model_button.isEnabled():
                self.custom_model_button.setEnabled(True)
                QMessageBox.warning(self, "Custom Model Required", "You have selected Flair (Custom Model). "
                                        "Please load your custom model before analyzing sentiment.")
        else:
            QMessageBox.critical(self, "Error", "Flair model failed to load. Please try again.")








    def on_flair_model_error(self, error):
        self.model_progress_bar.setVisible(False)
        QMessageBox.critical(self, "Loading Error", f"Failed to load Flair model: {error}")

    def load_custom_lexicon(self):
        options = QFileDialog.Options()
        lexicon_path, _ = QFileDialog.getOpenFileName(self, "Select Custom Lexicon File", "", "Text Files (*.txt)", options=options)
        if lexicon_path:
            self.model_progress_bar.setVisible(True)
            self.model_progress_bar.setRange(0, 0)
            self.custom_lexicon_button.setEnabled(False)

            self.lexicon_loader_thread = CustomFileLoaderThread(lexicon_path, 'lexicon')
            self.lexicon_loader_thread.file_loaded.connect(self.on_lexicon_loaded)
            self.active_threads.append(self.lexicon_loader_thread) 
            self.lexicon_loader_thread.start()

    def load_custom_model(self):
        options = QFileDialog.Options()
        model_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Custom Model File", 
            "", 
            "Model Files (*.pt)", 
            options=options
        )
        if model_path:
            self.model_progress_bar.setVisible(True)
            self.custom_model_button.setEnabled(False)
            
            self.model_loader_thread = CustomFileLoaderThread(model_path, 'model')
            self.model_loader_thread.file_loaded.connect(self.on_model_loaded)
            self.active_threads.append(self.model_loader_thread)  
            self.model_loader_thread.start()

    def on_lexicon_loaded(self, result, success):
        self.model_progress_bar.setVisible(False)
        self.custom_lexicon_button.setEnabled(True)

        if success:
            self.custom_lexicon_path = result
            self.vader_analyzer = CustomVaderSentimentIntensityAnalyzer(custom_lexicon_file=self.custom_lexicon_path) 
            QMessageBox.information(self, "Success", "Custom lexicon loaded successfully! VADER will now use this lexicon.")
        else:
            QMessageBox.critical(self, "Error", f"Failed to load custom lexicon: {result}")


    def on_model_loaded(self, result, success):
        self.model_progress_bar.setVisible(False)
        self.custom_model_button.setEnabled(True)

        if success:
            try:
                from flair.models import TextClassifier  # Local import
                from flair.data import Sentence

                if not isinstance(result, TextClassifier):
                    QMessageBox.critical(self, "Error", 
                                        "Invalid model type. Please load a valid Flair TextClassifier model.")
                    return
                    
                # Test the loaded model
                test_sentence = Sentence("This is a test sentence")
                result.predict(test_sentence)
                if not test_sentence.labels:
                    raise ValueError("Model didn't produce any labels")
                
                self.flair_classifier_cuslang = result
                QMessageBox.information(self, "Success", 
                                    "Custom model loaded successfully!")
                    
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Model Test Failed", 
                    f"Model validation failed: {str(e)}\nPlease ensure this is a valid sentiment analysis model."
                )
        else:
            QMessageBox.critical(self, "Error", f"Failed to load custom model: {result}")

    def cleanup_flair_thread(self):
        """Ensure the Flair model thread is properly cleaned up without premature deletion."""
        QTimer.singleShot(100, lambda: setattr(self, 'flair_loader_thread', None))  # ✅ Delayed cleanup to avoid conflicts

    def show_sentiment_mode_info(self):
        description = (
            "TextBlob: Suitable for formal text and classic NLP analysis, such as articles, reports, and long documents.\n\n"
            "VADER: Designed for informal text, such as social media, comments, tweets, and short reviews. Fast and effective for rule-based sentiment analysis.\n\n"
            "Flair: Ideal for long texts and advanced sentiment analysis, such as product reviews, professional documents, or research-based deep learning."
        )

        dialog = QDialog(self, Qt.Window)
        dialog.setWindowTitle("Sentiment Analysis Mode Descriptions")
        dialog.setMinimumSize(300, 200)
        dialog.setSizeGripEnabled(True)

        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setPlainText(description)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.show()

def is_connected():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WordCloudGenerator()
    ex.show()
    sys.exit(app.exec())