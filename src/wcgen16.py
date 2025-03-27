# Standard library imports
import sys
import os
import re
from pathlib import Path
from functools import lru_cache, cached_property
from contextlib import contextmanager
import asyncio
from collections import Counter
import socket
import base64
import importlib
import hashlib
import tempfile
import shutil

# Third-party imports
import torch
import psutil
os.environ["QT_API"] = "pyside6"

from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import yake
import rake_nltk
from qasync import QEventLoop
from deep_translator import GoogleTranslator
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel, QPushButton,
    QVBoxLayout, QGridLayout, QWidget, QLineEdit, QComboBox, QSpinBox, QDialog,
    QTextEdit, QProgressBar, QFrame, QColorDialog, QInputDialog, QTextBrowser,
    QHBoxLayout, QGroupBox
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QMutex, QThreadPool, QObject
from PySide6.QtGui import QIcon, QGuiApplication
import matplotlib
matplotlib.use("QtAgg")
import numpy as np
from PIL import Image
from matplotlib.font_manager import FontProperties

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

os.environ["QT_SCALE_FACTOR"] = "0.75"

def setup_dll_path():
    if getattr(sys, 'frozen', False):
        # Jika running sebagai executable
        application_path = os.path.dirname(sys.executable)
        dll_path = os.path.join(application_path, 'lib')
        # Tambahkan path DLL ke PATH environment
        if dll_path not in os.environ['PATH']:
            os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']
        
setup_dll_path()

# Constants
if getattr(sys, 'frozen', False):
    APP_DIR = Path(sys.executable).parent
else:
    APP_DIR = Path(__file__).parent

ICON_PATH = APP_DIR / "icon.ico"

ABOUT_TEXT = "PGgyPvCfjJ8gV0NHZW4gLSBXb3JkIENsb3VkIEdlbmVyYXRvciArIFNlbnRpbWVudCBBbmFseXNpczwvaDI+CjxwPjxiPlZlcnNpb246PC9iPiAxLjU8L3A+CjxwPiZjb3B5OyAyMDI1IE0uIEFkaWIgWmF0YSBJbG1hbTwvcD4KPHA+PGEgaHJlZj0iaHR0cHM6Ly9naXRodWIuY29tL3phdGFpbG0vd2Nsb3VkZ3VpIj7wn5OMIEdpdEh1YiBSZXBvc2l0b3J5PC9hPjwvcD4KPGgzPvCflI0gV2hhdCBpcyBXQ0dlbj88L2gzPgo8cD5XQ0dlbiBpcyBhbiBhZHZhbmNlZCB5ZXQgZWFzeS10by11c2UgYXBwbGljYXRpb24gZGVzaWduZWQgZm9yIGNyZWF0aW5nIHZpc3VhbGx5IGFwcGVhbGluZyB3b3JkIGNsb3VkcyBmcm9tIHRleHQgZGF0YS48L3A+CjxoMz7wn5qAIEtleSBGZWF0dXJlczwvaDM+Cjx1bD4KICAgIDxsaT48Yj5TdXBwb3J0cyBtdWx0aXBsZSBmaWxlIGZvcm1hdHM6PC9iPiBUWFQsIFBERiwgRE9DL0RPQ1gsIENTViwgWExTWDwvbGk+CiAgICA8bGk+PGI+RnVsbHkgY3VzdG9taXphYmxlIHdvcmQgY2xvdWRzOjwvYj4gQ2hvb3NlIGNvbG9ycywgZm9udHMsIHNoYXBlcywgYW5kIHRoZW1lczwvbGk+CiAgICA8bGk+PGI+U3RvcHdvcmQgZmlsdGVyaW5nOjwvYj4gUmVtb3ZlIGNvbW1vbiB3b3JkcyBmb3IgY2xlYXJlciBpbnNpZ2h0czwvbGk+CiAgICA8bGk+PGI+U21hcnQgdGV4dCBwcm9jZXNzaW5nOjwvYj4gSGFuZGxlcyBsYXJnZSBkYXRhc2V0cyBlZmZpY2llbnRseTwvbGk+CiAgICA8bGk+PGI+RXhwb3J0ICYgc2F2ZSBvcHRpb25zOjwvYj4gSGlnaC1yZXNvbHV0aW9uIGltYWdlIHNhdmluZyBpbiBtdWx0aXBsZSBmb3JtYXRzPC9saT4KICAgIDxsaT48Yj5TZW50aW1lbnQgYW5hbHlzaXM6PC9iPiBPcHRpb25hbCBhbmFseXNpcyB1c2luZyBUZXh0QmxvYiwgVkFERVIsIGFuZCBGbGFpcjwvbGk+CjwvdWw+CjxoMz7wn5OWIEhvdyBXQ0dlbiBIZWxwcyBZb3U8L2gzPgo8cD5XaGV0aGVyIHlvdeKAmXJlIGFuYWx5emluZyBjdXN0b21lciBmZWVkYmFjaywgY29uZHVjdGluZyBhY2FkZW1pYyByZXNlYXJjaCwgb3IgdmlzdWFsaXppbmcgdGV4dC1iYXNlZCBpbnNpZ2h0czwgV0NHZW4gc2ltcGxpZmllcyB0aGUgcHJvY2VzcyBhbmQgZW5oYW5jZXMgeW91ciB3b3JrZmxvdyB3aXRoIGl0cyBpbnR1aXRpdmUgZGVzaWduIGFuZCBwb3dlcmZ1bCBmZWF0dXJlcy48L3A+CjxoMz7wn5OcIExpY2Vuc2U8L2gzPgo8cD5XQ0dlbiBpcyBmcmVlIGZvciBwZXJzb25hbCBhbmQgZWR1Y2F0aW9uYWwgdXNlLiBGb3IgY29tbWVyY2lhbCBhcHBsaWNhdGlvbnMsIHBsZWFzZSByZWZlciB0byB0aGUgbGljZW5zaW5nIHRlcm1zLjwvcD4KPGgzPvCfk5ogSG93IHRvIENpdGUgV0NHZW48L2gzPgo8cD5JZiB5b3UgdXNlIFdDR2VuIGluIHlvdXIgcmVzZWFyY2ggb3IgcHVibGljYXRpb24sIHBsZWFzZSBjaXRlIGl0IGFzIGZvbGxvd3MgKEFQQSA3KTo8L3A+CjxwPklsbWFtLCBNLiBBLiBBLiAoMjAyNSkuIDxpPldDR2VuIC0gV29yZCBDbG91ZCBHZW5lcmF0b3IgKyBTZW50aW1lbnQgQW5hbHlzaXM8L2k+IChWZXJzaW9uIDEuNSkgW1NvZnR3YXJlXS4gWmVub2RvLiA8YSBocmVmPSJodHRwczovL2RvaS5vcmcvMTAuNTI4MS96ZW5vZG8uMTUwMzQ4NDMiPmh0dHBzOi8vZG9pLm9yZy8xMC41MjgxL3plbm9kby4xNDkzMjY1MDwvYT48L3A+Cg=="

MODE_INFO = "PGgyPlNlbnRpbWVudCBBbmFseXNpcyBNb2RlczwvaDI+CjxwPlNlbGVjdCB0aGUgbW9zdCBzdWl0YWJsZSBzZW50aW1lbnQgYW5hbHlzaXMgbWV0aG9kIGJhc2VkIG9uIHlvdXIgdGV4dCB0eXBlIGFuZCBhbmFseXNpcyBuZWVkcy48L3A+CjxoMz7wn5OdIFRleHRCbG9iPC9oMz4KPHA+PGI+QmVzdCBmb3I6PC9iPiBGb3JtYWwgdGV4dHMsIHdlbGwtc3RydWN0dXJlZCBkb2N1bWVudHMsIG5ld3MgYXJ0aWNsZXMsIHJlc2VhcmNoIHBhcGVycywgYW5kIHJlcG9ydHMuPC9wPgo8cD5UZXh0QmxvYiBpcyBhIGxleGljb24tYmFzZWQgc2VudGltZW50IGFuYWx5c2lzIHRvb2wgdGhhdCBwcm92aWRlcyBhIHNpbXBsZSB5ZXQgZWZmZWN0aXZlIGFwcHJvYWNoIGZvciBldmFsdWF0aW5nIHRoZSBzZW50aW1lbnQgb2Ygc3RydWN0dXJlZCB0ZXh0LiBJdCBhc3NpZ25zIGEgcG9sYXJpdHkgc2NvcmUgKHBvc2l0aXZlLCBuZWdhdGl2ZSwgb3IgbmV1dHJhbCkgYW5kIGNhbiBhbHNvIGFuYWx5emUgc3ViamVjdGl2aXR5IGxldmVscy48L3A+CjxoMz7wn5KsIFZBREVSIChWYWxlbmNlIEF3YXJlIERpY3Rpb25hcnkgYW5kIHNFbnRpbWVudCBSZWFzb25lcik8L2gzPgo8cD48Yj5CZXN0IGZvcjo8L2I+IFNvY2lhbCBtZWRpYSBwb3N0cywgdHdlZXRzLCBzaG9ydCBjb21tZW50cywgY2hhdCBtZXNzYWdlcywgYW5kIGluZm9ybWFsIHJldmlld3MuPC9wPgo8cD5WQURFUiBpcyBzcGVjaWZpY2FsbHkgZGVzaWduZWQgZm9yIGFuYWx5emluZyBzaG9ydCwgaW5mb3JtYWwgdGV4dHMgdGhhdCBvZnRlbiBjb250YWluIHNsYW5nLCBlbW9qaXMsIGFuZCBwdW5jdHVhdGlvbi1iYXNlZCBlbW90aW9ucy4gSXQgaXMgYSBydWxlLWJhc2VkIHNlbnRpbWVudCBhbmFseXNpcyBtb2RlbCB0aGF0IGVmZmljaWVudGx5IGRldGVybWluZXMgc2VudGltZW50IGludGVuc2l0eSBhbmQgd29ya3MgZXhjZXB0aW9uYWxseSB3ZWxsIGZvciByZWFsLXRpbWUgYXBwbGljYXRpb25zLjwvcD4KPGgzPvCflKwgRmxhaXI8L2gzPgo8cD48Yj5CZXN0IGZvcjo8L2I+IExvbmctZm9ybSBjb250ZW50LCBwcm9kdWN0IHJldmlld3MsIHByb2Zlc3Npb25hbCBkb2N1bWVudHMsIGFuZCBBSS1iYXNlZCBkZWVwIHNlbnRpbWVudCBhbmFseXNpcy48L3A+CjxwPkZsYWlyIHV0aWxpemVzIGRlZXAgbGVhcm5pbmcgdGVjaG5pcXVlcyBmb3Igc2VudGltZW50IGFuYWx5c2lzLCBtYWtpbmcgaXQgaGlnaGx5IGFjY3VyYXRlIGZvciBjb21wbGV4IHRleHRzLiBJdCBpcyBpZGVhbCBmb3IgYW5hbHl6aW5nIGxhcmdlLXNjYWxlIHRleHR1YWwgZGF0YSwgY2FwdHVyaW5nIGNvbnRleHQgbW9yZSBlZmZlY3RpdmVseSB0aGFuIHRyYWRpdGlvbmFsIHJ1bGUtYmFzZWQgbW9kZWxzLiBIb3dldmVyLCBpdCByZXF1aXJlcyBtb3JlIGNvbXB1dGF0aW9uYWwgcmVzb3VyY2VzIGNvbXBhcmVkIHRvIFRleHRCbG9iIGFuZCBWQURFUi48L3A+CjxoMz7wn4yQIEltcG9ydGFudCBOb3RlIGZvciBMYW5ndWFnZSBTdXBwb3J0PC9oMz4KPHA+V2hpbGUgdGhpcyBhcHBsaWNhdGlvbiBzdXBwb3J0cyBub24tRW5nbGlzaCB0ZXh0IHRocm91Z2ggYXV0b21hdGljIHRyYW5zbGF0aW9uLCBpdCBpcyA8Yj5oaWdobHkgcmVjb21tZW5kZWQ8L2I+IHRvIHVzZSA8Yj5tYW51YWxseSB0cmFuc2xhdGVkIGFuZCByZWZpbmVkIEVuZ2xpc2ggdGV4dDwvYj4gZm9yIHRoZSBtb3N0IGFjY3VyYXRlIHNlbnRpbWVudCBhbmFseXNpcy4gVGhlIGJ1aWx0LWluIGF1dG9tYXRpYyB0cmFuc2xhdGlvbiBmZWF0dXJlIG1heSBub3QgYWx3YXlzIGZ1bmN0aW9uIGNvcnJlY3RseSwgbGVhZGluZyB0byBwb3RlbnRpYWwgbWlzaW50ZXJwcmV0YXRpb25zIG9yIGluYWNjdXJhdGUgc2VudGltZW50IHJlc3VsdHMuPC9wPgo8cD5Gb3IgdGhlIGJlc3QgcGVyZm9ybWFuY2UsIGVuc3VyZSB0aGF0IG5vbi1FbmdsaXNoIHRleHQgaXMgcHJvcGVybHkgcmV2aWV3ZWQgYW5kIGFkanVzdGVkIGJlZm9yZSBzZW50aW1lbnQgYW5hbHlzaXMuIPCfmoA8L3A+CjxoMz7wn5OMIEN1c3RvbSBMZXhpY29uIEZvcm1hdCBFeGFtcGxlPC9oMz4KPHA+QmVsb3cgaXMgYW4gZXhhbXBsZSBvZiBhIGN1c3RvbSBsZXhpY29uIGZvcm1hdCBmb3Igc2VudGltZW50IGFuYWx5c2lzOjwvcD4KPHByZSBzdHlsZT0nYmFja2dyb3VuZC1jb2xvcjojZjRmNGY0OyBwYWRkaW5nOjEwcHg7IGJvcmRlci1yYWRpdXM6NXB4Oyc+CmV4Y2VsbGVudCAgIDEuNQphd2Z1bCAgICAgIC0xLjUKbm90ICAgICAgICBuZWdhdGlvbiAgICAgICAgICMgTWFyayBhcyBuZWdhdGlvbiB3b3JkCmludGVuc2VseSAgaW50ZW5zaWZpZXI6MS43ICAjIEN1c3RvbSBpbnRlbnNpZmllciB3aXRoIG11bHRpcGxpZXIKPC9wcmU+CjxwPlRoaXMgY3VzdG9tIGxleGljb24gYWxsb3dzIGZpbmUtdHVuaW5nIG9mIHNlbnRpbWVudCBzY29yZXMgYnkgYWRkaW5nIGN1c3RvbSB3b3JkcywgbmVnYXRpb25zLCBhbmQgaW50ZW5zaWZpZXJzIHRvIGltcHJvdmUgc2VudGltZW50IGFuYWx5c2lzIGFjY3VyYWN5LjwvcD4K"

SCORE_TYPES = {
    "VADER": "Compound Score",
    "VADER (Custom Lexicon)": "Compound Score",
    "TextBlob": "Polarity Score", 
    "TextBlob (Custom Lexicon)": "Polarity Score",
    "Flair": "Confidence Score",
    "Flair (Custom Model)": "Confidence Score"
}

SENTIMENT_MODES = list(SCORE_TYPES.keys())

THREAD_TIMEOUT = {
    "SOFT": 1000,
    "FORCE": 500,
    "TERMINATION": 2000
}

# Utility Functions
def sanitize_path(path):
    path = os.path.normpath(path)
    base_dir = os.path.abspath(path)
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(base_dir):
        raise ValueError("Path traversal attempt detected")
    return path

@lru_cache(maxsize=128)
def load_stopwords():
    if getattr(sys, 'frozen', False):
        stopwords_path = os.path.join(os.path.dirname(sys.executable), "lib", "wordcloud", "stopwords")
    else:
        from wordcloud import STOPWORDS
        return set(STOPWORDS)
        
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return set(word.strip() for word in f)
    except Exception as e:
        print(f"Warning: Could not load stopwords from {stopwords_path}: {e}")
        return set()

def is_connected():
    """Check if there is internet connection"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

# Main Application Class
class MainClass(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.perf_monitor = PerformanceMonitor()
        
        self.active_threads = ThreadSafeSet()
        self.resource_manager = ResourceManager()
        self.lazy_loader = LazyLoader()
        
        self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.lda_model = None
        self.nmf_model = None
        self.bert_model = None
        
        self._cached_models = {}
        self._cached_fonts = {}
        self._cached_colormaps = {}
        
        self._init_basic()
        self._setup_lazy_loading()
        
        QThreadPool.globalInstance().setMaxThreadCount(3)
        
        self.text_data = ""
        self.additional_stopwords = set()
        self.custom_color_palettes = {}
        self.flair_classifier = None
        self.flair_classifier_cuslang = None
        self.flair_first_load = True
        self.mask_path = ""
        self.current_figure = None
        self.custom_lexicon_path = None
        self.custom_textblob_lexicon_path = None
        self.textblob_analyzer = None
        self.flair_loader_thread = None
        
        self.import_thread = None
        self.active_threads = []
        self.threads_mutex = QMutex()
        
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.setup_thread_management()
        self.init_analyzers()
        
        self.progress_states = {
            'keywords': {'color': 'red', 'text': 'Extracting keywords'},
            'file': {'color': 'blue', 'text': 'Loading file'},
            'topics': {'color': 'yellow', 'text': 'Analyzing topics'},
            'model': {'color': 'green', 'text': 'Loading model'},
            'wordcloud': {'color': 'purple', 'text': 'Generating word cloud'}
        }
        
        self.initUI()
        self.setup_timers()
        self._init_imports()

        self.warning_emitter = WarningEmitter()
        self.warning_handler = CustomWarningHandler(self.warning_emitter)
        self.warning_emitter.token_warning.connect(self.show_token_warning)

        import warnings
        warnings.showwarning = self.warning_handler.handle_warning        

        self.button_manager = ButtonStateManager(self)

    @cached_property 
    def token_opts(self):
        """Cached token configuration for vectorizers"""
        return {
            'token_pattern': r'(?u)\b[a-zA-Z][a-zA-Z0-9\'-]*[a-zA-Z0-9]\b',
            'lowercase': True,
            'strip_accents': 'unicode',
            'max_features': 1000,
            'min_df': 1,
            'max_df': 0.95
        }

    def _init_basic(self):
        """Initialize basic components"""
        self.text_data = ""
        self.additional_stopwords = set()
        self.custom_color_palettes = {}
        self.setup_thread_management()
        
    def _setup_lazy_loading(self):
        """Configure lazy loading for heavy dependencies"""
        self._vader = None
        self._flair = None
        self._textblob = None
        
    @cached_property
    def vader_analyzer(self):
        """Lazy load VADER analyzer"""
        if not self._vader:
            VaderAnalyzer = self.lazy_loader.load('vaderSentiment.vaderSentiment', 'SentimentIntensityAnalyzer')
            if VaderAnalyzer:
                self._vader = VaderAnalyzer()
        return self._vader

    @lru_cache(maxsize=32)
    def get_wordcloud(self, **kwargs):
        """Get cached wordcloud instance"""
        return OptimizedWordCloud(**kwargs)

    def _init_imports(self):
        """Pre-initialize heavy imports in background with proper cleanup"""
        if self.import_thread is not None:
            return
            
        self.import_thread = ImportThread()
        self.import_thread.finished.connect(self._cleanup_import_thread)
        self.add_managed_thread(self.import_thread)
        self.import_thread.start()
        
    def _cleanup_import_thread(self):
        """Cleanup import thread when finished"""
        if self.import_thread:
            try:
                self.import_thread.stop()
                self.remove_managed_thread(self.import_thread)
                self.import_thread.deleteLater()
            finally:
                self.import_thread = None

    def add_managed_thread(self, thread):
        """Add thread to managed threads list"""
        self.threads_mutex.lock()
        try:
            self.active_threads.append(thread)
        finally:
            self.threads_mutex.unlock()

    def remove_managed_thread(self, thread):
        """Remove thread from managed threads list"""
        self.threads_mutex.lock()
        try:
            if thread in self.active_threads:
                self.active_threads.remove(thread)
        finally:
            self.threads_mutex.unlock()

    def setup_thread_management(self):
        """Initialize thread management"""
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(3)
        self.thread_manager = ThreadManager()
        self.threads_mutex = QMutex()
        
    def init_analyzers(self):
        """Initialize analyzers with caching"""
        from functools import lru_cache
        
        @lru_cache(maxsize=32)
        def get_vader_analyzer():
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
            
        self.vader_analyzer = get_vader_analyzer()
        self.sentiment_mode = "TextBlob"
        
    def set_progress(self, state_key, visible=True, progress=None):
        """Enhanced progress bar control"""
        if not hasattr(self, 'unified_progress_bar'):
            return
            
        state = self.progress_states.get(state_key)
        if not state:
            return

        if visible:
            if state_key == 'file':
                pass
            elif state_key == 'model':
                self.sentiment_button.setEnabled(False)
            elif state_key == 'wordcloud':
                self.generate_wordcloud_button.setEnabled(False)
            elif state_key == 'topics':
                self.topic_tab.analyze_topics_btn.setEnabled(False)
            elif state_key == 'keywords':
                self.topic_tab.extract_keywords_btn.setEnabled(False)
                
            self.unified_progress_bar.setVisible(True)
            self.set_progress_style(state['color'])
            
            if progress is not None:
                self.progress_timer.stop()
                self.unified_progress_bar.setValue(int(progress))
            else:
                if not self.progress_timer.isActive():
                    self.unified_progress_bar.setValue(0)
                    self.progress_timer.start()
        else:
            has_text = bool(self.text_data)
            if state_key == 'file':
                self.load_file_button.setEnabled(True)
            elif state_key == 'model':
                self.sentiment_button.setEnabled(has_text)
            elif state_key == 'wordcloud':
                self.generate_wordcloud_button.setEnabled(has_text)
            elif state_key == 'topics':
                self.topic_tab.analyze_topics_btn.setEnabled(has_text)
            elif state_key == 'keywords':
                self.topic_tab.extract_keywords_btn.setEnabled(has_text)
            
            self.progress_timer.stop()
            self.unified_progress_bar.setValue(0)
            self.unified_progress_bar.setVisible(False)
            
        QApplication.processEvents()

    def set_progress_style(self, color):
        """Set progress bar color and style for indeterminate animation"""
        chunk_color = {
            'red': 'rgba(215, 0, 0, 0.2)',
            'blue': 'rgba(0, 120, 215, 0.2)', 
            'yellow': 'rgba(255, 255, 0, 0.2)',
            'green': 'rgba(0, 215, 0, 0.2)',
            'purple': 'rgba(160, 32, 240, 0.2)'
        }.get(color, 'rgba(0, 120, 215, 0.2)')

        style = f"""
            QProgressBar {{
                border: none;
                background: transparent;
                padding: 0px;
                margin: 0px;
            }}
            
            QProgressBar::chunk {{
                width: 1px;
                background: {chunk_color};
            }}
        """
        self.unified_progress_bar.setStyleSheet(style)

    def setup_timers(self):
        """Setup and optimize timers"""
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.timeout.connect(self.thread_manager.cleanup_finished)
        self.cleanup_timer.start(10000)
        
    def initUI(self):
        WIN_TITLE = "V0NHZW4gKyBUZXh0IEFuYWx5dGljcyAodjEuNikK"
        win_title = base64.b64decode(WIN_TITLE.encode()).decode()
        self.setWindowTitle(win_title)
        self.setFixedSize(550, 850)

        layout = QVBoxLayout()

        file_group = QGroupBox("File Input")
        file_layout = QGridLayout()

        filename_container = QFrame()
        filename_container.setStyleSheet("""
            QFrame { 
                border: 1px solid #c0c0c0; 
                background: white;
                padding: 0px;
                margin: 0px;
            }
        """)
        filename_container.setFixedHeight(25)
        filename_layout = QHBoxLayout(filename_container)
        filename_layout.setContentsMargins(0, 0, 0, 0)
        filename_layout.setSpacing(0)

        text_container = QFrame()
        text_container.setLayout(QHBoxLayout())
        text_container.layout().setContentsMargins(5, 0, 5, 0)
        text_container.layout().setSpacing(0)
        
        self.file_name = QLineEdit()
        self.file_name.setReadOnly(True)
        self.file_name.setFrame(False)
        self.file_name.setPlaceholderText("No file selected")
        self.file_name.setStyleSheet("""
            QLineEdit {
                border: none;
                background: transparent;
                padding: 0px;
            }
        """)
        text_container.layout().addWidget(self.file_name)
        filename_layout.addWidget(text_container)

        self.unified_progress_bar = QProgressBar(filename_container)
        self.unified_progress_bar.setRange(0, 100)
        self.unified_progress_bar.setValue(0)
        self.unified_progress_bar.setTextVisible(False)
        self.unified_progress_bar.setMinimumWidth(filename_container.width())
        self.unified_progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background: transparent;
                padding: 0px;
                margin: 0px;
            }
            
            QProgressBar::chunk {
                width: 1px;
                background: rgba(0, 120, 215, 0.2);
            }
        """)
        
        self.unified_progress_bar.setFixedWidth(filename_container.width())
        self.unified_progress_bar.setFixedHeight(filename_container.height())
        
        self.unified_progress_bar.setGeometry(0, 0, filename_container.width(), filename_container.height())
        self.file_name.raise_()

        file_layout.addWidget(filename_container, 0, 0, 1, 6)

        self.load_file_button = QPushButton("Load Text File", self)
        self.load_file_button.clicked.connect(self.open_file)
        self.load_file_button.setToolTip(
            "Upload a text file for word cloud generation and sentiment analysis.\n"
            "Supports TXT, CSV, XLS/XLSX, PDF, DOC/DOCX.\n"
            "Ensure your text is well-formatted for better results."
        )
        file_layout.addWidget(self.load_file_button, 1, 0, 1, 2)

        self.view_fulltext_button = QPushButton("View Full Text", self)
        self.view_fulltext_button.clicked.connect(self.view_full_text)
        self.view_fulltext_button.setEnabled(False)
        self.view_fulltext_button.setToolTip(
            "Click to view the full text content in a separate window.\n"
            "Allows you to inspect the complete text before generating the word cloud.\n"
            "Useful for verifying text input and checking formatting."
        )
        file_layout.addWidget(self.view_fulltext_button, 1, 2, 1, 2)

        self.summarize_button = QPushButton("Summarize", self)
        self.summarize_button.clicked.connect(self.summarize_text)
        self.summarize_button.setEnabled(False)
        self.summarize_button.setToolTip(
            "Summarize the text content using the LSA (Latent Semantic Analysis) method.\n"
            "This extractive summarization technique selects the most important sentences,\n"
            "helping to generate concise word clouds and improve sentiment analysis."
        )
        file_layout.addWidget(self.summarize_button, 1, 4, 1, 2)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        wordcloud_group = QGroupBox("Word Cloud Generation")
        wordcloud_layout = QGridLayout()

        stopwords_container = QHBoxLayout()
        self.stopword_entry = QTextEdit(self)
        self.stopword_entry.setFixedHeight(50)
        self.stopword_entry.setPlaceholderText("Enter stopwords, separated by spaces or new lines (optional)")
        self.stopword_entry.setToolTip(
            "Enter stopwords (words to be excluded) separated by spaces or new lines.\n"
            "The words you enter will be ignored in the word cloud.\n"
            "Use custom stopwords to refine the visualization and focus on meaningful words."
        )        
        stopwords_container.addWidget(self.stopword_entry)     
        
        self.load_stopwords_btn = QPushButton("Load", self)
        self.load_stopwords_btn.setFixedSize(50, 50)
        self.load_stopwords_btn.clicked.connect(self.load_stopwords_file)
        stopwords_container.addWidget(self.load_stopwords_btn)
        
        wordcloud_layout.addLayout(stopwords_container, 0, 0, 1, 6)

        self.color_theme_label = QLabel("Color Theme:", self)
        wordcloud_layout.addWidget(self.color_theme_label, 1, 0, 1, 2)

        self.color_theme = QComboBox(self)
        QTimer.singleShot(100, self.load_colormaps)
        self.color_theme.setToolTip(
            "Choose a color palette for the word cloud.\n"
            "Darker themes work well with light backgrounds, and vice-versa."
        )        
        wordcloud_layout.addWidget(self.color_theme, 1, 2, 1, 3)

        self.custom_palette_button = QPushButton("Custom", self)
        self.custom_palette_button.clicked.connect(self.create_custom_palette)
        wordcloud_layout.addWidget(self.custom_palette_button, 1, 5, 1, 1)

        self.bg_color_label = QLabel("Background Color:", self)
        wordcloud_layout.addWidget(self.bg_color_label, 2, 0, 1, 2)

        self.bg_color = QComboBox(self)
        self.bg_color.addItems(["white", "black", "gray", "blue", "red", "yellow"])
        self.bg_color.setToolTip(
            "Select the background color for the word cloud.\n"
            "Use contrast for better visibility.\n"
            "White or black backgrounds usually work best."
        )        
        wordcloud_layout.addWidget(self.bg_color, 2, 2, 1, 3)

        self.custom_bg_color_button = QPushButton("Custom", self)
        self.custom_bg_color_button.clicked.connect(self.select_custom_bg_color)
        wordcloud_layout.addWidget(self.custom_bg_color_button, 2, 5, 1, 1)

        self.title_label = QLabel("WordCloud Title:", self)
        wordcloud_layout.addWidget(self.title_label, 3, 0, 1, 2)

        self.title_entry = QLineEdit(self)
        self.title_entry.setPlaceholderText("Enter title (optional)")
        self.title_entry.setToolTip(
            "Enter a title for your word cloud (optional).\n"
            "This title will be displayed above the word cloud.\n"
            "Leave blank if no title is needed."
        )        
        wordcloud_layout.addWidget(self.title_entry, 3, 2, 1, 2)

        self.title_font_size = QSpinBox(self)
        self.title_font_size.setRange(8, 72)
        self.title_font_size.setValue(14)
        self.title_font_size.setToolTip(
            "Set the font size for the word cloud title.\n"
            "Larger values make the title more prominent.\n"
            "Recommended: 14-24 px for a balanced look.\n"
            "Too large titles may overlap with the word cloud."
        )        
        wordcloud_layout.addWidget(self.title_font_size, 3, 4, 1, 1)

        self.title_position = QComboBox(self)
        self.title_position.addItems(["Left", "Center", "Right"])
        self.title_position.setCurrentText("Center")
        self.title_position.setToolTip(
            "Choose where to display the title relative to the word cloud.\n"
            "Positioning affects the overall layout and readability.\n"
            "Recomended: Center"
        )        
        wordcloud_layout.addWidget(self.title_position, 3, 5, 1, 1)

        self.font_choice_label = QLabel("Font Choice:", self)
        wordcloud_layout.addWidget(self.font_choice_label, 4, 0, 1, 2)

        self.font_choice = QComboBox(self)
        self.font_choice.addItem("Default")
        self.font_choice.setToolTip(
            "Choose a font style for the word cloud.\n"
            "Different fonts affect readability and aesthetics.\n"
            "Sans-serif fonts are recommended for clarity."
        )        
        wordcloud_layout.addWidget(self.font_choice, 4, 2, 1, 4)
        QTimer.singleShot(100, self.load_matplotlib_fonts)

        self.min_font_size_label = QLabel("Minimum Font Size:", self)
        wordcloud_layout.addWidget(self.min_font_size_label, 5, 0, 1, 2)

        self.min_font_size_input = QSpinBox(self)
        self.min_font_size_input.setValue(11)
        self.min_font_size_input.setToolTip(
            "Set the smallest font size for words in the word cloud.\n"
            "Prevents low-frequency words from becoming too small to read.\n"
            "Recommended value: 10-12 px for readability."
        )        
        wordcloud_layout.addWidget(self.min_font_size_input, 5, 2, 1, 1)

        self.max_words_label = QLabel("Maximum Words:", self)
        wordcloud_layout.addWidget(self.max_words_label, 5, 3, 1, 2, Qt.AlignRight)

        self.max_words_input = QSpinBox(self)
        self.max_words_input.setMaximum(10000)
        self.max_words_input.setValue(200)
        self.max_words_input.setToolTip(
            "Set the maximum number of words displayed in the word cloud.\n"
            "Higher values provide more detail but may reduce clarity.\n"
            "Recommended: 100-200 words for balanced visualization"
        )        
        wordcloud_layout.addWidget(self.max_words_input, 5, 5, 1, 1)

        self.mask_label = QLabel("Mask Image:", self)
        wordcloud_layout.addWidget(self.mask_label, 6, 0, 1, 2)

        self.mask_path_label = QLineEdit("default (rectangle)", self)
        self.mask_path_label.setReadOnly(True)
        wordcloud_layout.addWidget(self.mask_path_label, 6, 2, 1, 4)

        self.mask_button = QPushButton("Load Mask Image", self)
        self.mask_button.clicked.connect(self.choose_mask)
        self.mask_button.setToolTip(
            "Upload an image to shape the word cloud (PNG/JPG/BMP).\n"
            "White areas will be ignored, and words will fill the dark areas.\n"
            "Use simple shapes for best results."
        )        
        wordcloud_layout.addWidget(self.mask_button, 7, 2, 1, 2)

        self.reset_mask_button = QPushButton("Remove Mask Image", self)
        self.reset_mask_button.clicked.connect(self.reset_mask)
        self.reset_mask_button.setToolTip(
            "Remove the selected mask image and revert to the default shape.\n"
            "The word cloud will be displayed in a rectangular format.\n"
            "Use this if you no longer want a custom shape for the word cloud."
        )        
        wordcloud_layout.addWidget(self.reset_mask_button, 7, 4, 1, 2)

        self.generate_wordcloud_button = QPushButton("Generate Word Cloud", self)
        self.generate_wordcloud_button.clicked.connect(self.generate_wordcloud)
        self.generate_wordcloud_button.setEnabled(False)
        wordcloud_layout.addWidget(self.generate_wordcloud_button, 8, 0, 1, 6)

        self.text_stats_button = QPushButton("View Text Statistics", self)
        self.text_stats_button.clicked.connect(self.text_analysis_report)
        self.text_stats_button.setEnabled(False)
        wordcloud_layout.addWidget(self.text_stats_button, 9, 0, 1, 3)

        self.save_wc_button = QPushButton("Save Word Cloud", self)
        self.save_wc_button.clicked.connect(self.save_wordcloud)
        self.save_wc_button.setEnabled(False)
        wordcloud_layout.addWidget(self.save_wc_button, 9, 3, 1, 3)

        wordcloud_group.setLayout(wordcloud_layout)
        layout.addWidget(wordcloud_group)

        sentiment_group = QGroupBox("Sentiment Analysis")
        sentiment_layout = QGridLayout()

        self.sentiment_mode_label = QLabel("Analysis Mode:", self)
        sentiment_layout.addWidget(self.sentiment_mode_label, 0, 0, 1, 2)

        self.sentiment_mode_combo = QComboBox(self)
        self.sentiment_mode_combo.addItems([
            "TextBlob", "TextBlob (Custom Lexicon)", "VADER", "VADER (Custom Lexicon)", 
            "Flair", "Flair (Custom Model)"
        ])
        self.sentiment_mode_combo.setToolTip(
            "TextBlob (Best for formal texts like news articles and reports.)\n"
            "VADER (Best for social media, tweets, and informal texts with slang/emojis.)\n"
            "Flair (Best for long-form content and complex sentiment analysis using deep learning.)"
        )
        self.sentiment_mode_combo.currentTextChanged.connect(self.change_sentiment_mode)
        sentiment_layout.addWidget(self.sentiment_mode_combo, 0, 2, 1, 3)

        self.sentiment_mode_info_button = QPushButton("Info", self)
        self.sentiment_mode_info_button.clicked.connect(self.show_sentiment_mode_info)
        sentiment_layout.addWidget(self.sentiment_mode_info_button, 0, 5, 1, 1)

        self.custom_lexicon_button = QPushButton("Load Lexicon", self)
        self.custom_lexicon_button.clicked.connect(self.load_custom_lexicon)
        self.custom_lexicon_button.setEnabled(False)
        sentiment_layout.addWidget(self.custom_lexicon_button, 1, 0, 1, 3)

        self.custom_model_button = QPushButton("Load Model", self)
        self.custom_model_button.clicked.connect(self.load_custom_model)
        self.custom_model_button.setEnabled(False)
        sentiment_layout.addWidget(self.custom_model_button, 1, 3, 1, 3)

        self.sentiment_button = QPushButton("Analyze Sentiment", self)
        self.sentiment_button.clicked.connect(self.analyze_sentiment)
        self.sentiment_button.setEnabled(False)
        sentiment_layout.addWidget(self.sentiment_button, 2, 0, 1, 6)

        sentiment_group.setLayout(sentiment_layout)
        layout.addWidget(sentiment_group)

        self.topic_tab = TopicAnalysisTab(parent=self)
        layout.addWidget(self.topic_tab)

        bottom_group = QGroupBox()
        bottom_layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        
        self.about_button = QPushButton("About", self)
        self.about_button.clicked.connect(self.show_about)
        button_layout.addWidget(self.about_button)

        self.panic_button = QPushButton("STOP", self)
        self.panic_button.clicked.connect(self.stop_all_processes)
        button_layout.addWidget(self.panic_button)

        self.quit_button = QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close)
        button_layout.addWidget(self.quit_button)

        bottom_layout.addLayout(button_layout)

        bottom_group.setLayout(bottom_layout)
        layout.addWidget(bottom_group)

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self._update_progress_animation)
        self.progress_timer.setInterval(20)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_matplotlib_fonts(self):
        try:
            from matplotlib import font_manager
            self.font_map = {}

            weight_conversion = {
                100: "Thin", 200: "Extra Light", 300: "Light", 400: "Regular",
                500: "Medium", 600: "Semi Bold", 700: "Bold", 800: "Extra Bold", 900: "Black",
            }

            for font in font_manager.fontManager.ttflist:
                try:
                    family = font.family_name if hasattr(font, "family_name") else font.name
                    style = font.style_name.lower() if hasattr(font, "style_name") else font.style.lower()
                    weight = weight_conversion.get(font.weight, str(font.weight))

                    display_parts = [family]
                    if "italic" in style or "oblique" in style:
                        display_parts.append("Italic")
                    elif "normal" not in style:
                        display_parts.append(style.title())

                    if weight != "Regular":
                        display_parts.append(weight)

                    display_name = " ".join(display_parts)
                    if display_name not in self.font_map:
                        self.font_map[display_name] = font.fname

                except Exception as e:
                    continue

            self.font_choice.clear()
            self.font_choice.addItem("Default")
            self.font_choice.addItems(sorted(self.font_map.keys()))

        except Exception as e:
            QMessageBox.warning(self, "Font Error", f"Failed to load fonts: {str(e)}")
            self.font_choice.addItems(["Arial", "Times New Roman", "Verdana"])

    def load_colormaps(self):
        try:
            import matplotlib.pyplot as plt
            self.color_theme.addItems(plt.colormaps())
        except ImportError as e:
            self.color_theme.clear()
            self.color_theme.addItem("Default")
            QMessageBox.warning(self, "Dependency Error", f"Failed to load color maps: {str(e)}")

    def analyze_sentiment(self):
        """Analyze sentiment with button state management"""
        if not self.text_data:
            QMessageBox.critical(self, "Error", "No text data available for sentiment analysis.\nPlease load a text file first.")
            return

        if self.sentiment_mode in ["Flair", "TextBlob"]:
            try:
                from langdetect import detect
            except:
                pass

        if (self.sentiment_mode == "Flair" and not self.flair_classifier and self.sentiment_mode_combo.currentText() == "Flair"):
            QMessageBox.warning(self, "Model Loading", "Default Flair model is still loading. Please wait...")
            return

        if self.sentiment_mode == "Flair (Custom Model)":
            if not self.flair_classifier_cuslang:
                QMessageBox.warning(self, "Model Required", "Please load a custom Flair model first!")
                return

        classifier = self.flair_classifier_cuslang if self.sentiment_mode == "Flair (Custom Model)" else self.flair_classifier

        self.button_manager.disable_other_buttons('sentiment_button')

        self.sentiment_thread = SentimentAnalysisThread(
            self.text_data, 
            self.sentiment_mode,
            self.vader_analyzer,
            self.flair_classifier,
            self.flair_classifier_cuslang,
            self.textblob_analyzer
        )  
        self.sentiment_thread.translation_failed.connect(self.handle_translation_failure)
        self.sentiment_thread.offline_warning.connect(self.handle_offline_warning)
        self.sentiment_thread.sentiment_analyzed.connect(self.on_sentiment_analyzed)

        self.thread_manager.add_thread(self.sentiment_thread)

        self.set_progress('model')
        self.sentiment_thread.start()

    def stop_all_processes(self):
        """Enhanced process termination including word cloud generation"""
        print("Starting enhanced process termination...")
        self.disable_buttons()

        self.progress_timer.stop()
        self.unified_progress_bar.setValue(0)
        self.unified_progress_bar.setVisible(False)

        QApplication.processEvents()        

        try:
            self.threads_mutex.lock()
            all_threads = self.active_threads.copy()
            self.threads_mutex.unlock()

            stopped_threads = []
            failed_to_stop = []

            if hasattr(self, 'file_loader_thread') and self.file_loader_thread:
                if self.file_loader_thread.isRunning():
                    print("Terminating File Loader thread...")
                    all_threads.append(self.file_loader_thread)

            if hasattr(self, 'sentiment_thread') and self.sentiment_thread:
                if self.sentiment_thread.isRunning():
                    print("Terminating Sentiment Analysis thread...")
                    all_threads.append(self.sentiment_thread)

            if hasattr(self, 'flair_loader_thread') and self.flair_loader_thread:
                if self.flair_loader_thread.isRunning():
                    print("Terminating Flair Model Loader thread...")
                    all_threads.append(self.flair_loader_thread)

            if hasattr(self, 'lexicon_loader_thread') and self.lexicon_loader_thread:
                if self.lexicon_loader_thread.isRunning():
                    print("Terminating Lexicon Loader thread...")
                    all_threads.append(self.lexicon_loader_thread)

            if hasattr(self, 'model_loader_thread') and self.model_loader_thread:
                if self.model_loader_thread.isRunning():
                    print("Terminating Model Loader thread...")
                    all_threads.append(self.model_loader_thread)

            if hasattr(self, 'import_thread') and self.import_thread:
                if self.import_thread.isRunning():
                    print("Terminating Import thread...")
                    self.import_thread.stop()
                    all_threads.append(self.import_thread)

            if hasattr(self, 'topic_tab'):
                if hasattr(self.topic_tab, 'topic_thread') and self.topic_tab.topic_thread:
                    if self.topic_tab.topic_thread.isRunning():
                        print("Terminating Topic Analysis thread...")
                        all_threads.append(self.topic_tab.topic_thread)
                        
                if hasattr(self.topic_tab, 'keyword_thread') and self.topic_tab.keyword_thread:
                    if self.topic_tab.keyword_thread.isRunning():
                        print("Terminating Keyword Extraction thread...")
                        all_threads.append(self.topic_tab.keyword_thread)

            for thread in all_threads:
                try:
                    if not thread.isRunning():
                        continue
                        
                    print(f"Terminating {type(thread).__name__}...")
                    thread.requestInterruption()
                    
                    if thread.wait(800):
                        print(f"Graceful stop: {type(thread).__name__}")
                        stopped_threads.append(thread)
                    else:
                        print(f"Forcing termination: {type(thread).__name__}")
                        thread.terminate()
                        if thread.wait(500):
                            stopped_threads.append(thread)
                        else:
                            failed_to_stop.append(thread)
                            
                except Exception as e:
                    print(f"Termination error: {str(e)}")
                    failed_to_stop.append(thread)

            self.threads_mutex.lock()
            self.active_threads = [
                t for t in self.active_threads 
                if t not in stopped_threads and t.isRunning()
            ]
            self.threads_mutex.unlock()

            report = f"""
            <b>Termination Report:</b>
            • Total threads: {len(all_threads)}
            • Successfully stopped: {len(stopped_threads)}
            • Failed to stop: {len(failed_to_stop)}
            • Remaining active: {len(self.active_threads)}
            """
            QMessageBox.information(self, "Process Report", report)

        except Exception as e:
            print(f"Critical termination error: {str(e)}")
        finally:
            self.progress_timer.stop()
            self.unified_progress_bar.setValue(0)
            self.unified_progress_bar.setVisible(False)      
            self.enable_buttons()
            print("Enhanced termination completed")

    def disable_buttons(self):
        """Disable all buttons during processing"""
        process_buttons = [
            self.generate_wordcloud_button,
            self.sentiment_button, 
            self.custom_lexicon_button,
            self.custom_model_button,
            self.save_wc_button,
            self.text_stats_button
        ]
        
        if hasattr(self, 'topic_tab'):
            if hasattr(self.topic_tab, 'analyze_topics_btn'):
                process_buttons.append(self.topic_tab.analyze_topics_btn)
            if hasattr(self.topic_tab, 'extract_keywords_btn'):
                process_buttons.append(self.topic_tab.extract_keywords_btn)
                
        for button in process_buttons:
            button.setEnabled(False)
            
        self.load_file_button.setEnabled(True)
        self.panic_button.setEnabled(True)
        
        QApplication.processEvents()

    def disable_processing_buttons(self):
        """Alias for disable_buttons for compatibility"""
        self.disable_buttons()

    def enable_buttons(self):
        """Re-enable buttons after processing"""
        has_text = bool(self.text_data)
        
        self.load_file_button.setEnabled(True)
        self.view_fulltext_button.setEnabled(has_text)
        self.generate_wordcloud_button.setEnabled(has_text)
        self.text_stats_button.setEnabled(has_text)
        self.save_wc_button.setEnabled(has_text)
        self.sentiment_button.setEnabled(has_text)
        self.summarize_button.setEnabled(has_text)
        
        if hasattr(self, 'topic_tab'):
            if hasattr(self.topic_tab, 'analyze_topics_btn'):
                self.topic_tab.analyze_topics_btn.setEnabled(has_text)
            if hasattr(self.topic_tab, 'extract_keywords_btn'):
                self.topic_tab.extract_keywords_btn.setEnabled(has_text)
        
        if self.sentiment_mode == "VADER (Custom Lexicon)":
            self.custom_lexicon_button.setEnabled(True)
            self.custom_model_button.setEnabled(False)
        elif self.sentiment_mode == "Flair (Custom Model)":
            self.custom_lexicon_button.setEnabled(False)  
            self.custom_model_button.setEnabled(True)
        else:
            self.custom_lexicon_button.setEnabled(False)
            self.custom_model_button.setEnabled(False)
            
        QApplication.processEvents()    

    def closeEvent(self, event):
        """Enhanced application shutdown"""
        reply = QMessageBox.question(
            self,
            "Quit Confirmation",
            "<b>Are you sure you want to quit?</b><br>"
            "All ongoing processes will be terminated.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            event.ignore()
            return

        print("Initiating application shutdown...")
        
        try:
            if self.import_thread:
                self.import_thread.stop()
                self.import_thread.wait()
            
            self.threads_mutex.lock()
            active_threads = self.active_threads.copy()
            self.threads_mutex.unlock()
            
            for thread in active_threads:
                try:
                    if thread and thread.isRunning():
                        thread.requestInterruption()
                        thread.quit()
                        if not thread.wait(500):
                            thread.terminate()
                            thread.wait()
                except:
                    pass
                    
            self.cleanup()
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
        finally:
            event.accept()

    def cleanup(self):
        """Clean up resources before exit"""
        try:
            self.threads_mutex.lock()
            try:
                active_threads = self.active_threads.copy()
                for thread in active_threads:
                    try:
                        if thread.isRunning():
                            thread.requestInterruption()
                            thread.quit()
                            if not thread.wait(500):
                                thread.terminate()
                    except:
                        pass
                self.active_threads.clear()
            finally:
                self.threads_mutex.unlock()

            self.get_wordcloud.cache_clear()
            LazyLoader._cache.clear()
            self._cached_models.clear()
            self._cached_fonts.clear()
            self._cached_colormaps.clear()

            self.resource_manager.cleanup()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            import matplotlib.pyplot as plt
            plt.close('all')

            temp_dir = Path(tempfile.gettempdir()) / "wcgen_cache"
            if (temp_dir.exists()):
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            print(f"Cleanup error: {e}")
        finally:
            QApplication.processEvents()

    def show_about(self):
        about_text = base64.b64decode(ABOUT_TEXT.encode()).decode()

        dialog = QDialog(self)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setWindowTitle("About WCGen")
        dialog.setMinimumSize(500, 400)
        dialog.setSizeGripEnabled(True)
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
        dialog.show()

    # def open_file(self):
    #     options = QFileDialog.Options()
    #     file_path, _ = QFileDialog.getOpenFileName(
    #         self, "Load Text File", "", "Supported Files (*.txt *.pdf *.doc *.docx *.csv *.xlsx *.xls);;All Files (*)", options=options
    #     )
    #     if not file_path:
    #         return

    #     self.set_progress('file')
    
    #     try:
    #         self.file_loader_thread = FileLoaderThread(file_path)
    #         self.file_loader_thread.file_loaded.connect(self.on_file_loaded)
    #         self.file_loader_thread.file_error.connect(self.handle_file_error)

    #         self.thread_manager.add_thread(self.file_loader_thread)

    #         self.file_loader_thread.start()

    #     except Exception as e:
    #         self.unified_progress_bar.setVisible(False)
    #         QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def open_file(self):
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Load Text File")
        dialog.setNameFilter("Supported Files (*.txt *.pdf *.doc *.docx *.csv *.xlsx *.xls);;All Files (*)")
        dialog.setOptions(QFileDialog.Options())
        dialog.setContextMenuPolicy(Qt.NoContextMenu)  # Menonaktifkan context menu

        if dialog.exec_():
            file_path = dialog.selectedFiles()[0]
            if not file_path:
                return

            self.set_progress('file')

            try:
                self.file_loader_thread = FileLoaderThread(file_path)
                self.file_loader_thread.file_loaded.connect(self.on_file_loaded)
                self.file_loader_thread.file_error.connect(self.handle_file_error)

                self.thread_manager.add_thread(self.file_loader_thread)

                self.file_loader_thread.start()

            except Exception as e:
                self.unified_progress_bar.setVisible(False)
                QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def handle_file_error(self, error_message):
        self.unified_progress_bar.setVisible(False)
        QMessageBox.critical(self, "File Load Error", error_message)
        self._reset_file_state()

    def on_file_loaded(self, file_path, text_data):
        """Handle file loading completion"""
        self.set_progress('file', False)

        try:
            if not text_data.strip():
                raise ValueError("File appears empty after loading")

            self.text_data = text_data
            self.file_name.setText(os.path.basename(file_path))
            self.topic_tab.set_text(self.text_data)

            self.generate_wordcloud_button.setEnabled(True)
            self.save_wc_button.setEnabled(True)
            self.text_stats_button.setEnabled(True)
            self.view_fulltext_button.setEnabled(True)
            self.summarize_button.setEnabled(True)
            
            self.load_file_button.setEnabled(True)

            self.change_sentiment_mode(self.sentiment_mode)

        except ValueError as e:
            self.handle_file_error(str(e))
            self._reset_file_state()
            
        self.load_file_button.setEnabled(True)

    def _reset_file_state(self):
        self.text_data = ""
        self.file_name.setText("")
        self.generate_wordcloud_button.setEnabled(False)
        self.save_wc_button.setEnabled(False)
        self.text_stats_button.setEnabled(False)
        self.view_fulltext_button.setEnabled(False)
        self.sentiment_button.setEnabled(False)
        self.summarize_button.setEnabled(False)

    def choose_mask(self):
        options = QFileDialog.Options()
        mask_path, _ = QFileDialog.getOpenFileName(
            self, "Select Mask Image", "", "Image Files (*.png *.jpg *.bmp)", options=options
        )
        if mask_path:
            self.mask_path = mask_path
            self.mask_path_label.setText(f"{mask_path}")

    def reset_mask(self):
        self.mask_path = ""
        self.mask_path_label.setText("default (rectangle)")

    def import_stopwords(self):
        custom_words = self.stopword_entry.toPlainText().strip().lower()
        if custom_words:
            self.additional_stopwords = set(custom_words.split())
        return STOPWORDS.union(self.additional_stopwords)

    def text_analysis_report(self):
        if not self.text_data:
            QMessageBox.critical(self, "Error", "Load text file first!")
            return

        self.button_manager.disable_other_buttons('text_stats_button')
        
        try:
            stopwords = self.import_stopwords()
            words = [word.lower() for word in self.text_data.split() if word.lower() not in stopwords]
            word_counts = Counter(words)
            sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

            text_length = len(self.text_data)
            word_count = len(words)
            char_count_excl_spaces = len(self.text_data.replace(" ", ""))
            avg_word_length = char_count_excl_spaces / word_count if word_count > 0 else 0
            most_frequent_words = [word for word, count in sorted_word_counts[:5]]

            if hasattr(self, "stats_dialog") and self.stats_dialog is not None:
                self.stats_dialog.close()

            self.stats_dialog = QDialog(self)
            self.stats_dialog.setWindowModality(Qt.NonModal)
            self.stats_dialog.setWindowTitle("Text Analysis Report")
            self.stats_dialog.setMinimumSize(500, 400)
            self.stats_dialog.setSizeGripEnabled(True)

            layout = QVBoxLayout()

            text_browser = QTextBrowser()

            txtstat_content = f"""
            <h3 style="text-align: center;">Text Analysis Overview</h3>
            <table border="1" cellspacing="0" cellpadding="2" width="100%" style="margin-top: 20px;">
                <tr style="background-color: #d3d3d3;">
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr><td>Text Length</td><td>{text_length} characters</td></tr>
                <tr><td>Word Count</td><td>{word_count}</td></tr>
                <tr><td>Character Count (excluding spaces)</td><td>{char_count_excl_spaces}</td></tr>
                <tr><td>Average Word Length</td><td>{avg_word_length:.2f}</td></tr>
                <tr><td>Most Frequent Words</td><td>{", ".join(most_frequent_words)}</td></tr>
            </table>

            <h3 style="margin-top: 20px; text-align: center;">Word Count</h3>  
            <table border="1" cellspacing="0" cellpadding="2" width="100%"style="margin-top: 20px;">
                <tr style="background-color: #d3d3d3;">
                    <th>Word</th>
                    <th>Count</th>
                </tr>
            """
            for word, count in sorted_word_counts:
                txtstat_content += f"<tr><td>{word}</td><td>{count}</td></tr>"
            txtstat_content += "</table>"

            text_browser.setHtml(txtstat_content)
            text_browser.setOpenExternalLinks(True)
            text_browser.setReadOnly(True)
            layout.addWidget(text_browser)

            close_button = QPushButton("Close")
            close_button.clicked.connect(self.stats_dialog.accept)
            layout.addWidget(close_button)

            self.stats_dialog.setLayout(layout)
            self.stats_dialog.finished.connect(lambda: self.button_manager.restore_states())
            self.stats_dialog.show()
            
            # Ensure buttons are enabled after dialog is opened
            self.button_manager.restore_states()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate text analysis: {e}")
            self.button_manager.restore_states()

    def generate_wordcloud(self):
        """Generate word cloud with enhanced thread handling"""
        import matplotlib.pyplot as plt
        
        if not self.text_data:
            QMessageBox.critical(self, "Error", "Load text file first!")
            return

        try:
            if self.current_figure:
                plt.close(self.current_figure)
                self.current_figure = None

            font_path = None
            selected_font = self.font_choice.currentText()
            if selected_font != "Default":
                font_path = self.font_map.get(selected_font)
                if not font_path or not os.path.exists(font_path):
                    QMessageBox.warning(self, "Font Error", f"Font file not found: {font_path}")
                    return

            mask = None
            if self.mask_path:
                try:
                    mask = np.array(Image.open(self.mask_path))
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load mask image: {e}")
                    return

            colormap = self.color_theme.currentText()
            if (colormap in self.custom_color_palettes):
                colors = self.custom_color_palettes[colormap]
                colormap = LinearSegmentedColormap.from_list(colormap, colors)

            mem_available, cpu_available = self.perf_monitor.check_resources()
            if mem_available < 500:
                self.cleanup_cache()

            self.button_manager.disable_other_buttons('generate_wordcloud_button')
            self.set_progress('wordcloud')

            wc = MPWordCloud(
                width=800, height=400,
                background_color=self.bg_color.currentText(),
                stopwords=self.import_stopwords(),
                colormap=colormap,
                max_words=self.max_words_input.value(),
                min_font_size=self.min_font_size_input.value(),
                mask=mask,
                font_path=font_path
            )
            
            wc.generate_threaded(
                self.text_data,
                progress_callback=lambda p: self.unified_progress_bar.setValue(p),
                finished_callback=lambda wc_obj: self.display_wordcloud(wc_obj),
                error_callback=self.handle_wordcloud_error
            )

            self.current_wordcloud = wc

        except Exception as e:
            self.handle_wordcloud_error(str(e))

    def display_wordcloud(self, wordcloud):
        """Display generated wordcloud"""
        try:
            import matplotlib.pyplot as plt
            
            plt.ion()
            self.current_figure = plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")

            title_text = self.title_entry.text().strip()
            if title_text:
                title_font = None
                if self.font_choice.currentText() != "Default":
                    title_font = FontProperties(
                        fname=self.font_map.get(self.font_choice.currentText()),
                        size=self.title_font_size.value()
                    )
                plt.title(
                    title_text, 
                    loc=self.title_position.currentText().lower(),
                    fontproperties=title_font
                )

            plt.axis("off")
            plt.show(block=False)

            self.save_wc_button.setEnabled(True)
            
        except Exception as e:
            self.handle_wordcloud_error(str(e))
        finally:
            self.button_manager.restore_states()
            self.set_progress('wordcloud', False)

    def handle_wordcloud_error(self, error_msg):
        """Handle word cloud generation errors"""
        import matplotlib.pyplot as plt
        
        if self.current_figure:
            try:
                plt.close(self.current_figure)
            except:
                pass
            finally:
                self.current_figure = None
                
        QMessageBox.critical(self, "Error", f"Failed to generate word cloud: {error_msg}")
        self.button_manager.restore_states()
        self.set_progress('wordcloud', False)

    def save_wordcloud(self):
        self.button_manager.disable_other_buttons('save_wc_button')
        
        try:
            options = QFileDialog.Options()
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save WordCloud", "", "PNG file (*.png);;JPG file (*.jpg)", options=options
            )
            if not save_path:
                self.button_manager.restore_states()
                return

            stopwords = self.import_stopwords()
            mask = None

            font_path = None
            if self.font_choice.currentText() != "Default":
                from matplotlib import font_manager
                selected_font = self.font_choice.currentText()
                font_path = self.font_map.get(selected_font)

                try:
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
                    width=800, height=400, background_color=self.bg_color.currentText(), stopwords=stopwords,
                    colormap=self.color_theme.currentText(), max_words=self.max_words_input.value(),
                    min_font_size=self.min_font_size_input.value(), mask=mask,
                    font_path=None if self.font_choice.currentText() == "Default" else self.font_choice.currentText()
                ).generate(self.text_data)
                wc.to_file(save_path)
                QMessageBox.information(self, "Succeed", "Word cloud saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save word cloud: {e}")
        finally:
            self.button_manager.restore_states()

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
        
        try:
            if hasattr(self, "text_dialog") and self.text_dialog is not None:
                self.text_dialog.close()

            self.text_dialog = QDialog(self)
            self.text_dialog.setWindowModality(Qt.NonModal)
            self.text_dialog.setWindowTitle("Full Text")
            self.text_dialog.setMinimumSize(500, 400)
            self.text_dialog.setSizeGripEnabled(True)

            layout = QVBoxLayout()

            text_browser = QTextBrowser()
            text_browser.setPlainText(self.text_data)
            text_browser.setOpenExternalLinks(True)
            text_browser.setReadOnly(True)
            layout.addWidget(text_browser)

            close_button = QPushButton("Close")
            close_button.clicked.connect(self.text_dialog.accept)
            layout.addWidget(close_button)

            self.text_dialog.setLayout(layout)
            self.text_dialog.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display text: {e}")

    def handle_offline_warning(self, msg):
        self.unified_progress_bar.setVisible(False)
        QMessageBox.warning(self, "Offline Mode", msg)

    def handle_translation_failure(self, error_msg):
        """Handle translation failure gracefully"""
        self.unified_progress_bar.setVisible(False)
        QMessageBox.warning(self, "Translation Error", error_msg)
        self.enable_buttons()

    def on_sentiment_analyzed(self, result):
        self.set_progress('model', False)
        self.button_manager.restore_states()
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Sentiment Analysis Results - {self.sentiment_mode}")
        dialog.setMinimumSize(400, 350)  # Tambahkan tinggi untuk catatan
        dialog.setSizeGripEnabled(True)

        layout = QVBoxLayout()
        text_browser = QTextBrowser()

        score_type = ""
        if self.sentiment_mode in ["VADER", "VADER (Custom Lexicon)"]:
            score_type = "Compound Score"
        elif self.sentiment_mode in ["TextBlob", "TextBlob (Custom Lexicon)"]:
            score_type = "Polarity Score"
        elif self.sentiment_mode in ["Flair", "Flair (Custom Model)"]:
            score_type = "Confidence Score"

        # Tabel Hasil Analisis Sentimen
        sentiment_result = f"""
        <h3 style="text-align: center;">Sentiment Analysis Results</h3>
        <table border="1" cellspacing="0" cellpadding="2" width="100%" style="margin-top: 20px;">
            <tr style="background-color: #d3d3d3;">
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr><td>Analysis Mode</td><td>{self.sentiment_mode}</td></tr>
            <tr><td>Sentiment Label</td><td><b>{result["sentiment_label"]}</b></td></tr>
            <tr><td>Positive Sentiment</td><td>{result["positive_score"]:.2f}</td></tr>
            <tr><td>Neutral Sentiment</td><td>{result["neutral_score"]:.2f}</td></tr>
            <tr><td>Negative Sentiment</td><td>{result["negative_score"]:.2f}</td></tr>
            <tr><td>{score_type}</td><td>{result["compound_score"]:.2f}</td></tr>
        """

        if self.sentiment_mode in ["TextBlob", "TextBlob (Custom Lexicon)"]:
            try:
                subj_value = float(result["subjectivity"])
                sentiment_result += f'<tr><td>Subjectivity</td><td>{subj_value:.2f}</td></tr>'
            except (ValueError, TypeError):
                sentiment_result += f'<tr><td>Subjectivity</td><td>{result["subjectivity"]}</td></tr>'

        sentiment_result += "</table>"

        # **Tambahkan Catatan Berdasarkan Metode Sentimen**
        sentiment_notes = ""

        if self.sentiment_mode in ["TextBlob", "TextBlob (Custom Lexicon)"]:
            sentiment_notes = """
            <p><b>Note:</b></p>
            <ul>
                <li>Sentiment analysis is performed using <b>TextBlob</b>, which assigns a sentiment polarity score between -1 and 1.</li>
                <li><b>Polarity:</b> A value close to <b>-1</b> indicates a strongly negative sentiment, while a value close to <b>1</b> indicates a strongly positive sentiment.</li>
                <li><b>Neutral Range:</b> Scores near <b>0</b> suggest a neutral sentiment.</li>
                <li><b>Subjectivity Score:</b> Ranges from <b>0</b> (very objective) to <b>1</b> (very subjective).</li>
                <li><b>Objective vs Subjective:</b> 
                    <ul>
                        <li>A low subjectivity score (<b>≤ 0.3</b>) suggests that the text is more factual and objective.</li>
                        <li>A high subjectivity score (<b>≥ 0.7</b>) indicates that the text contains opinions, emotions, or subjective statements.</li>
                    </ul>
                </li>
                <li>Results may vary depending on the text's context, sarcasm, and linguistic nuances.</li>
            </ul>
            """

        elif self.sentiment_mode in ["VADER", "VADER (Custom Lexicon)"]:
            sentiment_notes = """
            <p><b>Note:</b></p>
            <ul>
                <li>Sentiment analysis is performed using <b>VADER</b>, which is optimized for social media, short texts, and informal language.</li>
                <li><b>Compound Score:</b> The overall sentiment score, ranging from <b>-1</b> (very negative) to <b>1</b> (very positive).</li>
                <li><b>Thresholds:</b>
                    <ul>
                        <li>Compound score <b>≥ 0.05</b>: Positive sentiment.</li>
                        <li>Compound score <b>≤ -0.05</b>: Negative sentiment.</li>
                        <li>Compound score between <b>-0.05</b> and <b>0.05</b>: Neutral sentiment.</li>
                    </ul>
                </li>
                <li>VADER considers punctuation, capitalization, and emoticons to enhance sentiment detection.</li>
            </ul>
            """

        elif self.sentiment_mode in ["Flair", "Flair (Custom Model)"]:
            sentiment_notes = """
            <p><b>Note:</b></p>
            <ul>
                <li>Sentiment analysis is performed using <b>Flair</b>, a deep learning-based model for text classification.</li>
                <li><b>Model:</b> Flair uses a pre-trained <b>Bidirectional LSTM</b> trained on large sentiment datasets.</li>
                <li><b>Sentiment Labels:</b> The output consists of two possible classifications:
                    <ul>
                        <li><b>POSITIVE</b>: Indicates an overall positive sentiment.</li>
                        <li><b>NEGATIVE</b>: Indicates an overall negative sentiment.</li>
                    </ul>
                </li>
                <li><b>Confidence Score:</b> Flair provides a probability score (0 to 1) indicating how confident the model is in its classification.</li>
                <li>Unlike lexicon-based approaches (e.g., TextBlob, VADER), Flair captures <b>context and word relationships</b>, making it more robust for longer and complex texts.</li>
            </ul>
            """

        # **Gabungkan hasil analisis dan catatan**
        text_browser.setHtml(sentiment_result + sentiment_notes)
        text_browser.setOpenExternalLinks(True)
        text_browser.setReadOnly(True)
        layout.addWidget(text_browser)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.show()

    @lru_cache(maxsize=128)
    def get_most_frequent_words(self, text, n):
        stopwords = self.import_stopwords().union(STOPWORDS)
        words = [word.lower() for word in text.split() if word.lower() not in stopwords]
        word_counts = Counter(words)
        most_common = word_counts.most_common(n)
        return [word for word, count in most_common]

    def change_sentiment_mode(self, mode):
        """Handle sentiment mode changes with proper button states"""
        self.sentiment_mode = mode
        status_text = ""
        has_text = bool(self.text_data.strip())

        # Reset button states
        self.custom_lexicon_button.setEnabled(False)
        self.custom_model_button.setEnabled(False)
        self.sentiment_button.setEnabled(False)

        if mode == "Flair":
            if self.flair_classifier:
                # Flair sudah dimuat
                status_text = "Flair: Ready"
                self.sentiment_button.setEnabled(has_text)
            else:
                # Flair belum dimuat
                status_text = "Loading..."
                self.load_flair_model()

        elif mode == "Flair (Custom Model)":
            if not self.flair_classifier:
                # Flair belum dimuat sama sekali
                self.load_flair_model()
                status_text = "Initializing Flair environment..."
            else:
                # Flair sudah dimuat
                self.custom_model_button.setEnabled(True)
                if self.flair_classifier_cuslang:  
                    status_text = "Ready"
                    self.sentiment_button.setEnabled(has_text)
                else:
                    status_text = "Load custom model first!"

        elif mode == "VADER (Custom Lexicon)":
            self.custom_lexicon_button.setEnabled(True)
            status_text = "Load custom lexicon first!" if not self.custom_lexicon_path else "Ready"
            self.sentiment_button.setEnabled(has_text and bool(self.custom_lexicon_path))
            if not self.custom_lexicon_path:
                QMessageBox.warning(
                    self, "Lexicon Required", 
                    "Please load a custom lexicon before analyzing sentiment."
                )

        elif mode == "TextBlob (Custom Lexicon)":
            self.custom_lexicon_button.setEnabled(True)
            status_text = "Load custom lexicon first!" if not self.custom_textblob_lexicon_path else "Ready"
            self.sentiment_button.setEnabled(has_text and bool(self.custom_textblob_lexicon_path))
            if not self.custom_textblob_lexicon_path:
                QMessageBox.warning(
                    self, "Lexicon Required", 
                    "Please load a custom lexicon before analyzing sentiment."
                )

        else:
            # Untuk mode TextBlob dan VADER default
            status_text = "Ready" if has_text else "Load text first!"
            self.sentiment_button.setEnabled(has_text)

        self.sentiment_button.setToolTip(
            f"{mode} - {status_text}\nClick to analyze sentiment using {mode}"
        )

    def load_custom_lexicon(self):
        options = QFileDialog.Options()
        lexicon_path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom Lexicon File", "", "Text Files (*.txt)", options=options
        )
        if lexicon_path:
            try:
                self.button_manager.disable_other_buttons('custom_lexicon_button')
                self.custom_lexicon_button.setText("Loading Lexicon...")
                self.custom_lexicon_button.setEnabled(False)
                self.sentiment_button.setEnabled(False)

                self.lexicon_loader_thread = CustomFileLoaderThread(
                    lexicon_path, 
                    "TextBlob (Custom Lexicon)" if self.sentiment_mode == "TextBlob (Custom Lexicon)" else "lexicon"
                )
                self.lexicon_loader_thread.file_loaded.connect(self.on_lexicon_loaded)
                self.thread_manager.add_thread(self.lexicon_loader_thread)
                self.lexicon_loader_thread.start()

            except Exception as e:
                self.button_manager.restore_states()
                self.custom_lexicon_button.setText("Load Lexicon")
                QMessageBox.critical(self, "Error", f"Failed to start lexicon loading: {str(e)}")

    def on_lexicon_loaded(self, result, success):
        self.unified_progress_bar.setVisible(False)
        self.button_manager.restore_states()
        self.custom_lexicon_button.setText("Load Lexicon")
        
        try:
            if success:
                if self.sentiment_mode == "TextBlob (Custom Lexicon)":
                    self.custom_textblob_lexicon_path = result
                    self.textblob_analyzer = CustomTextBlobSentimentAnalyzer(self.custom_textblob_lexicon_path)
                    QMessageBox.information(self, "Success", "Custom TextBlob lexicon loaded!")
                else:
                    self.custom_lexicon_path = result
                    self.vader_analyzer = CustomVaderSentimentIntensityAnalyzer(custom_lexicon_file=self.custom_lexicon_path)
                    QMessageBox.information(self, "Success", "Custom VADER lexicon loaded!")
                
                has_text = bool(self.text_data.strip())
                self.sentiment_button.setEnabled(has_text)
                
            else:
                raise ValueError(str(result))
                
        except Exception as e:
            if self.sentiment_mode == "TextBlob (Custom Lexicon)":
                self.custom_textblob_lexicon_path = None
                self.textblob_analyzer = None
            else:
                self.custom_lexicon_path = None
                self.vader_analyzer = None
                
            QMessageBox.critical(self, "Error", f"Failed to load lexicon: {str(e)}")
            self.sentiment_button.setEnabled(False)

    def load_custom_model(self):
        options = QFileDialog.Options()
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom Model File", "", "Model Files (*.pt)", options=options
        )
        if model_path:
            try:
                self.button_manager.disable_other_buttons('custom_model_button')
                self.custom_model_button.setText("Loading Model...")
                self.custom_model_button.setEnabled(False)
                self.sentiment_button.setEnabled(False)

                self.model_loader_thread = CustomFileLoaderThread(model_path, "model")
                self.model_loader_thread.file_loaded.connect(self.on_model_loaded)
                self.thread_manager.add_thread(self.model_loader_thread)
                self.model_loader_thread.start()

            except Exception as e:
                self.button_manager.restore_states()
                self.custom_model_button.setText("Load Model")
                QMessageBox.critical(self, "Error", f"Failed to start model loading: {str(e)}")

    def on_model_loaded(self, result, success):
        """Handle when custom Flair model is loaded"""
        self.unified_progress_bar.setVisible(False)
        self.custom_model_button.setText("Load Model")
        
        try:
            if success:
                # Validasi model
                from flair.models import TextClassifier
                from flair.data import Sentence

                if not isinstance(result, TextClassifier):
                    raise ValueError("Invalid model type. Please load a valid Flair TextClassifier model.")

                test_sentence = Sentence("test")
                result.predict(test_sentence)
                
                if not test_sentence.labels:
                    raise ValueError("Model didn't produce any labels")
                    
                label = test_sentence.labels[0].value
                if label not in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                    raise ValueError(f"Model produces incompatible labels: {label}")

                # Set model dan notifikasi
                self.flair_classifier_cuslang = result
                QMessageBox.information(self, "Success", "Custom model loaded successfully!")
                
                # Restore semua state tombol sebelumnya dan aktifkan tombol yang perlu
                self.button_manager.restore_states()
                has_text = bool(self.text_data.strip())
                
                # Aktifkan tombol-tombol yang bergantung pada teks
                if has_text:
                    self.sentiment_button.setEnabled(True)
                    self.view_fulltext_button.setEnabled(True)
                    self.summarize_button.setEnabled(True)
                    self.text_stats_button.setEnabled(True)
                    if hasattr(self, 'topic_tab'):
                        if hasattr(self.topic_tab, 'analyze_topics_btn'):
                            self.topic_tab.analyze_topics_btn.setEnabled(True)
                        if hasattr(self.topic_tab, 'extract_keywords_btn'):
                            self.topic_tab.extract_keywords_btn.setEnabled(True)
                
                # Pastikan tombol load model tetap aktif
                self.custom_model_button.setEnabled(True)
                
            else:
                raise ValueError(str(result))

        except Exception as e:
            self.flair_classifier_cuslang = None
            QMessageBox.critical(self, "Error", f"Failed to load custom model: {str(e)}")
            self.sentiment_button.setEnabled(False)
            # Restore state tombol lainnya
            self.button_manager.restore_states()
            # Pastikan tombol load model tetap aktif untuk retry
            self.custom_model_button.setEnabled(True)

    def load_flair_model(self):
        
        try:
            self.button_manager.disable_other_buttons('sentiment_button')
            self.sentiment_button.setText("Loading Flair...")
            self.sentiment_button.setEnabled(False)
            
            self.flair_loader_thread = FlairModelLoaderThread()
            self.flair_loader_thread.model_loaded.connect(self.on_flair_model_loaded)
            self.flair_loader_thread.error_occurred.connect(self.on_flair_model_error)
            self.flair_loader_thread.finished.connect(self.cleanup_flair_thread)

            self.thread_manager.add_thread(self.flair_loader_thread)
            self.add_managed_thread(self.flair_loader_thread)

            self.flair_loader_thread.start()
            
        except Exception as e:
            self.button_manager.restore_states()
            self.sentiment_button.setText("Analyze Sentiment")
            QMessageBox.critical(self, "Error", f"Failed to load Flair: {str(e)}")

    def on_flair_model_loaded(self, model):
        """Handle when default Flair model is loaded"""
        self.unified_progress_bar.setVisible(False)
        self.button_manager.restore_states()
        self.sentiment_button.setText("Analyze Sentiment")
        
        if model:
            self.flair_classifier = model
            has_text = bool(self.text_data.strip())
            
            if self.flair_first_load:
                self.flair_first_load = False
                QMessageBox.information(self, "Ready", "Flair library loaded successfully!")

            if self.sentiment_mode == "Flair":
                # Aktifkan tombol sentiment jika ada teks
                self.sentiment_button.setEnabled(has_text)
                
            elif self.sentiment_mode == "Flair (Custom Model)":
                # Aktifkan tombol load model
                self.custom_model_button.setEnabled(True)
                if self.flair_classifier_cuslang:
                    # Jika custom model sudah dimuat, aktifkan sentiment button jika ada teks
                    self.sentiment_button.setEnabled(has_text)
                else:
                    QMessageBox.information(self, "Next Step", "Please load your custom model")
                    self.sentiment_button.setEnabled(False)
            
        else:
            self.sentiment_button.setEnabled(False)
            QMessageBox.critical(self, "Error", "Failed to load Flair model")

    def on_flair_model_error(self, error):
        self.unified_progress_bar.setVisible(False)
        QMessageBox.critical(self, "Loading Error", f"Failed to load Flair model: {error}")

    def cleanup_flair_thread(self):
        self.progress_timer.stop()
        self.unified_progress_bar.setValue(0)
        self.unified_progress_bar.setVisible(False)
        self.thread_manager.remove_thread(self.flair_loader_thread)
        self.flair_loader_thread = None        

    def show_sentiment_mode_info(self):
        mode_info = base64.b64decode(MODE_INFO.encode()).decode()

        dialog = QDialog(self)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setWindowTitle("Sentiment Analysis Modes")
        dialog.setMinimumSize(500, 400)
        dialog.setSizeGripEnabled(True)

        layout = QVBoxLayout()
        text_browser = QTextBrowser()
        text_browser.setHtml(mode_info)
        text_browser.setOpenExternalLinks(True)
        text_browser.setReadOnly(True)
        layout.addWidget(text_browser)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.show()

    def update_topic_analysis(self, text):
        """Update topic analysis tab with new text"""
        self.topic_tab.set_text(text)

    def _update_progress_animation(self):
        """Smooth indeterminate progress animation"""
        if hasattr(self, 'unified_progress_bar') and self.unified_progress_bar.isVisible():
            current = self.unified_progress_bar.value()
            next_value = (current + 2) % 100
            self.unified_progress_bar.setValue(next_value)

    def resizeEvent(self, event):
        """Handle resize to keep progress bar full width"""
        super().resizeEvent(event)
        if hasattr(self, 'unified_progress_bar'):
            container = self.unified_progress_bar.parent()
            self.unified_progress_bar.setFixedWidth(container.width())
            self.unified_progress_bar.setGeometry(0, 0, container.width(), container.height())

    def update_stopwords_for_topic(self, custom_stopwords=None):
        """Initialize and update stopwords with consistent preprocessing"""
        from nltk.corpus import stopwords
        
        try:
            default_stopwords = set(stopwords.words('english'))
        except:
            default_stopwords = STOPWORDS
            
        default_stopwords = {word.lower().strip() for word in default_stopwords}
        
        if custom_stopwords:
            custom_stopwords = {word.lower().strip() for word in custom_stopwords if word.strip()}
            self.stop_words = default_stopwords.union(custom_stopwords)
        else:
            self.stop_words = default_stopwords

        vectorizer_config = {
            'token_pattern': r'(?u)\b[a-zA-Z][a-zA-Z0-9\'-]*[a-zA-Z0-9]\b',
            'stop_words': list(self.stop_words),
            'lowercase': True,
            'strip_accents': 'unicode',
            'max_features': 1000,
            'min_df': 1,
            'max_df': 1 #0.95
        }
        
        self.vectorizer = CountVectorizer(**vectorizer_config)
        self.tfidf = TfidfVectorizer(**vectorizer_config)

    def _extract_tfidf(self, text, num_keywords):
        """Extract keywords using TF-IDF with consistent preprocessing"""
        if not hasattr(self, 'stop_words'):
            self.update_stopwords_for_topic()
            
        vectorizer = TfidfVectorizer(
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9\'-]*[a-zA-Z0-9]\b',
            stop_words=list(self.stop_words),
            lowercase=True,
            strip_accents='unicode',
            min_df=1,
            max_df=1  # Diubah dari 0.95 menjadi 1 untuk konsistensi dengan LDA/NMF
        )
        
        try:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                sentences = [text]
                
            response = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            scores = response.mean(axis=0).A[0]
            
            pairs = list(zip(scores, feature_names))
            pairs.sort(reverse=True)
            
            return [{'keyword': word, 'score': score} 
                    for score, word in pairs[:num_keywords]]
                    
        except ValueError as e:
            print(f"TF-IDF extraction error: {str(e)}")
            return [{'keyword': 'Error: No valid keywords found', 'score': 0.0}]

    def _process_topic_modeling(self, text, num_topics, model_type='lda'):
        """Generic topic modeling processor"""
        try:
            sentences = [sent.strip() for sent in text.split('.') if sent.strip()] or [text]
            
            vectorizer = self.vectorizer if model_type == 'lda' else self.tfidf
            dtm = vectorizer.fit_transform(sentences)
            
            # Tambahkan pengecekan khusus untuk NMF
            if model_type == 'nmf':
                max_possible_topics = min(dtm.shape[0], dtm.shape[1])
                if num_topics > max_possible_topics:
                    raise ValueError(f"Number of topics ({num_topics}) cannot exceed {max_possible_topics} for NMF with this dataset")
            
            # Pengecekan umum untuk kedua model
            num_topics = min(num_topics, max(2, dtm.shape[1] - 1))
            
            model_class = LatentDirichletAllocation if model_type == 'lda' else NMF
            model_opts = {
                'n_components': num_topics,
                'random_state': 42,
                'max_iter': 20 if model_type == 'lda' else 500
            }
            if model_type == 'nmf':
                model_opts['init'] = 'nndsvd'
                model_opts['tol'] = 1e-4

            if model_type == 'lda':
                model_opts['learning_method'] = 'online'
                
            model = model_class(**model_opts)
            model.fit(dtm)
            
            terms = vectorizer.get_feature_names_out()
            return [{
                'topic': f'Topic {idx + 1}',
                'terms': [terms[i] for i in topic.argsort()[:-10-1:-1]],
                'weight': topic.sum()
            } for idx, topic in enumerate(model.components_)]
            
        except Exception as e:
            print(f"{model_type.upper()} Analysis error: {str(e)}")
            raise ValueError(f"Topic analysis failed: {str(e)}")

    def analyze_topics_lda(self, text, num_topics=5):
        """LDA topic analysis wrapper"""
        return self._process_topic_modeling(text, num_topics, 'lda')

    def analyze_topics_nmf(self, text, num_topics=5):
        """NMF topic analysis wrapper"""
        return self._process_topic_modeling(text, num_topics, 'nmf')

    def extract_keywords(self, text, method='tfidf', num_keywords=10):
        """Keyword extraction using various methods"""
        if method == 'tfidf':
            return self._extract_tfidf(text, num_keywords)
        elif method == 'yake':
            return self._extract_yake(text, num_keywords)
        elif method == 'rake':
            return self._extract_rake(text, num_keywords)
        else:
            raise ValueError(f"Unknown keyword extraction method: {method}")

    def _extract_yake(self, text, num_keywords):
        """YAKE extraction"""
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=2,
            windowsSize=2,
            top=num_keywords,
            stopwords=list(self.stop_words) if hasattr(self, 'stop_words') else None
        )
        try:
            keywords = kw_extractor.extract_keywords(text)
            return [{'keyword': kw[0], 'score': 1-kw[1]} for kw in keywords]
        except Exception:
            return [{'keyword': 'Error: YAKE extraction failed', 'score': 0.0}]

    def _extract_rake(self, text, num_keywords):
        """RAKE extraction"""
        try:
            rake = rake_nltk.Rake(stopwords=list(self.stop_words) if hasattr(self, 'stop_words') else None)
            rake.extract_keywords_from_text(text)
            keywords = rake.get_ranked_phrases_with_scores()
            keywords.sort(reverse=True)
            return [{'keyword': kw[1], 'score': kw[0]} for kw in keywords[:num_keywords]]
        except Exception:
            return [{'keyword': 'Error: RAKE extraction failed', 'score': 0.0}]
    
    def load_stopwords_file(self):
        """Load stopwords from a text file"""
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Load Stopwords File", 
                "", 
                "Text Files (*.txt);;All Files (*)", 
                options=options
            )
            
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as file:
                    stopwords = file.read().strip()
                    current_text = self.stopword_entry.toPlainText().strip()
                    if (current_text):
                        stopwords = current_text + "\n" + stopwords
                    self.stopword_entry.setPlainText(stopwords)
                    QMessageBox.information(self, "Success", "Stopwords loaded successfully!")
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load stopwords file: {str(e)}")

    def show_token_warning(self, tokens):
        """Show non-blocking warning message about detected tokens"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Analysis Notification")
        msg.setWindowModality(Qt.NonModal)
        msg.setTextFormat(Qt.TextFormat.RichText)
        
        message = f"""
        <b>Inconsistent tokens detected:</b> "{tokens}"<br><br>

        These tokens were detected by the tokenizer but are not found in your stopwords list.  
        This is not necessarily an issue, but it may indicate:<br>
        • Some unimportant words slipping through.<br>
        • Potential noise in your dataset.<br><br>

        <b>No immediate action is required.</b> However, if you’d like to refine your stopwords list, you may consider the following:<br>
        1. Review these tokens.<br>
        2. Add them to the stopwords list if they are irrelevant.<br>
        3. Regenerate the analysis for improved results.<br><br>

        <i>You can add them directly via the stopwords entry field.</i>
        """

        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        
        msg.show()

    def cleanup_cache(self):
        """Clear cached resources when memory is low"""
        try:
            self.get_wordcloud.cache_clear()
            
            self._cached_models.clear()
            self._cached_fonts.clear()
            self._cached_colormaps.clear()
                
        except Exception as e:
            print(f"Cache cleanup error: {e}")

    def summarize_text(self):
        if not self.text_data:
            QMessageBox.critical(self, "Error", "No text data available for summarization.\nPlease load a text file first.")
            return

        self.disable_buttons()
        self.summarize_thread = SummarizeThread(self.text_data)
        self.summarize_thread.summary_ready.connect(self.show_summary)
        self.summarize_thread.error_occurred.connect(self.handle_summarize_error)
        self.summarize_thread.finished.connect(self.enable_buttons)
        self.summarize_thread.start()

    def show_summary(self, summary):
        dialog = QDialog(self)
        dialog.setWindowTitle("Text Summary")
        dialog.setMinimumSize(500, 300)
        dialog.setSizeGripEnabled(True)
        layout = QVBoxLayout()

        text_browser = QTextBrowser()
        text_browser.setPlainText(summary)
        layout.addWidget(text_browser)

        # copy_button = QPushButton("Copy to Clipboard")
        # copy_button.clicked.connect(lambda: QApplication.clipboard().setText(summary))
        # layout.addWidget(copy_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.show()

    def handle_summarize_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"Failed to summarize text: {error_msg}")
        self.enable_buttons()

# Supporting Classes 
class LazyLoader:
    """
    Optimized lazy loading system for managing heavy dependencies.
    
    This class implements a singleton pattern with caching to ensure
    efficient loading and reuse of module imports. It helps reduce
    memory usage and startup time by loading modules only when needed.
    
    Attributes:
        _instances (dict): Singleton instances storage
        _cache (dict): Module cache storage
        
    Methods:
        load(module_name, class_name=None): 
            Lazily loads a module or class and caches it
            
    Example:
        >>> loader = LazyLoader()
        >>> TextBlob = loader.load('textblob', 'TextBlob')
        >>> vader = loader.load('vaderSentiment.vaderSentiment')
    """
    _instances = {}
    _cache = {}
    
    @classmethod
    def load(cls, module_name: str, class_name: str = None) -> object:
        """
        Lazily load and cache a module or specific class.
        
        Args:
            module_name (str): Name of the module to import
            class_name (str, optional): Specific class to import from module
            
        Returns:
            object: Loaded module or class
            
        Raises:
            ImportError: If module/class cannot be loaded
        """
        key = f"{module_name}.{class_name}" if class_name else module_name
        if key not in cls._cache:
            try:
                module = __import__(module_name, fromlist=[class_name] if class_name else [])
                cls._cache[key] = getattr(module, class_name) if class_name else module
            except ImportError as e:
                print(f"Failed to load {key}: {e}")
                return None
        return cls._cache[key]

class ThreadSafeSet:
    """Thread-safe set with context manager"""
    def __init__(self):
        self._items = set()
        self._mutex = QMutex()
    
    def locked(self):
        self._mutex.lock()
        return self._items

    def unlock(self):
        self._mutex.unlock()
            
    def add(self, item):
        items = self.locked()
        try:
            items.add(item)
        finally:
            self.unlock()
            
    def remove(self, item):
        items = self.locked()
        try:
            items.discard(item)
        finally:
            self.unlock()
            
    def clear(self):
        items = self.locked()
        try:
            items.clear()
        finally:
            self.unlock()

class ResourceManager:
    """Resource management with caching"""
    def __init__(self):
        self._resources = {}
        self._mutex = QMutex()
        
    def get(self, resource_id, creator_func):
        self._mutex.lock()
        try:
            if resource_id not in self._resources:
                self._resources[resource_id] = creator_func()
            return self._resources[resource_id]
        finally:
            self._mutex.unlock()
            
    def cleanup(self):
        self._mutex.lock()
        try:
            for resource in self._resources.values():
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
            self._resources.clear()
        finally:
            self._mutex.unlock()

class OptimizedWordCloud(WordCloud):
    """Optimized WordCloud with caching"""
    @lru_cache(maxsize=32)
    def _get_font_properties(self, font_path):
        return super()._get_font_properties(font_path)
        
    @lru_cache(maxsize=128)
    def _process_text(self, text):
        return super()._process_text(text)
        
    @lru_cache(maxsize=16)
    def _get_colormap(self, colormap_name):
        if isinstance(colormap_name, list):
            from matplotlib.colors import LinearSegmentedColormap
            return LinearSegmentedColormap.from_list('custom', colormap_name)
        return colormap_name

class ThreadManager:
    def __init__(self):
        self.active_threads = []
        self.mutex = QMutex()
        
    def add_thread(self, thread):
        self.mutex.lock()
        try:
            self.active_threads.append(thread)
        finally:
            self.mutex.unlock()
            
    def remove_thread(self, thread):
        self.mutex.lock()
        try:
            if thread in self.active_threads:
                self.active_threads.remove(thread)
        finally:
            self.mutex.unlock()
            
    def cleanup_finished(self):
        self.mutex.lock()
        try:
            self.active_threads = [
                t for t in self.active_threads 
                if t.isRunning() and not t.isFinished()
            ]
        finally:
            self.mutex.unlock()

class BaseAnalyzer:
    def __init__(self):
        self._initialized = False
        
    @property
    def is_initialized(self):
        return self._initialized
        
    def cleanup(self):
        pass

class SentimentAnalyzer(BaseAnalyzer):
    def __init__(self, mode, text_data, analyzers):
        super().__init__()
        self.mode = mode
        self.text_data = text_data
        self.analyzers = analyzers
        
    def get_score_type(self):
        return SCORE_TYPES.get(self.mode, "Score")

class CustomVaderSentimentIntensityAnalyzer:
    def __init__(self, lexicon_file="vader_lexicon.txt", custom_lexicon_file=None):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()
        if custom_lexicon_file:
            self.load_custom_lexicon(custom_lexicon_file)

    def load_custom_lexicon(self, custom_lexicon_file):
        try:
            with open(custom_lexicon_file, "r", encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        word, measure = parts
                        try:
                            self.analyzer.lexicon[word] = float(measure)
                        except ValueError:
                            pass
        except Exception as e:
            QMessageBox.warning(None, "Warning", f"Failed to load custom lexicon: {e}")

    def polarity_scores(self, text):
        return self.analyzer.polarity_scores(text)

class CustomTextBlobSentimentAnalyzer:
    def __init__(self, custom_lexicon_file=None):
        self.lexicon = {}
        self.intensifiers = {}
        self.negations = set()
        if custom_lexicon_file:
            self.load_custom_lexicon(custom_lexicon_file)

    def load_custom_lexicon(self, custom_lexicon_file):
        try:
            with open(custom_lexicon_file, "r", encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        word, measure = parts
                        if measure == "negation":
                            self.negations.add(word)
                        elif measure.startswith("intensifier:"):
                            try:
                                self.intensifiers[word] = float(measure.split(":")[1])
                            except ValueError:
                                continue
                        else:
                            try:
                                self.lexicon[word] = float(measure)
                            except ValueError:
                                continue
        except Exception as e:
            QMessageBox.warning(None, "Warning", f"Failed to load custom lexicon: {e}")

    def analyze(self, text):
        from textblob import TextBlob
        blob = TextBlob(text)
        total_polarity = 0.0
        words_with_context = []

        for sentence in blob.sentences:
            words = sentence.words
            for i, word in enumerate(words):
                word_lower = word.lower()
                current_score = self.lexicon.get(word_lower, 0.0)
                if i > 0 and words[i-1].lower() in self.negations:
                    current_score *= -1
                if i > 0 and words[i-1].lower() in self.intensifiers:
                    multiplier = self.intensifiers[words[i-1].lower()]
                    current_score *= multiplier
                words_with_context.append(current_score)

        valid_scores = [s for s in words_with_context if s != 0]
        avg_polarity = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return {
            'polarity': avg_polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'processed_words': len(valid_scores)
        }

class FlairModelLoaderThread(QThread):
    model_loaded = Signal(object)
    error_occurred = Signal(str)
    finished = Signal()
    _cached_model = None

    def __init__(self, model_path="sentiment"):
        super().__init__()
        self.model_path = model_path
        self._is_loading = False
        self._interrupt_requested = False

    def run(self):
        try:
            from flair.models import TextClassifier

            if self.isInterruptionRequested():
                return

            self._is_loading = True
            self._interrupt_requested = False

            if FlairModelLoaderThread._cached_model is None:
                try:
                    FlairModelLoaderThread._cached_model = TextClassifier.load(self.model_path)
                except Exception as e:
                    if not self.isInterruptionRequested():
                        self.error_occurred.emit(str(e))
                    return

            if self.isInterruptionRequested():
                return

            self.model_loaded.emit(FlairModelLoaderThread._cached_model)
        except Exception as e:
            if not self.isInterruptionRequested():
                self.error_occurred.emit(str(e))
        finally:
            self._is_loading = False
            self.finished.emit()

    def requestInterruption(self):
        print("FlairModelLoader: Interruption requested")
        self._interrupt_requested = True
        self._is_loading = False
        super().requestInterruption()

class TopicAnalysisThread(QThread):
    """Thread for topic analysis to prevent UI freezing"""
    finished = Signal(list)
    error = Signal(str)
    
    def __init__(self, text, num_topics, method='lda'):
        super().__init__()
        self.text = text
        self.num_topics = num_topics
        self.method = method
        
    def run(self):
        try:
            if self.method == 'lda':
                result = self.parent().analyze_topics_lda(self.text, self.num_topics)
            else:
                result = self.parent().analyze_topics_nmf(self.text, self.num_topics)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class KeywordExtractionThread(QThread):
    """Thread for keyword extraction to prevent UI freezing"""
    finished = Signal(list)
    error = Signal(str)
    
    def __init__(self, text, method, num_keywords):
        super().__init__()
        self.text = text
        self.method = method
        self.num_keywords = num_keywords
        
    def run(self):
        try:
            result = self.parent().extract_keywords(self.text, self.method, self.num_keywords)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class TopicAnalysisTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = ""
        self.parent_widget = parent
        self.setFixedHeight(140)
        self.initUI()
        self.topic_thread = None
        self.keyword_thread = None
        
    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(1, 1, 1, 1)
        main_layout.setSpacing(5)

        group_layout = QHBoxLayout()
     
        topic_group = QGroupBox("Topic Modeling")
        topic_layout = QGridLayout()
        topic_layout.setContentsMargins(10, 5, 10, 5)
        topic_layout.setVerticalSpacing(2)
        
        topic_layout.addWidget(QLabel("Method:"), 0, 0)
        self.topic_method = QComboBox()
        self.topic_method.addItems(['LDA', 'NMF'])
        self.topic_method.setToolTip(
            "LDA (For discovering topics in large text corpora)\n"
            "NMF (For extracting hidden topics using non-negative features)"
        )
        topic_layout.addWidget(self.topic_method, 0, 1)
        
        topic_layout.addWidget(QLabel("Number of Topics:"), 1, 0)
        self.num_topics = QSpinBox()
        self.num_topics.setRange(1, 20)
        self.num_topics.setValue(5)
        topic_layout.addWidget(self.num_topics, 1, 1)
        
        self.analyze_topics_btn = QPushButton("Analyze Topics")
        topic_layout.addWidget(self.analyze_topics_btn, 2, 0, 1, 2)
        
        topic_group.setLayout(topic_layout)
        group_layout.addWidget(topic_group)

        keyword_group = QGroupBox("Keyword Extraction") 
        keyword_layout = QGridLayout()
        keyword_layout.setContentsMargins(10, 5, 10, 5)
        keyword_layout.setVerticalSpacing(2)
        
        keyword_layout.addWidget(QLabel("Method:"), 0, 0)
        self.keyword_method = QComboBox()
        self.keyword_method.addItems(['TF-IDF', 'YAKE', 'RAKE'])
        self.keyword_method.setToolTip(
            "TF-IDF (For identifying important words based on frequency)\n"
            "RAKE (For extracting keywords from short texts)\n"
            "YAKE (For extracting context-based keywords from documents)"            
        )
        keyword_layout.addWidget(self.keyword_method, 0, 1)
        
        keyword_layout.addWidget(QLabel("Number of Keywords:"), 1, 0)
        self.num_keywords = QSpinBox()
        self.num_keywords.setRange(5, 50)
        self.num_keywords.setValue(10)
        keyword_layout.addWidget(self.num_keywords, 1, 1)
        
        self.extract_keywords_btn = QPushButton("Extract Keywords")
        keyword_layout.addWidget(self.extract_keywords_btn, 2, 0, 1, 2)
        
        keyword_group.setLayout(keyword_layout)
        group_layout.addWidget(keyword_group)
        
        main_layout.addLayout(group_layout)
        self.setLayout(main_layout)

        self.analyze_topics_btn.setEnabled(False)
        self.extract_keywords_btn.setEnabled(False)
        
        self.analyze_topics_btn.clicked.connect(self.analyze_topics)
        self.extract_keywords_btn.clicked.connect(self.extract_keywords)

    def set_text(self, text):
        self.text = text
        has_text = bool(text.strip())
        self.analyze_topics_btn.setEnabled(has_text)
        self.extract_keywords_btn.setEnabled(has_text)
        
    def analyze_topics(self):
        # if not hasattr(self, 'text'):
        if not self.text.strip():
            QMessageBox.warning(self, "Error", "Please load text first!")
            return
            
        try:
            if self.parent_widget:
                custom_stopwords = self.parent_widget.import_stopwords()
                self.parent_widget.update_stopwords_for_topic(custom_stopwords)

            method = self.topic_method.currentText()
            num_topics = self.num_topics.value()
            
            self.parent_widget.button_manager.disable_other_buttons('analyze_topics_btn')
            
            if self.parent_widget:
                self.parent_widget.set_progress('topics')
            
            self.topic_thread = TopicAnalysisThread(self.text, num_topics, method.lower())
            self.topic_thread.setParent(self.parent_widget)
            self.topic_thread.finished.connect(self.on_topic_analysis_complete)
            self.topic_thread.error.connect(self.on_analysis_error)
            self.topic_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Topic analysis failed: {str(e)}")
            self.parent_widget.button_manager.restore_states()
            if self.parent_widget:
                self.parent_widget.set_progress('topics', False)
            
    def extract_keywords(self):
        if not hasattr(self, 'text'):
            QMessageBox.warning(self, "Error", "Please load text first!")
            return
            
        try:
            if self.parent_widget:
                custom_stopwords = self.parent_widget.import_stopwords()
                self.parent_widget.update_stopwords_for_topic(custom_stopwords)

            method = self.keyword_method.currentText().lower().replace('-', '')
            num_keywords = self.num_keywords.value()
            
            self.parent_widget.button_manager.disable_other_buttons('extract_keywords_btn')
            
            if self.parent_widget:
                self.parent_widget.set_progress('keywords')
            
            self.keyword_thread = KeywordExtractionThread(self.text, method, num_keywords)
            self.keyword_thread.setParent(self.parent_widget)
            self.keyword_thread.finished.connect(self.on_keyword_extraction_complete)
            self.keyword_thread.error.connect(self.on_analysis_error)
            self.keyword_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Keyword extraction failed: {str(e)}")
            self.parent_widget.button_manager.restore_states()
            if self.parent_widget:
                self.parent_widget.set_progress('keywords', False)

    def on_topic_analysis_complete(self, topics):
        if self.parent_widget:
            self.parent_widget.set_progress('topics', False)
        self.parent_widget.button_manager.restore_states()
        self.show_results_dialog(self.topic_method.currentText(), topics)

    def on_keyword_extraction_complete(self, keywords):
        if self.parent_widget:
            self.parent_widget.set_progress('keywords', False)
        self.parent_widget.button_manager.restore_states()
        self.show_keyword_dialog(self.keyword_method.currentText(), keywords)

    def on_analysis_error(self, error_msg):
        if self.parent_widget:
            self.parent_widget.set_progress('topics', False)
            self.parent_widget.set_progress('keywords', False)
        self.parent_widget.button_manager.restore_states()
        QMessageBox.critical(self, "Error", error_msg)

    def show_results_dialog(self, method, topics):
        result_html = "<h3>Topic Analysis Results</h3>"
        for topic in topics:
            result_html += f"<p><b>{topic['topic']}</b> (weight: {topic['weight']:.2f})<br>"
            result_html += ", ".join(topic['terms']) + "</p>"

        dialog = QDialog(self)
        dialog.setWindowTitle(f"{method} Topic Results") 
        dialog.setSizeGripEnabled(True)
        self.setup_results_dialog(dialog, result_html)
        
    def show_keyword_dialog(self, method, keywords):
        result_html = "<h3 style='text-align: center;'>Keyword Extraction Results</h3>"
        result_html += "<table border='1' cellspacing='0' cellpadding='3' width='100%' style='margin-top: 20px;'>"
        result_html += "<tr style='background-color: #d3d3d3;'><th>Keyword</th><th>Score</th></tr>"
        
        for kw in keywords:
            result_html += f"<tr><td>{kw['keyword']}</td><td>{kw['score']:.4f}</td></tr>"
            
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{method.upper()} Keywords")
        dialog.setSizeGripEnabled(True)
        self.setup_results_dialog(dialog, result_html)

    def setup_results_dialog(self, dialog, html_content):
        dialog.setMinimumSize(500, 300)
        
        text_browser = QTextBrowser()
        text_browser.setHtml(html_content)
        
        close_btn = QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.close)
        
        layout = QVBoxLayout()
        layout.addWidget(text_browser)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)
        
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()

class WarningEmitter(QObject):
    token_warning = Signal(str)

class CustomWarningHandler:
    def __init__(self, emitter):
        self.emitter = emitter
        self.reported_tokens = set()

    def handle_warning(self, message, category, filename, lineno, file=None, line=None):
        if "Your stop_words may be inconsistent" in str(message):
            tokens = self._extract_tokens(str(message))
            if tokens:
                self.emitter.token_warning.emit(", ".join(tokens))
                self.reported_tokens.update(tokens)

    def _extract_tokens(self, message):
        start = message.find('[') + 1
        end = message.find(']')
        if start > 0 and end > start:
            return [token.strip("'") for token in message[start:end].split(', ')]
        return []

class PerformanceMonitor:
    """Monitor system resources and performance"""
    
    @staticmethod
    def check_resources():
        """Check available system resources"""
        try:
            mem = psutil.virtual_memory()
            mem_available = mem.available / 1024 / 1024
            cpu_available = 100 - psutil.cpu_percent()
            return mem_available, cpu_available
        except Exception as e:
            print(f"Resource check error: {e}")
            return 1000, 50

class ButtonStateManager:
    """Manages button states during processing operations"""
    def __init__(self, main_window):
        self.main_window = main_window
        self.button_states = {}
        self.process_buttons = [
            'generate_wordcloud_button',
            'sentiment_button',
            'analyze_topics_btn',
            'extract_keywords_btn',
            'view_fulltext_button',
            'text_stats_button',
            'save_wc_button',
            'summarize_button',
            'load_file_button',
        ]
        
    def save_states(self):
        """Save current state of all process buttons"""
        for btn_name in self.process_buttons:
            if hasattr(self.main_window, btn_name):
                btn = getattr(self.main_window, btn_name)
                self.button_states[btn_name] = btn.isEnabled()
            elif hasattr(self.main_window.topic_tab, btn_name):
                btn = getattr(self.main_window.topic_tab, btn_name)
                self.button_states[btn_name] = btn.isEnabled()
                
    def disable_other_buttons(self, active_button):
        """Disable all process buttons except the active one"""
        self.save_states()
        for btn_name in self.process_buttons:
            if btn_name != active_button:
                if hasattr(self.main_window, btn_name):
                    getattr(self.main_window, btn_name).setEnabled(False)
                elif hasattr(self.main_window.topic_tab, btn_name):
                    getattr(self.main_window.topic_tab, btn_name).setEnabled(False)
                    
    def restore_states(self):
        """Restore buttons to their previous states"""
        for btn_name, was_enabled in self.button_states.items():
            if hasattr(self.main_window, btn_name):
                getattr(self.main_window, btn_name).setEnabled(was_enabled)
            elif hasattr(self.main_window.topic_tab, btn_name):
                getattr(self.main_window.topic_tab, btn_name).setEnabled(was_enabled)
        self.button_states.clear()

class ImportThread(QThread):
    """Dedicated thread for import initialization"""
    finished = Signal()
    
    def __init__(self):
        super().__init__()
        self._is_running = True
        
    def run(self):
        try:
            if not self._is_running:
                return
                
            import matplotlib.pyplot as plt
            if not self._is_running:
                return
                
            import nltk
            if not self._is_running:
                return
                
            from textblob import TextBlob
            if not self._is_running:
                return
                
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                if self._is_running:
                    nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"Import error: {e}")
        finally:
            if self._is_running:
                self.finished.emit()

    def stop(self):
        """Metode untuk menghentikan thread dengan aman"""
        self._is_running = False
        self.wait()

class FileLoaderThread(QThread):
    file_loaded = Signal(str, str)
    file_error = Signal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            if self.isInterruptionRequested():
                return

            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")

            file_size = os.path.getsize(self.file_path)
            if file_size > 100 * 1024 * 1024:
                self.process_large_file()
            else:
                self.process_normal_file()

        except Exception as e:
            self.file_error.emit(f"Error loading {os.path.basename(self.file_path)}: {str(e)}")

    def process_large_file(self):
        chunk_processor = ChunkProcessor()
        
        def process_chunk(chunk):
            return chunk.strip() + '\n'
            
        try:
            text_data = ''
            for progress in chunk_processor.process_file_chunks_mp(
                self.file_path, process_chunk):
                self.progress.emit(progress)
                
            if not text_data.strip():
                raise ValueError("File is empty or contains no extractable text")
                
            self.file_loaded.emit(self.file_path, text_data)
            
        except Exception as e:
            self.file_error.emit(f"Error processing large file: {str(e)}")

    def process_normal_file(self):
        try:
            if self.isInterruptionRequested():
                return

            text_data = ""
            
            if self.file_path.endswith(".txt"):
                with open(self.file_path, "r", encoding="utf-8") as file:
                    while True:
                        if self.isInterruptionRequested():
                            return
                        line = file.readline()
                        if not line:
                            break
                        text_data += line

            elif self.file_path.endswith(".pdf"):
                text_data = self.extract_text_from_pdf(self.file_path)
                
            elif self.file_path.endswith((".doc", ".docx")):
                text_data = self.extract_text_from_word(self.file_path)
                
            elif self.file_path.endswith((".xlsx", ".xls")):
                text_data = self.extract_text_from_excel(self.file_path)
                
            elif self.file_path.endswith(".csv"):
                text_data = self.extract_text_from_csv(self.file_path)
                
            else:
                raise ValueError(f"Unsupported file format: {os.path.splitext(self.file_path)[1]}")

            if self.isInterruptionRequested():
                return

            if not text_data.strip():
                raise ValueError("File is empty or contains no extractable text")

            self.file_loaded.emit(self.file_path, text_data)

        except Exception as e:
            self.file_error.emit(f"Error loading {os.path.basename(self.file_path)}: {str(e)}")

    def extract_text_from_pdf(self, pdf_path):
        try:
            import pypdf
            text = ""
            with open(pdf_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                if not reader.pages:
                    raise ValueError("PDF contains no readable pages")
                
                for page in reader.pages:
                    if self.isInterruptionRequested():
                        return ""
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            raise RuntimeError(f"PDF processing failed: {str(e)}") from e

    def extract_text_from_word(self, word_path):
        try:
            import docx
            doc = docx.Document(word_path)
            if not doc.paragraphs:
                raise ValueError("Word document contains no readable text")
            
            text = ""
            for p in doc.paragraphs:
                if self.isInterruptionRequested():
                    return ""
                if p.text.strip():
                    text += p.text.strip() + "\n"
                    
            for table in doc.tables:
                if self.isInterruptionRequested():
                    return ""
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text += cell.text.strip() + "\n"
                            
            if not text.strip():
                raise ValueError("No readable text found in document")
                
            text = re.sub(r'\s+', ' ', text)
            text = text.replace('\n\n', '\n').strip()
            
            return text
            
        except Exception as e:
            raise RuntimeError(f"Word processing failed: {str(e)}") from e

    def extract_text_from_excel(self, excel_path):
        try:
            import pandas as pd
            text_data = ""
            df = pd.read_excel(excel_path, sheet_name=None)
            if not df:
                raise ValueError("Excel file is empty or unreadable")
            
            for sheet_name, sheet_df in df.items():
                if self.isInterruptionRequested():
                    return ""
                text_data += f"\n--- {sheet_name} ---\n"
                text_data += sheet_df.to_string(index=False, header=True)
            return text_data
        except Exception as e:
            raise RuntimeError(f"Excel processing failed: {str(e)}") from e

    def extract_text_from_csv(self, csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if df.empty:
                raise ValueError("CSV file is empty")
            
            if self.isInterruptionRequested():
                return ""
                
            return df.to_string(index=False, header=True)
        except Exception as e:
            raise RuntimeError(f"CSV processing failed: {str(e)}") from e

class SentimentAnalysisThread(QThread):
    sentiment_analyzed = Signal(dict)
    offline_warning = Signal(str)
    translation_failed = Signal(str)

    def __init__(self, text_data, sentiment_mode, vader_analyzer, flair_classifier, flair_classifier_cuslang, textblob_analyzer=None):
        super().__init__()
        self.text_data = text_data
        self.sentiment_mode = sentiment_mode
        self.vader_analyzer = vader_analyzer
        self.flair_classifier = flair_classifier
        self.flair_classifier_cuslang = flair_classifier_cuslang
        self.textblob_analyzer = textblob_analyzer
        self.cached_translation = None
        self.temp_dir = Path(tempfile.gettempdir()) / "wcgen_cache"
        
        self.text_hash = hashlib.md5(text_data.encode()).hexdigest()
        self.temp_file = self.temp_dir / f"trans_{self.text_hash}.txt"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def _async_translate(self, text):
        try:
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            return translated
        except Exception as e:
            return None

    def translate_text(self, text):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        sentences = [s.strip() for s in text.split(".") if s.strip()]
        translated = []

        async def run_translations():
            for sentence in sentences:
                try:
                    result = await self._async_translate(sentence)
                    if result:
                        translated.append(result)
                except Exception as e:
                    continue
            return ". ".join(translated)

        try:
            return loop.run_until_complete(run_translations())
        except Exception as e:
            return None
        finally:
            if loop.is_running():
                loop.close()

    def analyze_textblob(self, text, use_custom=False):
        if use_custom and self.textblob_analyzer:
            analysis = self.textblob_analyzer.analyze(text)
            polarity = analysis['polarity']
            subjectivity = analysis['subjectivity']
        else:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

        if abs(polarity) < 0.1:
            neutral_score = 1.0
            positive_score = negative_score = 0.0
        else:
            positive_score = (polarity + 1) / 2 if polarity > 0 else 0
            negative_score = (-polarity + 1) / 2 if polarity < 0 else 0
            neutral_score = 1 - (positive_score + negative_score)

        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "neutral_score": neutral_score,
            "compound_score": polarity,
            "sentiment_label": "POSITIVE" if polarity > 0 else "NEGATIVE" if polarity < 0 else "NEUTRAL",
            "subjectivity": subjectivity
        }

    def get_cached_translation(self):
        """Get translation from cache if exists"""
        if self.temp_file.exists():
            try:
                with open(self.temp_file, "r", encoding="utf-8") as f:
                    return f.read()
            except:
                return None
        return None
        
    def save_translation(self, translated_text):
        """Save translation to cache"""
        try:
            with open(self.temp_file, "w", encoding="utf-8") as f:
                f.write(translated_text)
        except:
            pass

    def run(self):
        result = {
            "positive_score": 0,
            "neutral_score": 0,
            "negative_score": 0,
            "compound_score": 0,
            "sentiment_label": "N/A",
            "subjectivity": "N/A",
        }

        try:
            text_to_analyze = self.text_data
            needs_translation = False

            if self.sentiment_mode in ["VADER", "Flair", "TextBlob"]:
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

                cached = self.get_cached_translation()
                if cached:
                    text_to_analyze = cached
                else:
                    translated_text = self.translate_text(self.text_data)
                    if not translated_text:
                        self.translation_failed.emit("Translation failed - analysis aborted")
                        return
                        
                    self.save_translation(translated_text)
                    text_to_analyze = translated_text

            if self.sentiment_mode == "TextBlob":
                result.update(self.analyze_textblob(text_to_analyze))
            elif self.sentiment_mode == "TextBlob (Custom Lexicon)":
                result.update(self.analyze_textblob(self.text_data, True))

            elif self.sentiment_mode == "VADER":
                if self.vader_analyzer:
                    sentiment = self.vader_analyzer.polarity_scores(text_to_analyze)
                    total = sentiment["pos"] + sentiment["neg"] + sentiment["neu"]
                    positive_score = sentiment["pos"]
                    negative_score = sentiment["neg"] / total if total > 0 else 0
                    neutral_score = sentiment["neu"] / total if total > 0 else 0

                    result.update({
                        "positive_score": positive_score,
                        "negative_score": negative_score,
                        "neutral_score": neutral_score,
                        "compound_score": sentiment["compound"],
                        "sentiment_label": "POSITIVE" if sentiment["compound"] >= 0.05 else "NEGATIVE" if sentiment["compound"] <= -0.05 else "NEUTRAL",
                    })
                else:
                    result["sentiment_label"] = "Error: VADER not initialized"

            elif self.sentiment_mode == "VADER (Custom Lexicon)":
                if self.vader_analyzer:
                    sentiment = self.vader_analyzer.polarity_scores(self.text_data)
                    result.update({
                        "positive_score": sentiment["pos"],
                        "negative_score": sentiment["neg"],
                        "neutral_score": sentiment["neu"],
                        "compound_score": sentiment["compound"],
                        "sentiment_label": "POSITIVE" if sentiment["compound"] >= 0.05 else "NEGATIVE" if sentiment["compound"] <= -0.05 else "NEUTRAL",
                    })
                else:
                    result["sentiment_label"] = "Error: Custom VADER not initialized"

            elif self.sentiment_mode == "Flair":
                from flair.data import Sentence
                sentence = Sentence(text_to_analyze)
                self.flair_classifier.predict(sentence)
                sentiment = sentence.labels[0]
                confidence = sentiment.score
                sentiment_label = sentiment.value

                model_labels = self.flair_classifier.label_dictionary.get_items()
                
                if 'NEUTRAL' not in model_labels:
                    if abs(confidence - 0.5) < 0.05:
                        sentiment_label = "NEUTRAL"
                        positive_score = 0.0
                        negative_score = 0.0
                        neutral_score = 1.0
                    else:
                        positive_score = confidence if sentiment_label == "POSITIVE" else 0.0
                        negative_score = confidence if sentiment_label == "NEGATIVE" else 0.0
                        neutral_score = 0.0
                else:
                    if confidence < 0.55:
                        sentiment_label = "NEUTRAL"
                        positive_score = 0.0
                        negative_score = 0.0
                        neutral_score = 1.0
                    else:
                        positive_score = confidence if sentiment_label == "POSITIVE" else 0.0
                        negative_score = confidence if sentiment_label == "NEGATIVE" else 0.0
                        neutral_score = confidence if sentiment_label == "NEUTRAL" else 0.0

                result.update({
                    "compound_score": confidence,
                    "sentiment_label": sentiment_label,
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "neutral_score": neutral_score,
                })            

            elif self.sentiment_mode == "Flair (Custom Model)":
                try:
                    from flair.data import Sentence
                    sentence = Sentence(text_to_analyze)
                    
                    if self.flair_classifier_cuslang is None:
                        raise ValueError("Custom model not loaded")
                        
                    self.flair_classifier_cuslang.predict(sentence)
                    if not sentence.labels:
                        raise ValueError("Model produced no labels")
                        
                    sentiment = sentence.labels[0]
                    confidence = sentiment.score
                    sentiment_label = sentiment.value

                    model_labels = self.flair_classifier.label_dictionary.get_items()
                    
                    if 'NEUTRAL' not in model_labels:
                        if abs(confidence - 0.5) < 0.05:
                            sentiment_label = "NEUTRAL"
                            positive_score = negative_score = 0.0
                            neutral_score = 1.0
                        else:
                            positive_score = confidence if sentiment_label == "POSITIVE" else 0
                            negative_score = confidence if sentiment_label == "NEGATIVE" else 0
                            neutral_score = 0
                    else:
                        if confidence < 0.55:
                            sentiment_label = "NEUTRAL"
                            positive_score = negative_score = 0.0
                            neutral_score = 1.0
                        else:
                            positive_score = confidence if sentiment_label == "POSITIVE" else 0
                            negative_score = confidence if sentiment_label == "NEGATIVE" else 0
                            neutral_score = 1.0 if sentiment_label == "NEUTRAL" else 0

                    result.update({
                        "compound_score": confidence,
                        "sentiment_label": sentiment_label,
                        "positive_score": positive_score,
                        "negative_score": negative_score, 
                        "neutral_score": neutral_score,
                    })            

                except ImportError as e:
                    result["sentiment_label"] = "Error: Flair not installed properly"
                except Exception as e:
                    result["sentiment_label"] = f"Error: {str(e)}"

            self.sentiment_analyzed.emit(result)

        except Exception as e:
            result["sentiment_label"] = f"Error: {str(e)}"
            self.sentiment_analyzed.emit(result)

class WordCloudGeneratorThread(QThread):
    """Thread for generating word clouds"""
    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, text, wc_params):
        super().__init__()
        self.text = text
        self.wc_params = wc_params
        self._cancelled = False

    def run(self):
        try:
            if len(self.text) > 100000:
                chunk_size = 10000
                chunks = [self.text[i:i+chunk_size] 
                         for i in range(0, len(self.text), chunk_size)]
                frequencies = {}
                
                for i, chunk in enumerate(chunks):
                    if self._cancelled:
                        return
                        
                    chunk_freqs = MPWordCloud._process_chunk(chunk, self.wc_params.get('stopwords'))
                    
                    for word, freq in chunk_freqs.items():
                        frequencies[word] = frequencies.get(word, 0) + freq
                        
                    progress = int((i + 1) / len(chunks) * 100)
                    self.progress.emit(progress)
                    
                wordcloud = WordCloud(**self.wc_params)
                wordcloud.generate_from_frequencies(frequencies)
            else:
                wordcloud = WordCloud(**self.wc_params)
                wordcloud.generate(self.text)
                self.progress.emit(100)
                
            self.finished.emit(wordcloud)
            
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True
        self.requestInterruption()

class CustomFileLoaderThread(QThread):
    file_loaded = Signal(object, bool)

    def __init__(self, file_path, file_type):
        super().__init__()
        self.file_path = file_path
        self.file_type = file_type

    def run(self):
        try:
            if self.file_type == "lexicon":
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.vader_analyzer = CustomVaderSentimentIntensityAnalyzer(custom_lexicon_file=self.file_path)
                self.file_loaded.emit(self.file_path, True)

            elif self.file_type == "TextBlob (Custom Lexicon)":
                from textblob import TextBlob
                self.textblob_analyzer = CustomTextBlobSentimentAnalyzer(self.file_path)
                self.file_loaded.emit(self.file_path, True)

            elif self.file_type == "model":
                from flair.models import TextClassifier
                from flair.data import Sentence
                try:
                    model = TextClassifier.load(self.file_path)
                    test = Sentence("test")
                    model.predict(test)
                    if not test.labels or test.labels[0].value not in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                        raise ValueError("Model is not a valid sentiment classifier")
                    self.file_loaded.emit(model, True)
                except Exception as e:
                    self.file_loaded.emit(f"Invalid sentiment model: {str(e)}", False)

        except Exception as e:
            self.file_loaded.emit(str(e), False)

class MPWordCloud(WordCloud):
    """Thread-enabled WordCloud for large texts"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator_thread = None
        
    @staticmethod
    def _process_chunk(chunk, stopwords=None):
        """Static method to process text chunks with stopwords support"""
        temp_wc = WordCloud(stopwords=stopwords)
        return temp_wc.process_text(chunk)

    def generate_threaded(self, text, progress_callback=None, finished_callback=None, error_callback=None):
        """Generate word cloud using threading"""
        wc_params = {
            'width': self.width,
            'height': self.height,
            'background_color': self.background_color,
            'mask': self.mask,
            'max_words': self.max_words,
            'min_font_size': self.min_font_size,
            'font_path': self.font_path,
            'colormap': self.colormap,
            'stopwords': self.stopwords,
        }
        
        self.generator_thread = WordCloudGeneratorThread(text, wc_params)
        
        if progress_callback:
            self.generator_thread.progress.connect(progress_callback)
        if finished_callback:    
            self.generator_thread.finished.connect(finished_callback)
        if error_callback:
            self.generator_thread.error.connect(error_callback)
            
        self.generator_thread.start()
        return self

class SummarizeThread(QThread):
    summary_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        try:
            parser = PlaintextParser.from_string(self.text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, 5)  # Summarize to 5 sentences
            summary_text = "\n".join(str(sentence) for sentence in summary)
            self.summary_ready.emit(summary_text)
        except Exception as e:
            self.error_occurred.emit(str(e))

# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    ex = MainClass()
    ex.show()

    def cleanup():
        """Enhanced cleanup"""
        try:
            ex.cleanup()
            loop.stop()
        finally:
            LazyLoader._cache.clear()
            for func in [func for func in globals().values() if hasattr(func, 'cache_clear')]:
                func.cache_clear()

    app.aboutToQuit.connect(cleanup)

    with loop:
        sys.exit(loop.run_forever())
