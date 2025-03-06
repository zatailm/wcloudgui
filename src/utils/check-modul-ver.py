import importlib
import pkg_resources  # Alternatif untuk mendapatkan versi modul
import importlib.metadata

# Daftar modul yang ingin dicek versinya
modules = [
    "sys", "asyncio", "PySide6", "matplotlib", "wordcloud", "pypdf",
    "docx", "PIL", "numpy", "pandas", "vaderSentiment", "flair",
    "googletrans", "langdetect", "textblob"
]

def get_version(modul):
    try:
        # Coba impor modul
        m = importlib.import_module(modul)

        # Coba ambil versi dengan __version__
        if hasattr(m, "__version__"):
            return m.__version__

        # Coba ambil versi dari metadata (Python 3.8+)
        return importlib.metadata.version(modul)

    except ModuleNotFoundError:
        return "Tidak ditemukan (belum diinstal)"
    except importlib.metadata.PackageNotFoundError:
        return "Versi tidak diketahui"
    except:
        try:
            # Coba ambil versi menggunakan pkg_resources (untuk modul lama)
            return pkg_resources.get_distribution(modul).version
        except:
            return "Versi tidak diketahui"

print("Cek Versi Modul:")
for modul in modules:
    print(f"{modul}: {get_version(modul)}")
