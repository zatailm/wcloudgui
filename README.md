# WCGen - WordCloud Generator  
[![DOI](https://zenodo.org/badge/937892074.svg)](https://doi.org/10.5281/zenodo.14916874)

![WCGen](https://img.shields.io/badge/version-1.5-blue)  ![License](https://img.shields.io/badge/license-Personal%20%26%20Educational-green)  ![Python](https://img.shields.io/badge/Python-3.12.3-blue)  ![Miniconda](https://img.shields.io/badge/Miniconda-Supported-orange)  ![GUI](https://img.shields.io/badge/GUI-PySide6-yellow)  

## **📌 Introduction**  
**WCGen** (WordCloud Generator) is a powerful and user-friendly desktop application that allows users to generate **customizable word clouds** from various text sources.  
It provides **interactive visualization tools** for analyzing word frequency, making it ideal for **research, presentations, and text analytics**.  

### **🎯 Key Features**  
- 📂 **Supports multiple file formats**: `TXT`, `PDF`, `DOC/DOCX`, `CSV`, `XLSX`.  
- 🎨 **Customization options**: Change **colors, fonts, mask images, and themes**.  
- ❌ **Stopword filtering**: Remove common words to refine visualization.  
- 📊 **Word frequency statistics**: Display a table of most common words.  
- 📝 **Interactive UI**: Simple and intuitive **PySide6-based GUI**.  
- 🧠 **Optional Sentiment Analysis**: Analyze text polarity using `TextBlob`, `VADER`, or `Flair`.  
- 💾 **Save & Export**: Save generated word clouds in `PNG` and `JPG` formats.  
- 💻 **EXE version available for non-Python users!**  

---

## **📥 Installation (Python 3.12.3 - Miniconda Recommended)**  

### **🔹 Prerequisites**  
Ensure you have **Miniconda (or Anaconda) installed**.  
If not, download and install **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)**.

### **🔹 Create a Virtual Environment (Recommended)**  
Using **Miniconda**, it's best to create a virtual environment for WCGen:
```sh  
conda create --name wcgen-env python=3.12.3  
conda activate wcgen-env  
```

### **🔹 Install Dependencies**  
Once the environment is activated, install the required packages:
```sh  
pip install -r src/requirements.txt  
```
Or manually:
```sh  
pip install PySide6==6.0.2.1 matplotlib==3.10.0 wordcloud==1.9.4 pypdf==5.3.0 docx==1.1.2 pillow==11.1.0 numpy==1.26.4 pandas==2.2.3 textblob==0.19.0 vaderSentiment==3.3.2 flair==0.15.1 langdetect==1.0.9 qasync>=0.23.0 deep-translator>=1.9.2
```

### **🔹 Running WCGen**  
To launch WCGen after installation:
```sh  
python src/wcgen15beta.py  
```

---

## **📦 Precompiled EXE (For Non-Python Users)**  

For users **not familiar with Python**, WCGen is available as a **standalone Windows executable (.exe)**.  
This version **does not require Python** and can be run directly.

### **🔹 Download & Run the EXE Version**  
1. **Download `WCGen.exe`** from the [Releases](https://github.com/zatailm/wcloudgui/releases) page.  
2. Extract the `.zip` file (if necessary).  
3. **Double-click `WCGen.exe`** to open the application.  
4. *(Optional)* Create a shortcut for easier access.  

### **🔹 System Requirements**  
- Windows **10/11** (64-bit)  
- At least **4GB RAM** and **600MB free disk space**  

### **🛠️ How the EXE Was Built**  
The `.exe` was created using **PyInstaller**:
```sh  
pyinstaller src/wcgen15beta.spec  
```
### **⚠️ IMPORTANT NOTE ⚠️**  
WCGen **1.5**, which includes the new **Sentiment Analysis feature** along with additional new features, is currently in **beta testing**.  
At this time, the **compiled EXE version for WCGen 1.5 is not yet available**. The latest available EXE version is **WCGen 1.3**.  

Stay tuned for updates on the official **[GitHub Releases](https://github.com/zatailm/wcloudgui/releases)** page! 🚀


---

## **🖼️ Screenshots**  
WCGen 1.5 (beta testing):
![WCGen 1.5 Main UI](https://github.com/zatailm/wcloudgui/blob/main/res/wcgen15.png)  

WCGen 1.3:
![WCGen 1.3 Main UI](https://github.com/zatailm/wcloudgui/blob/main/res/wcgen13.png)  


---

## **📖 Dependencies**  
WCGen uses the following libraries:  
- **PySide6** → GUI framework  
- **Matplotlib** → Visualization backend  
- **WordCloud** → Word cloud generator  
- **NumPy & PIL** → Image processing  
- **PyPDF2, python-docx, pandas** → File handling  
- **VADER, TextBlob, Flair** → Sentiment analysis  
- **Langdetect** → Language detection
- **deep-translator** → Translate service

---

## **📜 License**  
WCGen is **free for personal and educational use**. For commercial applications, please refer to the licensing terms.  

---

## **🤝 Contributing**  
We welcome contributions! Feel free to:  
- 📌 Report bugs or suggest features via [Issues](https://github.com/zatailm/wcloudgui/issues).  
- 📌 Fork the repository and submit a **Pull Request**.  

---

## **📬 Contact**  
For inquiries or support, reach out via:  
📧 Email: `inizata@gmail.com`  
🔗 GitHub: [zatailm/wcloudgui](https://github.com/zatailm/wcloudgui)  


## 📥 Download  
🖥 **[Download here](https://github.com/zatailm/wcloudgui/releases)**  
