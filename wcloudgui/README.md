# WordCloud Generator

WordCloud Generator is a Python-based GUI application designed to help users analyze the main themes and topics in a text through frequency-based word visualization. It utilizes libraries such as `wordcloud`, `matplotlib`, and `tkinter` to provide an interactive and user-friendly experience.

## Features

- Supports text files (.txt), PDF files (.pdf), and Word documents (.doc, .docx).
- Customizable word cloud generation with options for color themes, font choices, and stopwords.
- Displays word statistics for the most common words in the selected text.
- Ability to save generated word clouds as image files.

## Requirements

To run this project, you need to have Python installed along with the following dependencies:

- `wordcloud`
- `matplotlib`
- `tkinter`
- `PyPDF2`
- `python-docx`

You can install the required dependencies using the following command:

```
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/zatailm/wcloudgui.git
   cd wcloudgui
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python wc.py
   ```

4. Use the GUI to select a text file, PDF, or Word document, and generate a word cloud.

## License

This project is licensed under the MIT License. Feel free to use it for personal and educational purposes.