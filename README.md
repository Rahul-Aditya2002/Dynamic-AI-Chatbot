Dynamic AI Chatbot Project
Welcome to the Dynamic AI Chatbot project! This repository contains the source code and resources to run the chatbot application locally using Streamlit.

Contents
streamlit_app.py — Main Streamlit app interface

dynamic_ai_chatbot.py — Backend chatbot logic and data processing

dialogs.txt or equivalent — Dataset file containing question-answer pairs (you may have to add this)
Kindly edit the datasets location according to your system.
Setup Instructions for Local VS Code
Follow these steps to run the app successfully on your local machine:

1. Clone this repository
bash
git clone https://github.com/your-username/dynamic-ai-chatbot.git
cd dynamic-ai-chatbot
2. Install dependencies
Make sure Python 3.7+ is installed, then install required packages:

bash
pip install -r requirements.txt
3. Data File Setup
Ensure the dialogs.txt (or your Q&A dataset file) is placed in the root folder or the path provided in the code.

By default, dynamic_ai_chatbot.py expects the file at:

python
file_path = 'dialogs.txt'   # Default relative path
If your dataset file is elsewhere, update this path inside dynamic_ai_chatbot.py:

python
file_path = r'full/or/relative/path/to/your/dialogs.txt'
For example, if your dialogs file is inside a folder called data, change it to:

python
file_path = 'data/dialogs.txt'
4. Verify File Paths and Imports
Ensure streamlit_app.py imports the chatbot logic as:

python
from dynamic_ai_chatbot import enhanced_retrieval_bot
If you renamed any files, update imports accordingly.

5. Running the Application Locally
Run the Streamlit app via:

bash
streamlit run streamlit_app.py
The app will open in your browser.

Type messages to test the chatbot.

Check console logs for any errors related to missing data files or dependencies.

6. Common Changes to Make Before Running
Adjust file_path in dynamic_ai_chatbot.py to point to your dialogs file location.

If you use any additional datasets or config files, ensure their paths are correctly set relative to your working directory.

Confirm all dependencies are installed.

If you face issues with library versions or environment, try creating a new virtual environment.

Notes
This project was initially developed and tested running locally in VS Code.

Deployment to Streamlit Cloud may require additional changes for path and environment compatibility.

Feel free to create your own branch for cloud-specific fixes if needed.

Contact / Support
For questions or issues, please contact:

Rahul Aditya
Email: adirahul0408@gmail.com

Feel free to reach out if you need help setting this up!

This README gives clear guidance on changes users need to make for file paths and how to run the app locally in VS Code.
