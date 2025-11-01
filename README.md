# ATS Resume Screener & Career Assistant (Powered by Gemini)

This project is a powerful, dual-interface (Streamlit Web App & CLI) tool that leverages the Google Gemini API to analyze a PDF resume against a specific job description.

It acts as a personal career assistant, providing five distinct functions:

1. **Honest Resume Evaluation:** Simulates a technical recruiter's feedback.
2. **Skills Gap Analysis:** Acts as a career coach to identify missing skills.
3. **ATS Match Score:** Simulates an Applicant Tracking System (ATS) to give a match percentage and identify missing keywords.
4. **Resume Rewriting:** Acts as a professional resume writer to optimize your resume for ATS.
5. **Interview Prep:** Generates tailored technical and HR questions with sample answers based on the resume and job description.

The application is built to be robust, with a graceful fallback to a command-line interface (CLI) if streamlit is not installed, and includes offline-capable unit tests.

---

## Key Features

- 🖥️ **Dual Interface:** Run as a user-friendly Streamlit web app or as a robust CLI tool.
- 🧠 **Five-in-One Career Assistant:** Access five different AI personas (Recruiter, Coach, ATS Engine, Resume Writer, Interviewer) for comprehensive job application support.
- 📄 **PDF File Processing:** Uses the Gemini API's file uploading capability to directly analyze PDF resumes, ensuring accurate content extraction.
- 🔄 **Robust API Handling:** Includes logic to poll the API while the file is processing and provides automatic cleanup of uploaded files.
- 🧪 **Built-in Unit Tests:** Comes with a unittest suite that mocks all external google.generativeai calls, allowing for safe and fast offline testing of the core logic.

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.8+
- A Google Gemini API Key. You can get one from Google AI Studio.

### 2. Installation

Clone the repository (or save the script):

```bash
git clone [your-repo-url]
cd [your-repo-directory]
```
(Or, just save the Python file as `app.py`)

Create and activate a virtual environment:

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

Install the required dependencies:  
Create a `requirements.txt` file with the following content:

```
google-generativeai
python-dotenv
streamlit
```

Then, install them using pip:

```bash
pip install -r requirements.txt
```

### 3. Configuration

The application requires your Google Gemini API key.

1. Create a file named `.env` in the same directory as your Python script.
2. Add your API key to this file:

```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

---

## Usage

You can run this project in two modes:

### 1. Streamlit Web App (Recommended)

This is the primary, user-friendly interface.

Run the following command in your terminal:

```bash
streamlit run app.py
```
(Replace `app.py` with the name of your Python script).

- Your browser will automatically open to the web app.
- Paste the job description into the text area.
- Upload your resume PDF.
- Click any of the five buttons (e.g., "Percentage Match", "Improve My Skills") to get a response.

### 2. Command-Line Interface (CLI)

If Streamlit is not installed (or you set the environment variable `FORCE_CLI=1`), the script provides a full-featured CLI.

**Syntax:**

```bash
python app.py --resume <path_to_resume.pdf> --jobdesc "<job_description_text>" --action <action_name>
```

**Actions:**

- `evaluation` (default)
- `skills_improvement`
- `ats_match`
- `rewrite_resume`
- `interview_questions`

**Examples:**

```bash
# Get a general resume evaluation
python app.py --resume "My Resume.pdf" --jobdesc "Data Scientist role at Google..."

# Get an ATS match score
python app.py --resume "My Resume.pdf" --jobdesc "Data Scientist role..." --action ats_match

# Get interview questions
python app.py --resume "My Resume.pdf" --jobdesc "Data Scientist role..." --action interview_questions
```

---

## 🧪 Testing

The script includes self-contained unit tests that mock all external API calls. This allows you to verify the application's logic without using your API key or an internet connection.

To run the tests, use the unittest module from your terminal:

```bash
python -m unittest app.py
```
(Or, as specified in the script's `if __name__ == '__main__':` block, you can also run `python app.py --run-tests`)

---

## 🛠️ Code Overview

- `get_gemini_response(...)`: The core function. It handles:
  - Creating a temporary local file from the uploaded PDF.
  - Uploading the file to the Gemini API (`genai.upload_file`).
  - Polling the file status until it's DONE (no longer PROCESSING).
  - Calling `model.generate_content()` with the prompt, the processed file, and the job description.
  - Cleaning up the remote file (`genai.delete_file`) after the call.
- `prompts = {...}`: A dictionary holding the five detailed system prompts that define the AI's persona and task for each button.
- `start_streamlit_ui()`: Defines the full Streamlit application, including all UI elements (header, text area, file uploader, columns, and buttons) and the `run()` helper function that calls `get_gemini_response`.
- `run_cli(...)`: Parses command-line arguments using argparse and executes the `get_gemini_response` function, printing the result to the console.
- `GetGeminiResponseTests`: A `unittest.TestCase` class that uses `unittest.mock` to patch all genai functions (GenerativeModel, upload_file, etc.) to test the success and failure (e.g., FAILED state) paths of the application logic.
