"""
Ats App Upgraded (fixed)

This file contains:
- A robust implementation of `get_gemini_response` that labels the job description explicitly.
- A Streamlit UI *if* Streamlit is available.
- A CLI fallback so the code can run in environments without Streamlit (fixes ModuleNotFoundError).
- Unit tests that mock the `google.generativeai` calls so tests run offline.

Run tests with: python -m unittest this_file.py
Run CLI example: python this_file.py --resume sample.pdf --jobdesc "Data Analyst role"

DO NOT change expected behavior with respect to model prompts; only add fallbacks and tests.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import time
import tempfile
import io
import argparse
import sys
import unittest
from unittest import mock

# Streamlit may not be installed in the environment where tests run.
# Provide a graceful fallback: if Streamlit is available, use it; otherwise, provide a CLI interface.
try:
    import streamlit as st
except Exception:
    st = None

# Import google generative SDK. In unit tests we will mock this module's relevant pieces.
import google.generativeai as genai

# Configure Gemini API (reads key from .env). If GOOGLE_API_KEY is not set, the genai calls will likely fail;
# unit tests mock genai to avoid making real API calls.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"


#########################
# Core: Gemini function #
#########################

def get_gemini_response(uploaded_file, prompt, input_text, model_name=MODEL_NAME, wait_sleep=2):
    """
    Uploads a PDF-like file object to Gemini, requests content generation, and returns the text response.

    uploaded_file: file-like object implementing .read() -> bytes and having .name attribute (optional).
    prompt: system prompt string describing the role for the model.
    input_text: job description string.
    model_name: model identifier string.
    wait_sleep: seconds to sleep while polling processing state (kept short for tests).

    Returns: a string (response text)
    Raises: ValueError if processing fails or genai interactions raise exceptions.
    """
    # Create model wrapper
    model = genai.GenerativeModel(model_name)

    # Read bytes from the provided file-like object
    pdf_bytes = uploaded_file.read()
    temp_file_path = None

    try:
        # Save bytes to a temp file for upload_file which expects a filesystem path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_file_path = temp_file.name

        display_name = getattr(uploaded_file, "name", "uploaded_resume.pdf")

        gemini_file = genai.upload_file(
            temp_file_path,
            display_name=display_name,
            mime_type="application/pdf",
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

    # Poll until processed or failed
    while getattr(gemini_file, "state", None) and getattr(gemini_file.state, "name", "") == "PROCESSING":
        time.sleep(wait_sleep)
        gemini_file = genai.get_file(gemini_file.name)

    if getattr(gemini_file, "state", None) and getattr(gemini_file.state, "name", "") == "FAILED":
        # Clean remote file if possible
        try:
            genai.delete_file(gemini_file.name)
        except Exception:
            pass
        raise ValueError("PDF processing failed")

    # Ensure the job description is clearly labeled so the model doesn't ignore it
    labeled_jd = f"Job Description:\n{input_text}\n\nUse this job description while evaluating the resume."

    response = model.generate_content([
        prompt,
        gemini_file,
        labeled_jd
    ])

    # Clean-up remote file
    try:
        genai.delete_file(gemini_file.name)
    except Exception:
        pass

    # Some SDKs return different shapes; normalize to .text if available
    text = getattr(response, "text", None)
    if text is None:
        # Try to string-cast as fallback
        text = str(response)

    return text


#########################
# Prompts               #
#########################

"""
Ats App Upgraded (fixed)

This file contains:
- A robust implementation of `get_gemini_response` that labels the job description explicitly.
- A Streamlit UI *if* Streamlit is available.
- A CLI fallback so the code can run in environments without Streamlit (fixes ModuleNotFoundError).
- Unit tests that mock the `google.generativeai` calls so tests run offline.

Run tests with: python -m unittest this_file.py
Run CLI example: python this_file.py --resume sample.pdf --jobdesc "Data Analyst role"

DO NOT change expected behavior with respect to model prompts; only add fallbacks and tests.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import time
import tempfile
import io
import argparse
import sys
import unittest
from unittest import mock

# Streamlit may not be installed in the environment where tests run.
# Provide a graceful fallback: if Streamlit is available, use it; otherwise, provide a CLI interface.
try:
    import streamlit as st
except Exception:
    st = None

# Import google generative SDK. In unit tests we will mock this module's relevant pieces.
import google.generativeai as genai

# Configure Gemini API (reads key from .env). If GOOGLE_API_KEY is not set, the genai calls will likely fail;
# unit tests mock genai to avoid making real API calls.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"


#########################
# Core: Gemini function #
#########################

def get_gemini_response(uploaded_file, prompt, input_text, model_name=MODEL_NAME, wait_sleep=2):
    """
    Uploads a PDF-like file object to Gemini, requests content generation, and returns the text response.

    uploaded_file: file-like object implementing .read() -> bytes and having .name attribute (optional).
    prompt: system prompt string describing the role for the model.
    input_text: job description string.
    model_name: model identifier string.
    wait_sleep: seconds to sleep while polling processing state (kept short for tests).

    Returns: a string (response text)
    Raises: ValueError if processing fails or genai interactions raise exceptions.
    """
    # Create model wrapper
    model = genai.GenerativeModel(model_name)

    # Read bytes from the provided file-like object
    pdf_bytes = uploaded_file.read()
    temp_file_path = None

    try:
        # Save bytes to a temp file for upload_file which expects a filesystem path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_file_path = temp_file.name

        display_name = getattr(uploaded_file, "name", "uploaded_resume.pdf")

        gemini_file = genai.upload_file(
            temp_file_path,
            display_name=display_name,
            mime_type="application/pdf",
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

    # Poll until processed or failed
    while getattr(gemini_file, "state", None) and getattr(gemini_file.state, "name", "") == "PROCESSING":
        time.sleep(wait_sleep)
        gemini_file = genai.get_file(gemini_file.name)

    if getattr(gemini_file, "state", None) and getattr(gemini_file.state, "name", "") == "FAILED":
        # Clean remote file if possible
        try:
            genai.delete_file(gemini_file.name)
        except Exception:
            pass
        raise ValueError("PDF processing failed")

    # Ensure the job description is clearly labeled so the model doesn't ignore it
    labeled_jd = f"Job Description:\n{input_text}\n\nUse this job description while evaluating the resume."

    response = model.generate_content([
        prompt,
        gemini_file,
        labeled_jd
    ])

    # Clean-up remote file
    try:
        genai.delete_file(gemini_file.name)
    except Exception:
        pass

    # Some SDKs return different shapes; normalize to .text if available
    text = getattr(response, "text", None)
    if text is None:
        # Try to string-cast as fallback
        text = str(response)

    return text


#########################
# Prompts               #
#########################

prompts = {
    "evaluation": (
        "You are a highly experienced Technical Recruiter & Hiring Manager specializing in AI, ML, DS, Analytics, Cloud & Engineering roles.\n"
        "Evaluate the resume against the job description with uncompromising honesty.\n"
        "Return:\n"
        "- Overall fit and summary\n"
        "- Technical skills relevance\n"
        "- Work experience relevance\n"
        "- Academic relevance\n"
        "- Strengths\n"
        "- Weaknesses (be direct and specific, no sugarcoating)\n"
        "- Recommendation\n\n"
        "If the resume is weak, clearly say it and explain what exactly is lacking.\n"
        "If something is good, acknowledge it. Provide practical correction guidance."
    ),

    "skills_improvement": (
        "You are a Senior Career Coach for AI, ML, Data Science & Analytics.\n"
        "Provide a brutally honest skills gap analysis.\n"
        "Return:\n"
        "- Missing technical & soft skills (clearly highlight weaknesses)\n"
        "- Certifications to pursue\n"
        "- Tools/tech to learn\n"
        "- Portfolio & project ideas\n"
        "- Job readiness rating (strict and realistic)\n\n"
        "Keep advice actionable and beginner friendly.\n"
        "No motivational tone; pure facts, corrections, and improvement plan."
    ),

    "ats_match": (
        "You are an ATS scoring engine.\n"
        "Be strict and unbiased.\n"
        "Evaluate:\n"
        "1. Match percentage (0-100, do not inflate)\n"
        "2. Missing keywords\n"
        "3. Red flags (call them out clearly)\n"
        "4. ATS feedback (objective and direct)\n"
        "5. Final recommendation\n\n"
        "Return formatted tables when helpful.\n"
        "If the resume would be rejected by ATS, say it plainly."
    ),

    "rewrite_resume": (
        "You are a professional resume writer for tech roles.\n"
        "Rewrite the resume to be ATS-optimized.\n"
        "Ensure:\n"
        "- Strong action verbs\n"
        "- Achievement-based bullets\n"
        "- Quantified results\n"
        "- Job-focused keywords\n"
        "- ATS formatting compliance\n\n"
        "If experience/skills are weak, make realistic improvements without inventing fake achievements."
        "\nOutput in clean text format."
    ),

    "interview_questions": (
        "You are a tech interviewer.\n"
        "Based on this resume and job description, generate:\n"
        "- 10 technical questions\n"
        "- 5 HR questions\n"
        "- Best possible sample answers\n"
        "- Preparation tips\n\n"
        "Questions must expose weak areas if present.\n"
        "Sample answers should be realistic and helpful, not idealistic or exaggerated."
    ),
}



#########################
# CLI / Streamlit glue  #
#########################

def run_cli(args):
    # Read resume bytes from file
    if not os.path.exists(args.resume):
        print(f"Resume file not found: {args.resume}")
        return 2

    with open(args.resume, "rb") as f:
        class UploadedFile:
            def __init__(self, buf, name):
                self._buf = buf
                self.name = name
            def read(self):
                return self._buf

        pdf_bytes = f.read()
        uploaded = UploadedFile(pdf_bytes, os.path.basename(args.resume))

        prompt_key = args.action
        if prompt_key not in prompts:
            print(f"Unknown action '{prompt_key}'. Available: {', '.join(prompts.keys())}")
            return 3

        try:
            resp = get_gemini_response(uploaded, prompts[prompt_key], args.jobdesc)
            print("\n===== MODEL OUTPUT =====\n")
            print(resp)
            return 0
        except Exception as e:
            print("Error while calling Gemini:", e)
            return 4


def start_streamlit_ui():
    # This function will be invoked only if streamlit is installed.
    st.set_page_config(page_title="ATS Resume Screener", page_icon=":robot_face:")
    st.header("ATS Resume Screener & Career Assistant")

    input_text = st.text_area("Enter Job Description", key='input')
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"]) 

    if uploaded_file is not None:
        st.success("✅ Resume uploaded successfully!")

    col1, col2, col3 = st.columns(3)

    with col1:
        btn_eval = st.button("Tell Me About the Resume")
        btn_rewrite = st.button("Rewrite Resume (ATS Format)")

    with col2:
        btn_improve = st.button("Improve My Skills")
        btn_questions = st.button("Interview Prep Q&A")

    with col3:
        btn_match = st.button("Percentage Match")

    def run(prompt_key, title):
        if uploaded_file is not None and input_text:
            with st.spinner(title + "..."):
                response = get_gemini_response(uploaded_file, prompts[prompt_key], input_text)
                st.subheader(title)
                st.write(response)
        else:
            st.error("Please upload a resume and enter a job description.")

    if btn_eval: run("evaluation", "📄 Resume Evaluation")
    if btn_improve: run("skills_improvement", "🎯 Skill Improvement Suggestions")
    if btn_match: run("ats_match", "📊 ATS Match Result")
    if btn_rewrite: run("rewrite_resume", "📝 ATS-Optimized Resume")
    if btn_questions: run("interview_questions", "🎤 Interview Preparation Q&A")


#########################
# Unit tests            #
#########################

class FakeState:
    def __init__(self, name):
        self.name = name

class FakeFile:
    def __init__(self, name, state_name='DONE'):
        self.name = name
        self.display_name = name
        self.state = FakeState(state_name)

class FakeResponse:
    def __init__(self, text):
        self.text = text

class GetGeminiResponseTests(unittest.TestCase):
    def setUp(self):
        # Create a fake uploaded file with a .read() method and .name attribute
        self.uploaded = io.BytesIO(b"%PDF-1.4 fake pdf content")
        self.uploaded.name = "test_resume.pdf"
        self.prompt = "Test prompt"
        self.jd = "Test job description"

        # Patch genai calls
        patcher_model = mock.patch('google.generativeai.GenerativeModel')
        self.addCleanup(patcher_model.stop)
        self.mock_model_cls = patcher_model.start()

        # Instance returned by GenerativeModel()
        self.mock_model = mock.Mock()
        self.mock_model_cls.return_value = self.mock_model

        # Mock generate_content to return a FakeResponse
        self.mock_model.generate_content.return_value = FakeResponse("mocked response text")

        # Patch upload_file, get_file, delete_file
        patcher_upload = mock.patch('google.generativeai.upload_file')
        self.addCleanup(patcher_upload.stop)
        self.mock_upload = patcher_upload.start()

        patcher_get_file = mock.patch('google.generativeai.get_file')
        self.addCleanup(patcher_get_file.stop)
        self.mock_get_file = patcher_get_file.start()

        patcher_delete = mock.patch('google.generativeai.delete_file')
        self.addCleanup(patcher_delete.stop)
        self.mock_delete = patcher_delete.start()

        # configure upload_file to return a FakeFile in PROCESSING first, then DONE
        file_processing = FakeFile("remote1", state_name='PROCESSING')
        file_done = FakeFile("remote1", state_name='DONE')
        # upload_file returns processing file
        self.mock_upload.return_value = file_processing
        # get_file will return file_done after a poll
        self.mock_get_file.side_effect = [file_done]

    def test_get_gemini_response_success(self):
        resp_text = get_gemini_response(self.uploaded, self.prompt, self.jd, wait_sleep=0)
        self.assertEqual(resp_text, "mocked response text")
        # Ensure model.generate_content was called with labeled job description
        args = self.mock_model.generate_content.call_args[0][0]
        self.assertIn(self.prompt, args)
        self.assertIn('Job Description:', args[-1])

    def test_processing_failed_raises(self):
        # Make upload_file return a FAILED state file
        failed_file = FakeFile('remote_failed', state_name='FAILED')
        self.mock_upload.return_value = failed_file
        # get_file will return the failed file too
        self.mock_get_file.side_effect = [failed_file]

        with self.assertRaises(ValueError):
            get_gemini_response(self.uploaded, self.prompt, self.jd, wait_sleep=0)


if __name__ == '__main__':
    # If Streamlit is present, run the app. Otherwise, expose a CLI.
    if st is not None and os.getenv('FORCE_CLI') is None:
        start_streamlit_ui()
    else:
        parser = argparse.ArgumentParser(description='ATS Resume Screener CLI (fallback)')
        parser.add_argument('--resume', required=False, help='Path to resume PDF')
        parser.add_argument('--jobdesc', required=False, default='', help='Job description text')
        parser.add_argument('--action', required=False, default='evaluation', help='Action to run: evaluation, skills_improvement, ats_match, rewrite_resume, interview_questions')
        parser.add_argument('--run-tests', action='store_true', help='Run unit tests')
        args = parser.parse_args()

        if args.run_tests:
            # Run unit tests
            unittest.main(argv=[sys.argv[0]])
        elif args.resume:
            sys.exit(run_cli(args))
        else:
            print("No Streamlit installation detected. Use --resume and --jobdesc to run the CLI, or set FORCE_CLI=0 to force Streamlit UI if available.")
            print("To run unit tests: python this_file.py --run-tests")
            sys.exit(0)




#########################
# CLI / Streamlit glue  #
#########################

def run_cli(args):
    # Read resume bytes from file
    if not os.path.exists(args.resume):
        print(f"Resume file not found: {args.resume}")
        return 2

    with open(args.resume, "rb") as f:
        class UploadedFile:
            def __init__(self, buf, name):
                self._buf = buf
                self.name = name
            def read(self):
                return self._buf

        pdf_bytes = f.read()
        uploaded = UploadedFile(pdf_bytes, os.path.basename(args.resume))

        prompt_key = args.action
        if prompt_key not in prompts:
            print(f"Unknown action '{prompt_key}'. Available: {', '.join(prompts.keys())}")
            return 3

        try:
            resp = get_gemini_response(uploaded, prompts[prompt_key], args.jobdesc)
            print("\n===== MODEL OUTPUT =====\n")
            print(resp)
            return 0
        except Exception as e:
            print("Error while calling Gemini:", e)
            return 4


def start_streamlit_ui():
    # This function will be invoked only if streamlit is installed.
    st.set_page_config(page_title="ATS Resume Screener", page_icon=":robot_face:")
    st.header("ATS Resume Screener & Career Assistant")

    input_text = st.text_area("Enter Job Description", key='input')
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"]) 

    if uploaded_file is not None:
        st.success("✅ Resume uploaded successfully!")

    col1, col2, col3 = st.columns(3)

    with col1:
        btn_eval = st.button("Tell Me About the Resume")
        btn_rewrite = st.button("Rewrite Resume (ATS Format)")

    with col2:
        btn_improve = st.button("Improve My Skills")
        btn_questions = st.button("Interview Prep Q&A")

    with col3:
        btn_match = st.button("Percentage Match")

    def run(prompt_key, title):
        if uploaded_file is not None and input_text:
            with st.spinner(title + "..."):
                response = get_gemini_response(uploaded_file, prompts[prompt_key], input_text)
                st.subheader(title)
                st.write(response)
        else:
            st.error("Please upload a resume and enter a job description.")

    if btn_eval: run("evaluation", "📄 Resume Evaluation")
    if btn_improve: run("skills_improvement", "🎯 Skill Improvement Suggestions")
    if btn_match: run("ats_match", "📊 ATS Match Result")
    if btn_rewrite: run("rewrite_resume", "📝 ATS-Optimized Resume")
    if btn_questions: run("interview_questions", "🎤 Interview Preparation Q&A")


#########################
# Unit tests            #
#########################

class FakeState:
    def __init__(self, name):
        self.name = name

class FakeFile:
    def __init__(self, name, state_name='DONE'):
        self.name = name
        self.display_name = name
        self.state = FakeState(state_name)

class FakeResponse:
    def __init__(self, text):
        self.text = text

class GetGeminiResponseTests(unittest.TestCase):
    def setUp(self):
        # Create a fake uploaded file with a .read() method and .name attribute
        self.uploaded = io.BytesIO(b"%PDF-1.4 fake pdf content")
        self.uploaded.name = "test_resume.pdf"
        self.prompt = "Test prompt"
        self.jd = "Test job description"

        # Patch genai calls
        patcher_model = mock.patch('google.generativeai.GenerativeModel')
        self.addCleanup(patcher_model.stop)
        self.mock_model_cls = patcher_model.start()

        # Instance returned by GenerativeModel()
        self.mock_model = mock.Mock()
        self.mock_model_cls.return_value = self.mock_model

        # Mock generate_content to return a FakeResponse
        self.mock_model.generate_content.return_value = FakeResponse("mocked response text")

        # Patch upload_file, get_file, delete_file
        patcher_upload = mock.patch('google.generativeai.upload_file')
        self.addCleanup(patcher_upload.stop)
        self.mock_upload = patcher_upload.start()

        patcher_get_file = mock.patch('google.generativeai.get_file')
        self.addCleanup(patcher_get_file.stop)
        self.mock_get_file = patcher_get_file.start()

        patcher_delete = mock.patch('google.generativeai.delete_file')
        self.addCleanup(patcher_delete.stop)
        self.mock_delete = patcher_delete.start()

        # configure upload_file to return a FakeFile in PROCESSING first, then DONE
        file_processing = FakeFile("remote1", state_name='PROCESSING')
        file_done = FakeFile("remote1", state_name='DONE')
        # upload_file returns processing file
        self.mock_upload.return_value = file_processing
        # get_file will return file_done after a poll
        self.mock_get_file.side_effect = [file_done]

    def test_get_gemini_response_success(self):
        resp_text = get_gemini_response(self.uploaded, self.prompt, self.jd, wait_sleep=0)
        self.assertEqual(resp_text, "mocked response text")
        # Ensure model.generate_content was called with labeled job description
        args = self.mock_model.generate_content.call_args[0][0]
        self.assertIn(self.prompt, args)
        self.assertIn('Job Description:', args[-1])

    def test_processing_failed_raises(self):
        # Make upload_file return a FAILED state file
        failed_file = FakeFile('remote_failed', state_name='FAILED')
        self.mock_upload.return_value = failed_file
        # get_file will return the failed file too
        self.mock_get_file.side_effect = [failed_file]

        with self.assertRaises(ValueError):
            get_gemini_response(self.uploaded, self.prompt, self.jd, wait_sleep=0)


if __name__ == '__main__':
    # If Streamlit is present, run the app. Otherwise, expose a CLI.
    if st is not None and os.getenv('FORCE_CLI') is None:
        start_streamlit_ui()
    else:
        parser = argparse.ArgumentParser(description='ATS Resume Screener CLI (fallback)')
        parser.add_argument('--resume', required=False, help='Path to resume PDF')
        parser.add_argument('--jobdesc', required=False, default='', help='Job description text')
        parser.add_argument('--action', required=False, default='evaluation', help='Action to run: evaluation, skills_improvement, ats_match, rewrite_resume, interview_questions')
        parser.add_argument('--run-tests', action='store_true', help='Run unit tests')
        args = parser.parse_args()

        if args.run_tests:
            # Run unit tests
            unittest.main(argv=[sys.argv[0]])
        elif args.resume:
            sys.exit(run_cli(args))
        else:
            print("No Streamlit installation detected. Use --resume and --jobdesc to run the CLI, or set FORCE_CLI=0 to force Streamlit UI if available.")
            print("To run unit tests: python this_file.py --run-tests")
            sys.exit(0)
