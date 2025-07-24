import os
import re
import cv2
import pytesseract
import pandas as pd
import spacy
import tkinter as tk
from tkinter import filedialog, messagebox
from PyPDF2 import PdfReader

# Set up for frozen executable (PyInstaller)
import sys
if getattr(sys, 'frozen', False):
    model_path = os.path.join(sys._MEIPASS, "en_core_web_sm")
    nlp = spacy.load(model_path)
else:
    nlp = spacy.load("en_core_web_sm")
# Set the path to the Tesseract executable (update this path as needed)
import sys
import os

if getattr(sys, 'frozen', False):
    tesseract_path = os.path.join(sys._MEIPASS, "tesseract", "tesseract.exe")
else:
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Load skill lists
tech_skills = pd.read_csv("techskill.csv", header=None).values[0]
non_tech_skills = pd.read_csv("nontechnicalskills.csv", header=None).values[0]
tech_skill_descriptions = pd.read_csv("techatt.csv", header=None).values[0]

# Regex patterns
PHONE_REGEX = r'(?:\+91[-\s]?|0)?[6-9]\d{9}'
EMAIL_REGEX = r'[\w\.-]+@[\w\.-]+\.\w+'

# Add more stopwords/keywords to avoid as names (expand as needed)
NAME_STOPWORDS = set([
    "sex", "male", "female", "qualifications", "cv", "writing", "wi11", "help", "machine", "learning",
    "operating", "system", "speech", "text", "app", "aug", "ar", "summary", "objective", "skills", "address",
    "curriculum", "vitae", "resume", "b.tech", "diploma", "engineering", "projects", "experience", "qualification",
    "languages", "python", "c++", "java", "university", "institute", "school", "college", "personal", "project",
    "kendriya", "vidyalaya", "swami", "vivekananda", "galgotias", "patna", "cgpa", "professional"
])

def extract_text_from_pdf(pdf_path):
    try:
        text = ''
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ''
        return text
    except Exception as e:
        print(f"[PDF ERROR] {e}")
        return ""

def extract_text_from_image(img_path):
    try:
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return pytesseract.image_to_string(thresh)
    except Exception as e:
        print(f"[IMG ERROR] {e}")
        return ""

def extract_name(text):
    # Only scan the top 10 lines for names
    lines = [l.strip() for l in text.splitlines() if l.strip()][:10]

    # 1. Look for "Name:" or "Candidate:" patterns
    for line in lines:
        m = re.match(r'^(name|candidate)[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$', line, re.I)
        if m:
            name = m.group(2)
            if not any(x in name.lower() for x in NAME_STOPWORDS) and not re.search(r'\d', name):
                return name

    # 2. Use spaCy NER, but only for lines near the top, and filter out known bads
    for line in lines:
        doc = nlp(line)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and 2 <= len(ent.text.split()) <= 4:
                name = ent.text.strip()
                # Remove lines with stopwords, numbers, or likely non-names
                if not any(x in name.lower() for x in NAME_STOPWORDS) \
                   and not re.search(r'\d', name) \
                   and not re.search(r'@|\.com|\+91|\d{10}', name):
                    return name

    # 3. Fallback: Strict regex for two or three capitalized words (likely a name)
    for line in lines:
        if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+( [A-Z][a-z]+)?$', line) and not any(x in line.lower() for x in NAME_STOPWORDS):
            return line

    return "Not Found"

def clean_phone(text):
    # 1. Find all phone-like patterns (with or without +91, spaces, dashes)
    patterns = [
        r'(\+91[\-\s]?\d{5}[\-\s]?\d{5})',   # +91 12345 67890 or +91-12345-67890
        r'(\b[6-9]\d{9}\b)',                 # 10-digit Indian mobile
        r'(\d{3,5}[\-\s]?\d{6,8})',          # Landline or split numbers
        r'(\d{10,13})'                       # Any 10-13 digit sequence
    ]
    found = []
    for pat in patterns:
        found += re.findall(pat, text)

    # 2. Clean and validate each found number
    cleaned_numbers = set()
    for num in found:
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', num)
        # Fix common OCR errors
        digits = digits.replace('O', '0').replace('o', '0').replace('l', '1').replace('|', '1')
        # Remove leading zeros
        digits = digits.lstrip('0')
        # Accept only numbers with 10 or 12 (with 91) digits
        if len(digits) == 10 and digits.startswith(('6', '7', '8', '9')):
            cleaned_numbers.add(digits)
        elif len(digits) == 12 and digits.startswith('91') and digits[2] in '6789':
            cleaned_numbers.add(digits[2:])
        elif len(digits) == 13 and digits.startswith('091') and digits[3] in '6789':
            cleaned_numbers.add(digits[3:])

    # 3. Return the first valid phone number found
    if cleaned_numbers:
        return list(cleaned_numbers)[0]
    return "Not Found"

def normalize_email(email):
    # Fix common OCR artifacts and reject non-standard domains
    email = email.replace('c0m', 'com').replace('C0M', 'com')
    email = email.replace('1', 'l').replace('0', 'o')
    email = re.sub(r'\.{2,}', '.', email)
    email = email.strip(" -.,;:_")
    # Only allow standard domains (com, in, org, net, edu)
    if not re.search(r'\.(com|in|org|net|edu)$', email, re.I):
        return "Not Found"
    return email

def extract_email(text):
    # Try original method first
    matches = re.findall(r'\b[\w\.-]+@[\w\.-]+\.[a-z]{2,}\b', text)
    for email in matches:
        cleaned = normalize_email(email)
        if cleaned != "Not Found" and re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,4}$", cleaned):
            return cleaned
    # --- UPGRADE: fallback for missed emails ---
    # Try to catch emails with common OCR errors (spaces, missing dots, etc.)
    ocr_like = re.findall(r'[\w\.-]+[\s_]@[\s_]*[\w\.-]+[\s_]*\.[a-z]{2,}', text)
    for email in ocr_like:
        cleaned = normalize_email(email.replace(' ', '').replace('_', ''))
        if cleaned != "Not Found" and re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,4}$", cleaned):
            return cleaned
    return "Not Found"

def clean_ocr_email(email):
    # Only fix common OCR mistakes in emails
    email = email.replace('0', 'o').replace('1', 'l')
    email = re.sub(r'\.c0m\b', '.com', email)
    email = re.sub(r'gmai1', 'gmail', email)
    email = re.sub(r'yah00', 'yahoo', email)
    email = re.sub(r'h0tmail', 'hotmail', email)
    # Remove repeated 'o' or 'l' at end of domain
    email = re.sub(r'(\.com)[ol]+$', r'\1', email)
    return email

def clean_ocr_phone(phone):
    # Only fix common OCR mistakes in phone numbers
    phone = phone.replace('O', '0').replace('o', '0').replace('l', '1').replace('|', '1')
    phone = re.sub(r'[^\d+]', '', phone)
    digits = re.sub(r'\D', '', phone)
    if len(digits) < 10 or len(digits) > 13:
        return "Not Found"
    return phone

def extract_skills(text, skill_list):
    text = text.lower().replace(',', ' ')
    found = [word for word in text.split() if word in skill_list]
    return list(set(found))

def process_folder(folder_path, status_label):
    results = []
    files = os.listdir(folder_path)

    if not files:
        messagebox.showwarning("Empty Folder", "The selected folder has no files.")
        return

    for file in files:
        file_path = os.path.join(folder_path, file)
        text = ""

        if file.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file.lower().endswith((".png", ".jpg", ".jpeg")):
            text = extract_text_from_image(file_path)

        # Do NOT apply OCR corrections to the whole text!
        # Only apply to phone/email extraction

        name = extract_name_from_filename(file)
        phone = clean_phone(text)
        email = extract_email(text)
        tech = extract_skills(text, tech_skills)
        nontech = extract_skills(text, non_tech_skills)

        results.append({
            "File Name": file,
            "Name": name,
            "Phone": phone,
            "Email": email,
            "Technical Skills": ', '.join(tech),
            "Non-Technical Skills": ', '.join(nontech)
        })

    df = pd.DataFrame(results)
    output_dir = r"C:\Users\Dell\Desktop\Extracted Data sheet"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "extracted_data.xlsx")

    df.to_excel(output_path, index=False)
    status_label.config(text=f"✅ Done! Excel saved at:\n{output_path}")
    messagebox.showinfo("Success", "Extraction complete!")

def browse_folder(status_label):
    folder = filedialog.askdirectory()
    if folder:
        status_label.config(text="Processing...")
        root.update()
        process_folder(folder, status_label)

def extract_name_from_filename(filename):
    # Remove 'Resume' prefix and '.pdf' suffix (case-insensitive)
    name = filename
    if name.lower().startswith('resume'):
        name = name[6:]
    if name.lower().endswith('.pdf'):
        name = name[:-4]
    name = name.replace('_', ' ').strip()
    return name

# GUI setup
root = tk.Tk()
root.title("Resume Extractor Pro")
root.geometry("500x300")
root.resizable(False, False)

tk.Label(root, text="Resume Parser - Extract Name, Phone, Email, Skills", font=("Helvetica", 13), wraplength=480).pack(pady=20)
tk.Button(root, text="Select Folder with Resumes", font=("Helvetica", 12), command=lambda: browse_folder(status)).pack(pady=10)

status = tk.Label(root, text="", font=("Helvetica", 10), fg="green", wraplength=480)
status.pack(pady=20)

root.mainloop()
