import os
import pandas as pd
import re
from PyPDF2 import PdfReader
from urllib.parse import urlparse
from pathlib import Path
from difflib import SequenceMatcher

# Get user inputs
CSV_PATH = input("Enter the path to the CSV file: ").strip()
PDF_FOLDER = input("Enter the path to the PDF folder: ").strip()
DEFAULT_AUTHOR = input("Enter the default author name: ").strip()
OUTPUT_PATH = input("Enter output path for KBART file (include filename.csv): ").strip()

# Configuration
DEFAULT_ACCESS_TYPE = "free"
SIMILARITY_THRESHOLD = 0.35

KBART_FIELDS = [
    'publication_title', 'title_url', 'first_author', 'access_type', 'print_identifier',
    'online_identifier', 'date_first_issue_online',
    'num_first_vol_online', 'num_first_issue_online', 'date_last_issue_online',
    'num_last_vol_online', 'num_last_issue_online', 'coverage_depth',
    'publication_type', 'date_monograph_published_print',
    'date_monograph_published_online', 'embargo_info'
]


def clean_filename(name):
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower())
    name = re.sub(r'\b(v\d+|draft|final|version)\b', '', name)
    return re.sub(r'\s+', ' ', name).strip()


def format_title(title):
    if not title or title.lower().strip() in ("", "unknown", "unknown title"):
        return None  # Use None for proper NaN handling

    words = title.strip().split()
    if not words:
        return None

    formatted = []
    for i, word in enumerate(words):
        if len(word) >= 2 and word.isupper():
            formatted.append(word)
        else:
            processed = word.lower()
            if i == 0:
                processed = processed[0].upper() + processed[1:]
            formatted.append(processed)
    return ' '.join(formatted)


def find_matching_pdf(url):
    try:
        url_path = urlparse(url).path
        clean_url = clean_filename(Path(url_path).name)

        best_match = None
        best_score = 0

        for pdf_name in os.listdir(PDF_FOLDER):
            if not pdf_name.endswith(".pdf"):
                continue

            clean_pdf = clean_filename(pdf_name)
            current_score = SequenceMatcher(None, clean_url, clean_pdf).ratio()

            if current_score > best_score:
                best_score = current_score
                best_match = pdf_name

        if best_score >= SIMILARITY_THRESHOLD:
            return os.path.join(PDF_FOLDER, best_match)
        return None

    except Exception as e:
        print(f"Matching error: {str(e)[:50]}")
        return None


def generate_kbart_csv():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Missing {CSV_PATH}")
        return

    valid_data = []
    invalid_data = []

    for _, row in df.iterrows():
        url = row.get('link', '')
        if not url:
            continue

        pdf_path = find_matching_pdf(url)
        if not pdf_path:
            continue

        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                metadata = reader.metadata or {}
                raw_title = metadata.get('/Title', row.get('title', ''))
                formatted_title = format_title(raw_title)

                # Create base record
                record = {
                    'publication_title': formatted_title,
                    'title_url': url,
                    'first_author': DEFAULT_AUTHOR,
                    'access_type': DEFAULT_ACCESS_TYPE,
                    'print_identifier': '', 'online_identifier': '',
                    'date_first_issue_online': '', 'num_first_vol_online': '',
                    'num_first_issue_online': '', 'date_last_issue_online': '',
                    'num_last_vol_online': '', 'num_last_issue_online': '',
                    'coverage_depth': '', 'publication_type': '',
                    'date_monograph_published_print': '',
                    'date_monograph_published_online': '', 'embargo_info': ''
                }

                # Validate title
                if formatted_title is None:
                    record['publication_title'] = ''
                    invalid_record = record.copy()
                    invalid_record['Reason'] = 'Unknown title'
                    invalid_data.append(invalid_record)
                elif raw_title.isupper():
                    invalid_record = record.copy()
                    invalid_record['Reason'] = 'All caps title'
                    invalid_data.append(invalid_record)
                else:
                    valid_data.append(record)

        except Exception as e:
            print(f"Processing error: {str(e)[:50]}")
            continue

    # Save valid records
    if valid_data:
        try:
            pd.DataFrame(valid_data, columns=KBART_FIELDS).to_csv(OUTPUT_PATH, index=False)
            print(f"\nSuccess: Valid records saved to {OUTPUT_PATH}")
        except PermissionError:
            print("Error: Close existing CSV file first")
    else:
        print("Warning: No valid records processed")

    # Save invalid records with reason
    if invalid_data:
        try:
            output_path = Path(OUTPUT_PATH)
            invalid_path = output_path.parent / (output_path.stem + "_invalid.csv")

            # Create DataFrame with KBART fields + Reason
            invalid_df = pd.DataFrame(invalid_data, columns=KBART_FIELDS + ['Reason'])
            invalid_df.to_csv(invalid_path, index=False)
            print(f"Invalid records saved to: {invalid_path}")
        except Exception as e:
            print(f"Error saving invalid records: {str(e)}")
    else:
        print("No invalid records found")


if __name__ == "__main__":
    generate_kbart_csv()
