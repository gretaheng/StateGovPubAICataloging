import csv
import re
from urllib.parse import urlparse


# ===== Input variables =====
INPUT_FILE_PATH = "final_downloaded_pdfs.csv"
OUTPUT_FILE_PATH = "kbart_output_converted_from_final.csv"


# KBART header copied from kbart_output_with_online_dates.csv
KBART_HEADERS = [
    "publication_title",
    "title_url",
    "author",
    "access_type",
    "source_domain",
    "print_identifier",
    "online_identifier",
    "date_first_issue_online",
    "num_first_vol_online",
    "num_first_issue_online",
    "date_last_issue_online",
    "num_last_vol_online",
    "num_last_issue_online",
    "coverage_depth",
    "publication_type",
    "date_monograph_published_print",
    "date_monograph_published_online",
    "embargo_info",
]


def extract_year(date_value):
    """
    Extract year from final_downloaded_pdfs.csv date column.
    Examples:
        "2025-02-05" -> "2025"
        "February 5, 2025" -> "2025"
        "" -> ""
    """
    if not date_value:
        return ""

    match = re.search(r"\b(19|20)\d{2}\b", str(date_value))
    return match.group(0) if match else ""


def get_domain(row):
    """
    Use domain column first.
    If domain is empty, extract domain from link.
    """
    domain = row.get("domain", "").strip()
    if domain:
        return domain

    link = row.get("link", "").strip()
    if not link:
        return ""

    parsed = urlparse(link)
    return parsed.netloc


def convert_final_to_kbart(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        final_rows = list(reader)

    output_rows = []

    for row in final_rows:
        output_row = {header: "" for header in KBART_HEADERS}

        # Mapping from final_downloaded_pdfs.csv
        output_row["publication_title"] = row.get("title", "").strip()
        output_row["title_url"] = row.get("link", "").strip()
        output_row["source_domain"] = get_domain(row)

        # Default / fixed KBART values
        output_row["access_type"] = "free"
        output_row["print_identifier"] = ""

        # Date mapping: final date -> year only
        output_row["date_monograph_published_online"] = extract_year(
            row.get("date", "").strip()
        )

        output_rows.append(output_row)

    with open(output_file_path, "w", encoding="utf-8-sig", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=KBART_HEADERS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Converted {len(output_rows)} rows.")
    print(f"Output saved to: {output_file_path}")


if __name__ == "__main__":
    convert_final_to_kbart(INPUT_FILE_PATH, OUTPUT_FILE_PATH)