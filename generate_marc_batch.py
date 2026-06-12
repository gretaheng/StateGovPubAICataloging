"""
Batch MARC generator for the 63 PDFs in all_pdfs/.

For every PDF:
  1. Extract metadata (language, title, year) with Apache Tika.
  2. Look up the source URL from pdf_info.csv (matched by filename).
  3. Build a base MARC record (005, 006, 007, 008, 040, 245, 264, 300, 336, 337, 338, 856).
  4. Use OpenAI ChatGPT to generate a 520 summary and up to 5 653 keywords from the PDF text.
  5. Write the record to MARC/<pdf_stem>.mrc.

Run from the project root:
    python generate_marc_batch.py
"""

import csv
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pdfplumber
import pycountry
from openai import OpenAI
from pymarc import Field, Record, Subfield

from constants import APIKEY, organizationID
from LanDic import dlan

# ----------------------------- configuration ---------------------------------

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "all_pdfs"
MARC_DIR = ROOT / "MARC"
PDF_INFO_CSV = ROOT / "pdf_info.csv"

# CSV that lists exactly the 63 PDFs to process. Falls back to PDF_INFO_CSV
# if not present.
FINAL_CSV = ROOT / "final_downloaded_pdfs.csv"

OPENAI_MODEL = "gpt-4-turbo"
# gpt-4-turbo has a 128K-token context window. 50,000 chars (~12-15K tokens)
# comfortably fits and covers almost every document in the corpus in full.
MAX_TEXT_CHARS = 50000

MARC_DIR.mkdir(exist_ok=True)

# OpenAI client (v1+ SDK). 60s per-request timeout, 2 automatic retries
# on transient errors so a slow / dropped connection won't hang the loop.
client_kwargs = {"api_key": APIKEY, "timeout": 60.0, "max_retries": 2}
if organizationID:
    client_kwargs["organization"] = organizationID
client = OpenAI(**client_kwargs)


# ----------------------------- helpers ---------------------------------------

def load_url_map():
    """Map filename -> source URL. Prefer final_downloaded_pdfs.csv (the 63-file list)
    and fall back to pdf_info.csv for any leftovers."""
    url_map = {}
    for csv_path in (FINAL_CSV, PDF_INFO_CSV):
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                link = (row.get("link") or "").strip()
                if not link:
                    continue
                fname = link.rsplit("/", 1)[-1]
                url_map.setdefault(fname, link)
    return url_map


def safe_get(meta, key, default=""):
    val = meta.get(key, default)
    if isinstance(val, list):
        val = val[0] if val else default
    return val or default


def detect_language(meta):
    """Return a 3-letter MARC language code; default to 'eng'.
    PDF metadata rarely carries a language tag, so this almost always
    returns 'eng' for the CalSTA documents.
    """
    raw_lan = (safe_get(meta, "Language", "") or safe_get(meta, "Lang", ""))
    raw_lan = raw_lan.split("-")[0][:2]
    if raw_lan:
        try:
            name = pycountry.languages.get(alpha_2=raw_lan).name
            if name in dlan:
                return dlan[name]
        except Exception:
            pass
    return "eng"


def extract_title(meta, fallback):
    """Return (title, ind2) where ind2 is the 245 non-filing-character indicator."""
    title = safe_get(meta, "Title", "") or safe_get(meta, "Subject", "")
    if title == "Acrobat Accessibility Report":
        title = ""
    title = (title or fallback).strip()
    if title:
        title = title[0].upper() + title[1:]

    ind2 = " "
    low = title.lower()
    if low.startswith("a "):
        ind2 = "2"
    elif low.startswith("an "):
        ind2 = "3"
    elif low.startswith("the "):
        ind2 = "4"
    return title, ind2


def _ensure_period(s: str) -> str:
    """Append '.' if the string doesn't already end with sentence punctuation."""
    if not s:
        return s
    return s if s[-1] in ".?!" else s + "."


def extract_year(meta):
    """Pull a 4-digit year from PDF date fields.
    PDF dates look like  D:20240612120000-07'00'  — we just grab the first 4 digits.
    """
    for key in ("CreationDate", "ModDate", "xmp:CreateDate"):
        val = safe_get(meta, key, "")
        if not val:
            continue
        # Strip leading "D:" if present and grab the year
        digits = "".join(c for c in str(val) if c.isdigit())
        if len(digits) >= 4 and digits[:4].isdigit():
            year = digits[:4]
            if 1900 <= int(year) <= 2100:
                return year
    return ""


def read_pdf(pdf_path, url_map):
    """Open the PDF with pdfplumber and extract metadata + text."""
    with pdfplumber.open(str(pdf_path)) as pdf:
        meta = dict(pdf.metadata or {})
        text_parts = []
        for page in pdf.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                text_parts.append(t)
        content = "\n".join(text_parts).strip()

    title, ind2 = extract_title(
        meta,
        fallback=pdf_path.stem.replace("_", " ").replace("-", " "),
    )
    year = extract_year(meta)
    lan = detect_language(meta)
    url = url_map.get(pdf_path.name, "")

    return {
        "filename": pdf_path.name,
        "title": title,
        "titleind2": ind2,
        "year": year,
        "lan": lan,
        "url": url,
        "content": content,
    }


def build_base_record(info):
    """Build the MARC record (everything except 520/653)."""
    record = Record()

    # Leader -- pymarc computes pos 00-04 (record length) and 12-16 (base
    # address) on serialization, so we only set the cataloger-controlled
    # positions. Target leader: "_____nam a22_____ i 4500"
    ldr = list(record.leader)        # 24-char bytearray-like
    ldr[5]  = "n"   # Record status: new
    ldr[6]  = "a"   # Type of record: language material
    ldr[7]  = "m"   # Bibliographic level: monograph
    ldr[9]  = "a"   # Character coding scheme: Unicode (UCS/Unicode)
    ldr[17] = " "   # Encoding level: full
    ldr[18] = "i"   # Descriptive cataloging form: ISBD punctuation (RDA)
    ldr[19] = " "   # Multipart resource record level: not specified
    record.leader = "".join(ldr)

    # Force the cataloging year to 2025 regardless of the system clock.
    now = datetime.now().replace(year=2025)
    marc005 = now.strftime("%Y%m%d%H%M%S") + ".0"
    record.add_field(Field(tag="005", data=marc005))
    record.add_field(Field(tag="006", data="m     o  d        "))
    # 007 for online electronic resource:
    #   00=c  01=r  02= _  03=|  04=n  05-11=|
    #   12=u (level of compression: unknown)
    #   13=u (reformatting quality: unknown)
    record.add_field(Field(tag="007", data="cr |n|||||||uu"))

    # 008 -- build by position so each fixed-field byte is documented.
    date_entered = now.strftime("%y%m%d")          # pos 00-05
    b = list(" " * 40)
    b[0:6]   = list(date_entered)
    if info["year"]:
        b[6]    = "s"                              # single known date
        b[7:11] = list(info["year"])               # date 1
    else:
        b[6]    = "n"                              # date unknown
        b[7:11] = list("uuuu")
    b[11:15] = list("    ")                        # date 2 (blank for type=s)
    b[15:18] = list("cau")                         # place: California, USA
    b[18:22] = list("    ")                        # illustrations: none
    b[22]    = " "                                 # target audience
    b[23]    = "o"                                 # form of item: ONLINE  (was blank)
    b[24:28] = list("    ")                        # nature of contents
    b[28]    = "s"                                 # gov pub: STATE  (was "f")
    b[29]    = "0"                                 # conference pub: no
    b[30]    = "0"                                 # festschrift: no
    b[31]    = "0"                                 # index: no
    b[32]    = " "                                 # undefined
    b[33]    = "0"                                 # literary form: not fiction
    b[34]    = " "                                 # biography
    b[35:38] = list(info["lan"])                   # language
    b[38]    = " "                                 # modified record
    b[39]    = "d"                                 # cataloging source: other
    marc008 = "".join(b)
    record.add_field(Field(tag="008", data=marc008))

    record.add_field(Field(
        tag="040", indicators=[" ", " "],
        subfields=[
            Subfield(code="a", value="CDS"),
            Subfield(code="b", value="eng"),
            # $e rda intentionally omitted per cataloger preference
            Subfield(code="c", value="CDS"),
        ],
    ))

    # 245 -- ind1 = "0" (no main entry / no 1XX in our records);
    # ind2 = number of non-filing characters (0-9). Ensure title $a
    # ends with ISBD terminal period.
    title_a = _ensure_period(info["title"])
    ind2 = info["titleind2"] if info["titleind2"] != " " else "0"
    record.add_field(Field(
        tag="245", indicators=["0", ind2],
        subfields=[Subfield(code="a", value=title_a)],
    ))

    # 264 -- ISBD punctuation included.
    # $a "[California] :"  $b "California State Transportation Agency,"  $c "2024."
    year_c = (info["year"] or "[date of publication not identified]") + "."
    record.add_field(Field(
        tag="264", indicators=[" ", "1"],
        subfields=[
            Subfield(code="a", value="[California] :"),
            Subfield(code="b", value="California State Transportation Agency,"),
            Subfield(code="c", value=year_c),
        ],
    ))

    record.add_field(Field(tag="300", indicators=[" ", " "],
                           subfields=[Subfield(code="a", value="1 online resource.")]))
    record.add_field(Field(tag="336", indicators=[" ", " "], subfields=[
        Subfield(code="a", value="text"),
        Subfield(code="b", value="txt"),
        Subfield(code="2", value="rdacontent"),
    ]))
    record.add_field(Field(tag="337", indicators=[" ", " "], subfields=[
        Subfield(code="a", value="computer"),
        Subfield(code="b", value="c"),
        Subfield(code="2", value="rdamedia"),
    ]))
    record.add_field(Field(tag="338", indicators=[" ", " "], subfields=[
        Subfield(code="a", value="online resource"),
        Subfield(code="b", value="cr"),
        Subfield(code="2", value="rdacarrier"),
    ]))

    if info["url"]:
        record.add_field(Field(tag="856", indicators=["4", "0"],
                               subfields=[Subfield(code="u", value=info["url"])]))
    return record


def openai_summary_and_keywords(text):
    """Return (summary_text, [keyword, ...]) from ChatGPT."""
    snippet = (text or "")[:MAX_TEXT_CHARS]
    if not snippet.strip():
        return "", []

    summary_resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system",
             "content": "Write a clear and concise summary for the provided government publication. The summary should be less than 50 words."},
            {"role": "user",
             "content": f"Write a summary of the following government publication:\n{snippet}\n\nDETAILED SUMMARY:"},
        ],
    )
    summary = summary_resp.choices[0].message.content.strip()

    kw_resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system",
             "content": ("Assign at most 5 keywords; do not number the keywords; use ',' to separate keywords; "
                         "do not add any words before the first keyword or after the last; "
                         "response should be all lowercase; no temporal keywords, only topical keywords.")},
            {"role": "user",
             "content": f"Assign at most 5 keywords for the provided government documentation:\n{snippet}"},
        ],
    )
    kw_text = kw_resp.choices[0].message.content.strip()
    keywords = [k.strip().strip(".") for k in kw_text.split(",") if k.strip()]
    return summary, keywords[:5]


def add_ai_fields(record, summary, keywords):
    if summary:
        record.add_field(Field(tag="520", indicators=[" ", " "],
                               subfields=[Subfield(code="a", value=summary)]))
    for kw in keywords:
        if kw:
            record.add_field(Field(tag="653", indicators=[" ", " "],
                                   subfields=[Subfield(code="a", value=kw)]))


# ----------------------------- main loop -------------------------------------

def process_pdf(pdf_path, url_map):
    info = read_pdf(pdf_path, url_map)
    record = build_base_record(info)
    try:
        summary, keywords = openai_summary_and_keywords(info["content"])
        add_ai_fields(record, summary, keywords)
    except Exception as exc:
        print(f"  [WARN] OpenAI call failed for {pdf_path.name}: {exc}")
    out_path = MARC_DIR / (pdf_path.stem + ".mrc")
    with open(out_path, "wb") as f:
        f.write(record.as_marc())
    return out_path


def main():
    if not PDF_DIR.exists():
        sys.exit(f"PDF folder not found: {PDF_DIR}")

    url_map = load_url_map()
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {PDF_DIR}")
    print(f"Writing MARC records to {MARC_DIR}\n")

    ok, skipped, failed = 0, 0, []
    for idx, pdf in enumerate(pdfs, 1):
        out_path = MARC_DIR / (pdf.stem + ".mrc")
        if out_path.exists():
            print(f"[{idx}/{len(pdfs)}] {pdf.name}  (already done, skipping)")
            skipped += 1
            continue
        print(f"[{idx}/{len(pdfs)}] {pdf.name}")
        try:
            out = process_pdf(pdf, url_map)
            print(f"  -> {out.name}")
            ok += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user. Re-run to resume from here.")
            break
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            traceback.print_exc()
            failed.append(pdf.name)
        # gentle rate-limit between OpenAI calls
        time.sleep(1)

    print("\n========== Summary ==========")
    print(f"Successful: {ok}")
    print(f"Skipped:    {skipped}  (already had a .mrc in MARC/)")
    print(f"Failed:     {len(failed)}")
    if failed:
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
