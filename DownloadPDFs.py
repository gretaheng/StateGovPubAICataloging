import os
import csv
import requests

# Define the API URL and headers
rurl = "https://api.scaleserp.com/search?api_key=[YOUR API KEY]&q=site%3Ahttps%3A%2F%2Fcalsta.ca.gov%2F+filetype%3Apdf&num=100"
headers = {
    'User-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582"
}

# Make the API request
response = requests.get(rurl, headers=headers)
page = response.json()

print(response)
print("API Response:", page)

# Create a list to hold PDF information
pdfl = []

# Extract information from the response
for d in page.get("organic_results", []):
    k = {}
    if "title" in d:
        k["title"] = d["title"]
    if "date" in d:
        k["date"] = d["date"]
    if "link" in d:
        k["link"] = d["link"]
    if "domain" in d:
        k["domain"] = d["domain"]
    pdfl.append(k)

# Create a directory to save PDFs
pdf_directory = "pdfs"
if not os.path.exists(pdf_directory):
    os.makedirs(pdf_directory)

# Download each PDF
for pdf in pdfl:
    pdf_url = pdf.get("link")
    if pdf_url:
        # Get the PDF name from the URL
        pdf_name = os.path.join(pdf_directory, pdf_url.split("/")[-1])
        
        try:
            # Download the PDF
            pdf_response = requests.get(pdf_url, headers=headers)
            if pdf_response.status_code == 200:
                with open(pdf_name, 'wb') as f:
                    f.write(pdf_response.content)
                print(f"Downloaded: {pdf_name}")
            else:
                print(f"Failed to download: {pdf_url} (Status Code: {pdf_response.status_code})")
        except Exception as e:
            print(f"Error downloading {pdf_url}: {e}")

# Save the results to a CSV file
csv_file_path = "pdf_info.csv"
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['link', 'title', 'domain', 'date']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()
    
    # Write the data
    for pdf in pdfl:
        writer.writerow(pdf)

print("PDF download process completed.")
print(f"Results saved to {csv_file_path}.")