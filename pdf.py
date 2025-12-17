import fitz  # PyMuPDF

def extract(pdf_file, output_txt):
    doc = fitz.open(pdf_file)

    raw_text = ""

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")  # plain text mode
        if text:
            raw_text += f"\n\n--- Page {page_number} ---\n\n"
            raw_text += text

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(raw_text)

    doc.close()
    return raw_text


pdf = extract(
    "Introduction_to_Python_Programming_-_WEB.pdf",
    "Introduction_to_Python_Programming_-_WEB_pdf.txt"
)

print("Extraction completed and saved to file.")
