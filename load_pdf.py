import fitz  # this is PyMuPDF, fitz is just its internal name

def load_pdf(pdf_path):
    chunks = []  # empty list to store all our chunks
    
    doc = fitz.open(pdf_path)  # open the PDF file
    
    for page_num in range(len(doc)):  # go through every page one by one
        page = doc[page_num]  # get the current page
        text = page.get_text()  # extract the text from that page
        
        # split the page text into chunks of roughly 300 words
        words = text.split()  # split text into individual words
        
        for i in range(0, len(words), 150):  # jump 150 words at a time
            chunk = " ".join(words[i:i+150])  # join 150 words back into a string
            
            if chunk.strip():  # only save if chunk is not empty
                chunks.append({
                    "text": chunk,          # the actual text
                    "page": page_num + 1    # the page number (starts from 1)
                })
    
    print(f"Total chunks created: {len(chunks)}")
    return chunks
