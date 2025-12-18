from langchain_community.document_loaders import PyPDFLoader

# Point this to your file
pdf_path = "C:/Users/kavin/Downloads/financial.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"--- DEBUG REPORT ---")
print(f"Total Pages Found: {len(docs)}")

if len(docs) > 0:
    first_page_text = docs[0].page_content
    print(f"\n--- Content of Page 1 ---")
    print(f"'{first_page_text}'")  # We put quotes to see if it's empty
    print(f"-----------------------")
    
    if not first_page_text.strip():
        print("❌ DIAGNOSIS: The page content is empty! This is likely an Image-based PDF.")
    else:
        print("✅ DIAGNOSIS: Text was found. The issue might be the text splitter settings.")
else:
    print("❌ DIAGNOSIS: No pages were loaded at all.")