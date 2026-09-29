"""
Generates a clean two-page PDF file for testing PDF ingestion and page metadata.
Uses raw PDF format specification (no external dependencies required).
"""

def create_sample_pdf(filepath: str):
    # Minimal compliant PDF 1.4 format with two pages
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"6 0 obj\n<< /Length 264 >>\nstream\n"
        b"BT\n/F1 16 Tf\n50 720 Td\n(University Student Handbook - Page 1) Tj\n"
        b"/F1 12 Tf\n0 -40 Td\n(The attendance requirement for all undergraduate programs is 75%.) Tj\n"
        b"0 -25 Td\n(Students with attendance below 75% will not be permitted to take final exams.) Tj\n"
        b"0 -25 Td\n(A minimum cumulative grade point average CGPA of 5.0 is required for graduation.) Tj\n"
        b"ET\nendstream\nendobj\n"
        b"7 0 obj\n<< /Length 262 >>\nstream\n"
        b"BT\n/F1 16 Tf\n50 720 Td\n(University Student Handbook - Page 2) Tj\n"
        b"/F1 12 Tf\n0 -40 Td\n(The Central Library is open on all working days from 8:00 AM to 10:00 PM.) Tj\n"
        b"0 -25 Td\n(Undergraduate students may borrow up to 4 books for 14 calendar days.) Tj\n"
        b"0 -25 Td\n(Hostel residents must report inside hostel premises by 9:30 PM on weekdays.) Tj\n"
        b"ET\nendstream\nendobj\n"
        b"xref\n0 8\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000124 00000 n \n"
        b"0000000251 00000 n \n"
        b"0000000378 00000 n \n"
        b"0000000456 00000 n \n"
        b"0000000773 00000 n \n"
        b"trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n1088\n%%EOF\n"
    )
    with open(filepath, "wb") as f:
        f.write(pdf_content)

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_documents/college_handbook.pdf"
    create_sample_pdf(path)
    print(f"Created sample PDF at {path}")
