# smart-invoice-
A Hybrid AI System for Automated Invoice Processing and Expense Analytics
Smart Invoice is an intelligent, end-to-end invoice processing system that converts raw invoice documents into structured, searchable, and analyzable financial data. It integrates LayoutLM Transformer, EasyOCR, Streamlit, SQLite, and Plotly to automate extraction, cleaning, storage, and visualization of invoice details, eliminating the inefficiencies and errors associated with manual data entry.

Features
Hybrid AI Extraction
LayoutLM (Document Question–Answering) is used to extract structured fields such as vendor name, invoice date, totals, and tax values.
EasyOCR performs full-text extraction for improved robustness.

The system extracts:
Vendor
Invoice Number
Invoice Date
Subtotal
IGST / CGST / SGST
Grand Total
Expense Category (using keyword-based heuristics)

Streamlit Web Application
Simple user interface for uploading invoices (PDF, JPG, PNG)
Editable data table for human verification and corrections
Real-time dashboard for financial analytics

SQLite Database Integration
All validated invoice data is saved for long-term storage
SQLAlchemy ORM is used for secure and structured database interaction

Interactive Analytics with Plotly
Time-based expenditure patterns
Category-wise expense distribution
Vendor-wise spending analysis
Detailed GST breakdown

System Architecture
The system workflow consists of five primary phases:

Initialization:
Loads the LayoutLM model
Initializes EasyOCR
Connects to the SQLite database

File Preprocessing:
Converts PDF files into images using pdf2image
Converts all files into a standardized RGB format for processing

Hybrid AI Parsing:
LayoutLM answers a predefined set of questions extracted from the invoice
EasyOCR generates the full text of the document
A heuristic module assigns the most likely spending category

Post-Processing;
Cleans and normalizes numeric fields using regular expressions
Normalizes date formats into a unified standard
Computes GST totals where required

Storage and Analytics:
Users verify extracted information through an interactive editor
Data is stored in SQLite after confirmation
The dashboard presents aggregated analytics and visual insights

Tech Stack:
AI and OCR
   LayoutLM (HuggingFace Transformers)
   EasyOCR
Backend and Logic:
  Python
   SQLAlchemy
   Regular Expressions, dateutil
   pdf2image
Frontend
   Streamlit
   Plotly Express
Database
   SQLite
  
Installation and Setup
1. Clone the Repository
   git clone https://github.com/yourusername/smart-invoice.git
   cd smart-invoice
2. Install Dependencies
   pip install -r requirements.txt
3. Run the Application
   streamlit run app.py
4. Upload an Invoice and Access the Dashboard


Future Enhancements
Short-Term Improvements
   Line-item table extraction for detailed itemized invoices
   Machine learning–based categorization using fine-tuned BERT
   Fine-tuning the LayoutLM model for improved field accuracy
Long-Term Vision: ERP Integration
   Automated flow from Sales Order to Production, Inventory, Procurement, and Invoice Matching
   Automated three-way matching (Purchase Order, Goods Received Note, Vendor Invoice)
   Complete workflow automation suitable for MSMEs

Team Members
Abdul Sami	
Vaibhav Chadha	
Aashi Phulera	
Avadesh Kumar

Guided By
Dr. Himani Sharma
Assistant Professor
School of Computing
DIT University, Dehradun
