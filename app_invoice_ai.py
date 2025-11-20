# app_invoice_ai.py — v3.5
import os, io, tempfile, re
from datetime import datetime
from dateutil import parser as dateparser
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from transformers import pipeline 
import torch
import easyocr 
from sqlalchemy import create_engine, text
import plotly.express as px

#  Page + Theme 
st.set_page_config(page_title="Smart Invoice & Expense Manager", page_icon="💼", layout="wide")
st.markdown("""
<style>
/* --- VIBRANT BACKGROUND --- */
.stApp {
    background: linear-gradient(135deg, #0a0f1b 0%, #111827 30%, #37306B 70%, #03C9D7 100%);
    background-attachment: fixed;
}
.block-container { padding-top: 1rem; }

/* --- "GLASS" CARD DESIGN --- */
.card {
    background: rgba(11, 18, 32, 0.85); /* Deep blue, semi-transparent */
    border: 1px solid #03C9D7; /* Vibrant Teal border */
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 16px 40px rgba(0,0,0,.35);
    backdrop-filter: blur(5px); /* The "glass" effect */
    -webkit-backdrop-filter: blur(5px);
}
.h { font-size:22px; font-weight:700; color:#e5e7eb; margin-bottom:12px;}
.sub { color:#9CA3AF; font-size:12px;}
.pill { background:#1f2937; border:1px solid #374151; padding:6px 10px; border-radius:999px; font-size:12px; color:#cbd5e1; }
hr { border:0; border-top:1px solid #1f2937; margin: 18px 0; }

/* --- MODERN TABS DESIGN --- */
[data-testid="stTabs"] {
    background: rgba(11, 18, 32, 0.7);
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #1f2937;
}
[data-testid="stTabs"] button[role="tab"] {
    background: #1f2937;
    color: #9CA3AF;
    border: 1px solid #374151;
    border-radius: 8px;
    margin: 0 5px;
    transition: all 0.3s ease;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: #03C9D7; /* Vibrant Teal */
    color: #111827; /* Dark text on bright button */
    border: 1px solid #03C9D7;
    font-weight: 700;
    box-shadow: 0 0 15px rgba(3, 201, 215, 0.5);
}

/* --- KPI METRIC STYLE --- */
[data-testid="stMetric"] {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 10px 14px;
}
[data-testid="stMetric"] > div { margin-bottom: 0; }
[data-testid="stMetric"] .st-bd { font-size: 28px; }
[data-testid="stMetric"] label {
    font-size: 15px;
    font-weight: 600;
    color: #9CA3AF;
}
</style>
""", unsafe_allow_html=True)

st.title("💼 Smart Invoice & Expense Manager")
st.caption("Using Deep Learning (LayoutLM) → structured data → analytics & exports")

# Cache DL Models 
@st.cache_resource
def get_pipeline():
    """Loads the Transformer AI model into memory."""
    return pipeline('document-question-answering')

@st.cache_resource
def get_reader():
    """Loads the EasyOCR model into memory."""
    return easyocr.Reader(['en'])

try:
    p = get_pipeline()
    reader = get_reader()
    st.toast("AI models loaded successfully.")
except Exception as e:
    st.error(f"Fatal Error: Could not load AI models. {e}")
    st.stop()


#  DB 
try:
    engine = create_engine("sqlite:///invoices.db", echo=False)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT, vendor TEXT, invoice_no TEXT, order_id TEXT,
                invoice_date TEXT, subtotal REAL, tax REAL, total REAL,
                igst REAL, cgst REAL, sgst REAL,
                category TEXT, created_at TEXT
            )
        """))
except Exception as e:
    st.error(f"Failed to initialize database: {e}")

#  Helpers 
def parse_amount(s: Optional[str]) -> Optional[float]:
    if not s: return None
    s_cleaned = re.sub(r"[₹,$,€,£,\s]", "", s.split('\n')[0])
    try:
        return float(re.findall(r"[\d\.]+", s_cleaned)[0])
    except Exception:
        return None

def norm_date(s: Optional[str]) -> str:
    if not s or s == "Not Found": return ""
    try:
        return dateparser.parse(s.split('\n')[0], dayfirst=True).date().isoformat()
    except Exception:
        try:
            return dateparser.parse(s.split('\n')[0], dayfirst=False).date().isoformat()
        except Exception:
            return ""

def to_images(uploaded_file: Any) -> List[Image.Image]:
    ext = os.path.splitext(uploaded_file.name.lower())[-1]
    images = []
    if ext in [".png",".jpg",".jpeg",".bmp",".tiff",".webp"]:
        images = [Image.open(uploaded_file).convert("RGB")]
    elif ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
        try:
            images = convert_from_path(tmp.name, dpi=200)
        except Exception as e:
            st.error("PDF Processing Error: Poppler not found. See installation instructions.")
            st.exception(e)
            return []
        finally:
            os.remove(tmp.name)
    return images

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "invoice_date" in df.columns:
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce").dt.date
    for col in ["subtotal","tax","igst","cgst","sgst","total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return out.getvalue()

def best_category(text: str) -> str:
    """Categorizes an invoice based on keywords."""
    rules = {
        "Travel": ["flight","uber","ola","train","taxi","travel","cab", "lyft", "airline"],
        "Food & Stay": ["hotel","restaurant","food","meal","cafe","zomato","swiggy", "starbucks"],
        "Office Supplies": ["stationery","paper","printer","ink","toner","office", "staples"],
        "Utilities": ["electricity","internet","wifi","gas","broadband", "hydro", "telecom"],
        "Services": ["maintenance","subscription","consulting", "aws", "gcp", "azure", "saas"],
        "Hardware": ["computer", "laptop", "server", "ssd", "hardware", "usb", "adapter", "cable", "mouse", "keyboard", "monitor", "lan"]
    }
    t = text.lower()
    for cat, keys in rules.items():
        if any(k in t for k in keys): return cat
    return "General"

#  DEEP LEARNING PARSER (HYBRID)
def parse_invoice_with_ai(image: Image.Image, ocr_reader: easyocr.Reader) -> Dict[str, Any]:
    """
    Uses a hybrid AI approach:
    1. Transformer (LayoutLM) for Question-Answering.
    2. EasyOCR for full-text scan for categorization.
    """
    
    questions = {
        "vendor": "Who is the vendor or seller?",
        "invoice_no": "What is the invoice number?",
        "order_id": "What is the order ID?",
        "invoice_date": "What is the invoice date?",
        "subtotal": "What is the subtotal or net amount?",
        "total": "What is the grand total or total amount due?",
        "tax": "What is the total tax amount (e.g., VAT, GST)?",
        "igst": "What is the IGST amount?",
        "cgst": "What is the CGST amount?",
        "sgst": "What is the SGST amount?",
    }
    
    # Step 1: Transformer Q&A 
    answers = {}
    for key, question_text in questions.items():
        try:
            result = p(image=image, question=question_text)
            if result and isinstance(result, list) and result[0]:
                answers[key] = result[0]['answer']
            else:
                answers[key] = None
        except Exception as e:
            st.error(f"Error during AI inference: {e}")
            answers[key] = None
            
    data = {}
    data["vendor"] = answers["vendor"].split('\n')[0] if answers["vendor"] else "Unknown"
    data["invoice_no"] = answers["invoice_no"] if answers["invoice_no"] else "Not Found"
    data["order_id"] = answers["order_id"] if answers["order_id"] else "Not Found"
    
    data["invoice_date"] = norm_date(answers["invoice_date"])
    data["subtotal"] = parse_amount(answers["subtotal"])
    data["total"] = parse_amount(answers["total"])
    data["igst"] = parse_amount(answers["igst"])
    data["cgst"] = parse_amount(answers["cgst"])
    data["sgst"] = parse_amount(answers["sgst"])
    
    tax_sum = sum(x for x in [data['igst'], data['cgst'], data['sgst']] if x)
    data["tax"] = parse_amount(answers["tax"]) if parse_amount(answers["tax"]) else (tax_sum if tax_sum > 0 else None)
    
    # Step 2: OCR for Categorization 
    try:
        img_np = np.array(image.convert('L'))
        results = ocr_reader.readtext(img_np, detail=0, paragraph=True)
        full_text = "\n".join(results)
        data["category"] = best_category(full_text)
    except Exception as e:
        st.warning(f"Categorization failed: {e}")
        data["category"] = "General"
    
    return data

#  Sidebar
st.sidebar.header("⚙️ Options")
auto_save = st.sidebar.checkbox("Auto-save to DB", True)

#  TABS (MODERN DESIGN) 
tab1, tab2, tab3 = st.tabs(["🔍 **Analyze**", "📊 **Dashboard**", "🗃️ **Records**"])

# Analyze 
with tab1:
    st.markdown('<div class="card"><div class="h">Upload Invoice (PDF / Image)</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(" ", type=["pdf","png","jpg","jpeg"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        pages = to_images(uploaded)
        recs = []
        st.info(f"Analyzing **{len(pages)}** page(s) from {uploaded.name}...")
        my_bar = st.progress(0, text="Analyzing pages...")
        
        for i, page in enumerate(pages, start=1):
            with st.spinner(f"AI is reading page {i}/{len(pages)}... (This may take a moment)"):
                r = parse_invoice_with_ai(page, reader)
                if r:
                    if r.get("invoice_no", "Not Found") != "Not Found" and r.get("total") is not None:
                        r["file_name"] = f"{uploaded.name} (Page {i})"
                        r["created_at"] = datetime.now().isoformat(timespec="seconds")
                        recs.append(r)
            my_bar.progress(i / len(pages), text=f"Analyzed page {i}")

        my_bar.empty()
        st.success(f"Found **{len(recs)}** valid invoice(s) across {len(pages)} page(s).")
        
        if recs:
            df = normalize_dataframe(pd.DataFrame(recs))
            st.markdown('<div class="card" style="margin-top: 20px;"><div class="h">Extracted Data (Editable)</div>', unsafe_allow_html=True)
            edited = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "invoice_date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "total": st.column_config.NumberColumn("Total", format="%.2f"),
                    "tax": st.column_config.NumberColumn("Tax", format="%.2f"),
                }
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.download_button("⬇️ CSV", edited.to_csv(index=False).encode("utf-8"), "invoice_results.csv", use_container_width=True)
            with c2: st.download_button("⬇️ Excel", to_excel_bytes(edited), "invoice_results.xlsx", use_container_width=True)
            with c3:
                if st.button("💾 Save to DB", use_container_width=True):
                    try:
                        with engine.begin() as conn: edited.to_sql("invoices", conn, if_exists="append", index=False)
                        st.success("✅ Saved to database!")
                    except Exception as e: st.error(f"DB Error: {e}")
            with c4:
                if auto_save:
                    try:
                        with engine.begin() as conn: edited.to_sql("invoices", conn, if_exists="append", index=False)
                        st.toast("✅ Auto-saved to DB.")
                    except Exception as e: st.error(f"DB Error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("The AI could not find any valid invoices in the document.")

#  DASHBOARD (v3.5 - NEW FEATURES & FIXES) 
with tab2:
    st.markdown('<div class="card"><div class="h">Business Expense Dashboard</div>', unsafe_allow_html=True)
    try:
        with engine.begin() as conn:
            dbdf = pd.read_sql("SELECT * FROM invoices", conn)
    except Exception as e:
        st.error(f"Could not load dashboard data: {e}")
        dbdf = pd.DataFrame()

    if dbdf.empty:
        st.write("No records found. Upload and save invoices to build your dashboard.")
    else:
        dbdf["invoice_date"] = pd.to_datetime(dbdf["invoice_date"], errors="coerce")
        # --- NEW: Convert all relevant columns to numeric, fill NaNs ---
        for col in ["total", "tax", "subtotal", "igst", "cgst", "sgst"]:
            dbdf[col] = pd.to_numeric(dbdf[col], errors="coerce").fillna(0)
        
        st.markdown("### 🗓️ Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            all_vendors = sorted(dbdf.vendor.dropna().unique().tolist())
            vendor = st.selectbox("Filter by Vendor", ["All"] + all_vendors, key="d_vendor")
        with col2:
            all_cats = sorted(dbdf.category.dropna().unique().tolist())
            cat = st.selectbox("Filter by Category", ["All"] + all_cats, key="d_cat")
        with col3:
            min_date, max_date = dbdf["invoice_date"].min(), dbdf["invoice_date"].max()
            if min_date is not pd.NaT and max_date is not pd.NaT:
                date_range = st.date_input("Filter by Date Range", [min_date, max_date], min_value=min_date, max_value=max_date, key="d_date")
            else: date_range = []

        f = dbdf.copy()
        if vendor != "All": f = f[f.vendor == vendor]
        if cat != "All": f = f[f.category == cat]
        if len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range)
            f = f[(f["invoice_date"] >= start_date) & (f["invoice_date"] <= end_date)]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📊 Key Metrics")
        
        if not f.empty:
            total_spend, avg_spend = f["total"].sum(), f["total"].mean()
            count_invoices = len(f)
            #  GST LOGIC 
            total_gst = f['tax'].sum()
            total_subtotal = f['subtotal'].sum()
            largest_invoice = f["total"].max()
            top_vendor = f.groupby("vendor")["total"].sum().nlargest(1)
            unique_vendors = f['vendor'].nunique()
        else:
            total_spend = avg_spend = count_invoices = total_gst = largest_invoice = total_subtotal = unique_vendors = 0
            top_vendor = pd.Series(dtype='float64')

        # --- NEW KPI LAYOUT ---
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Spend", f"₹{total_spend:,.2f}")
        k2.metric("Total Invoices", f"{count_invoices}")
        k3.metric("Avg. Invoice", f"₹{avg_spend:,.2f}")
        
        k4, k5, k6 = st.columns(3)
        k4.metric("Total GST Paid", f"₹{total_gst:,.2f}")
        k5.metric("Total Subtotal", f"₹{total_subtotal:,.2f}")
        k6.metric("Largest Invoice", f"₹{largest_invoice:,.2f}")
        
        k7, k8 = st.columns(2)
        if not top_vendor.empty:
            k7.metric("Top Vendor", f"{top_vendor.index[0]} (₹{top_vendor.values[0]:,.0f})")
        else: k7.metric("Top Vendor", "N/A")
        k8.metric("Unique Vendors", f"{unique_vendors}")


        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📈 Visualizations")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("<div class='h' style='font-size:18px;'>Top 5 Vendors by Spend</div>", unsafe_allow_html=True)
            if not f.empty:
                vendor_spend = f.groupby("vendor")["total"].sum().nlargest(5).sort_values(ascending=True)
                fig_bar = px.bar(vendor_spend, x='total', y=vendor_spend.index, orientation='h', text='total',
                                 template="plotly_dark", color=vendor_spend.index,
                                 color_discrete_sequence=px.colors.sequential.Plasma_r)
                fig_bar.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                      yaxis_title=None, xaxis_title="Total Spend", showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            else: st.write("No vendor data to display.")

        with c2:
            st.markdown("<div class='h' style='font-size:18px;'>Spend by Category</div>", unsafe_allow_html=True)
            if not f.empty:
                cat_total = f[f['category'] != 'General'].groupby("category")["total"].sum().reset_index()
                if cat_total.empty:
                    st.write("No spend data for specific categories.")
                else:
                    fig_tree = px.treemap(cat_total, path=['category'], values='total',
                                          template="plotly_dark", color='category',
                                          color_discrete_sequence=px.colors.qualitative.Vivid)
                    fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_tree, use_container_width=True)
            else: st.write("No category data to display.")
        
        # --- NEW ROW FOR CHARTS ---
        c3, c4 = st.columns(2)

        with c3:
            st.markdown("<div class='h' style='font-size:18px; margin-top: 20px;'>Spend Over Time</div>", unsafe_allow_html=True)
            f_time = f.dropna(subset=["invoice_date"])
            if not f_time.empty and len(f_time) > 1:
                month_total = f_time.set_index("invoice_date").resample('M')["total"].sum().reset_index()
                month_total['invoice_date'] = month_total['invoice_date'].dt.strftime('%Y-%m')
                fig_area = px.area(month_total, x='invoice_date', y='total', markers=True,
                                   template="plotly_dark", color_discrete_sequence=['#7F00FF'])
                # --- THIS IS THE FIX ---
                fig_area.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                       xaxis_title=None, yaxis_title="Total Spend")
                # --- END FIX ---
                st.plotly_chart(fig_area, use_container_width=True)
            else: st.write("Not enough data to display time-series trend.")
        
        with c4:
            st.markdown("<div class='h' style='font-size:18px; margin-top: 20px;'>GST Component Breakdown</div>", unsafe_allow_html=True)
            if total_gst > 0:
                gst_data = pd.DataFrame({
                    'GST Type': ['IGST', 'CGST', 'SGST'],
                    'Amount': [f['igst'].sum(), f['cgst'].sum(), f['sgst'].sum()]
                })
                gst_data = gst_data[gst_data['Amount'] > 0] # Only show types with value
                fig_pie = px.pie(gst_data, names='GST Type', values='Amount',
                                 template="plotly_dark", hole=0.4,
                                 color_discrete_sequence=px.colors.sequential.Teal)
                fig_pie.update_traces(textinfo='percent+label')
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.write("No GST data to display.")
            
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📄 Recent Invoices (Filtered)")
        if not f.empty:
            recent = f.nlargest(10, "invoice_date")[["invoice_date", "vendor", "invoice_no", "category", "total"]]
            recent["invoice_date"] = recent["invoice_date"].dt.strftime('%Y-%m-%d')
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else: st.write("No invoices in the selected filter.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Records ----------
with tab3:
    st.markdown('<div class="card"><div class="h">All Saved Records</div>', unsafe_allow_html=True)
    try:
        with engine.begin() as conn:
            saved = pd.read_sql("SELECT * FROM invoices ORDER BY id DESC", conn)
        st.dataframe(saved, use_container_width=True, hide_index=True)
        r1, r2, r_spacer = st.columns([1, 1, 4])
        with r1: st.download_button("⬇️ Export CSV", saved.to_csv(index=False).encode("utf-8"), "all_invoices.csv", use_container_width=True)
        with r2: st.download_button("⬇️ Export Excel", to_excel_bytes(saved), "all_invoices.xlsx", use_container_width=True)
    except Exception as e:
        st.error(f"Could not load records from database: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><span class='sub'>Powered by LayoutLM (Deep Learning). Stable v3.5 build.</span>", unsafe_allow_html=True)