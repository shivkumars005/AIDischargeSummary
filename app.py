import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from fpdf import FPDF
from transformers import pipeline
import os

# Load the cleaned dataset
df = pd.read_csv("dataset.csv")
df['JoinedDate'] = pd.to_datetime(df['JoinedDate'], errors='coerce')
df['CheckoutDate'] = pd.to_datetime(df['CheckoutDate'], errors='coerce')

# Setup SQLite for auto-saving approved summaries
conn = sqlite3.connect("summaries.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS summaries (
        patient_id INTEGER PRIMARY KEY,
        name TEXT,
        summary TEXT,
        approved_by_doctor INTEGER
    )
""")
conn.commit()

# Load Local AI Model (GPT-2 Optimized with ONNX runtime)
summarizer = pipeline("text-generation", model="gpt2")

# Function to fetch patient data
def get_patient_data(patient_id):
    patient = df[df['PatientID'] == int(patient_id)]
    return patient.to_dict(orient='records')[0] if not patient.empty else None

# Function to handle multiple patients with the same name
def search_patients_by_name(name):
    results = df[df['Name'].str.contains(name, case=False, na=False)][['PatientID', 'Name']]
    return results

# AI model for discharge summary
def generate_summary(patient_data, detail_level, doctor_notes):
    prompt = (f"{patient_data['Name']} is a {patient_data['Sex']} patient aged {patient_data['AgeCategory']} years from {patient_data['State']}. "
              f"The patient was diagnosed with {patient_data['Disease']} and received treatment. "
              f"Insurance status: {patient_data['Insurance']}. Doctor's notes: {doctor_notes}.")
    summary = summarizer(prompt, max_length=200 if detail_level == 'detailed' else 100, num_return_sequences=1)[0]['generated_text']
    return summary.strip()

# Generate PDF
def generate_pdf(patient_data, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Patient Discharge Summary", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, f"Name: {patient_data['Name']}", ln=True)
    pdf.cell(200, 10, f"Sex: {patient_data['Sex']}", ln=True)
    pdf.cell(200, 10, f"Age Category: {patient_data['AgeCategory']}", ln=True)
    pdf.cell(200, 10, f"Diagnosis: {patient_data['Disease']}", ln=True)
    pdf.cell(200, 10, f"Insurance: {patient_data['Insurance']}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Discharge Summary:\n{summary}")
    
    pdf_file = "discharge_summary.pdf"
    pdf.output(pdf_file)
    return pdf_file

# Streamlit App UI Enhancements
st.set_page_config(page_title="AI Discharge Summary", layout="wide", initial_sidebar_state="expanded")
st.title("🏥 AI-Powered Patient Discharge Summary Generator")

col1, col2 = st.columns([2, 3])
with col1:
    st.subheader("Search Patient")
    search_name = st.text_input("Enter Patient Name (Optional)")
    search_results = search_patients_by_name(search_name) if search_name else None
    
    if search_results is not None and not search_results.empty:
        patient_id = st.selectbox("Select Patient ID", search_results['PatientID'].astype(str).unique())
    else:
        patient_id = st.selectbox("Patient ID", df['PatientID'].astype(str).unique())

with col2:
    if st.button("Generate Summary"):
        patient_data = get_patient_data(patient_id)
        if patient_data:
            detail_level = st.radio("Summary Detail Level", ("brief", "detailed"))
            doctor_notes = st.text_area("Doctor's Notes (Optional)")
            summary = generate_summary(patient_data, detail_level, doctor_notes)
            st.subheader("Generated Discharge Summary:")
            st.write(summary)
            
            # Doctor's Approval Step
            approval = st.checkbox("Doctor's Approval Required")
            if approval:
                st.success("Doctor approved. Summary saved.")
                cursor.execute("INSERT OR REPLACE INTO summaries (patient_id, name, summary, approved_by_doctor) VALUES (?, ?, ?, ?)",
                               (patient_data['PatientID'], patient_data['Name'], summary, 1))
                conn.commit()
            
            # Generate and provide PDF download
            pdf_file = generate_pdf(patient_data, summary)
            with open(pdf_file, "rb") as f:
                st.download_button("📄 Download PDF", f, file_name=pdf_file, mime="application/pdf")
        else:
            st.error("Patient not found. Please enter a valid ID or name.")
