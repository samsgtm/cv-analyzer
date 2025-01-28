import streamlit as st
import pandas as pd
import PyPDF2
import docx
import anthropic
import json
from io import BytesIO

st.set_page_config(page_title="CV Analyzer", layout="wide")

def read_file_content(uploaded_file):
    """Extract text from uploaded file"""
    text = ""
    try:
        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif uploaded_file.name.endswith('.docx'):
            doc = docx.Document(BytesIO(uploaded_file.read()))
            text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
        else:  # Assume text file
            text = uploaded_file.getvalue().decode()
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {str(e)}")
    return text

def analyze_cv(client, cv_text: str) -> dict:
    """Get CV analysis from Claude"""
    prompt = f"""
    Analyze this CV for a finance role with the following requirements:
    - Strong background in finance and economics
    - Experience in M&A or target identification
    - Strong analytical skills
    - Excel proficiency required
    - Python/SQL skills are bonus
    - Must show autonomy and ownership
    - Should be enthusiastic about learning
    - Should demonstrate low ego and team orientation
    - Location preference for UK-based candidates
    
    Provide analysis in JSON format with:
    {{
        "location": {{"is_uk": boolean, "location_details": string}},
        "skills": {{
            "finance_economics": float (0-10),
            "analytical": float (0-10),
            "excel": float (0-10),
            "python_sql": float (0-10),
            "identified_skills": [string]
        }},
        "experience": {{
            "years_relevant": float,
            "ma_experience": float (0-10),
            "leadership": float (0-10),
            "autonomy_indicators": [string]
        }},
        "cultural_fit": {{
            "learning_orientation": float (0-10),
            "impact_driven": float (0-10),
            "team_orientation": float (0-10),
            "supporting_evidence": [string]
        }},
        "overall_score": float (0-100),
        "key_strengths": [string],
        "potential_concerns": [string]
    }}

    CV Text:
    {cv_text}
    """
    
    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.content)
    except Exception as e:
        st.error(f"Error in Claude analysis: {str(e)}")
        return None

def main():
    st.title("CV Analyzer for Finance Roles")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input("Anthropic API Key", type="password")
        st.markdown("""
        ### Instructions:
        1. Enter your Anthropic API key
        2. Upload CVs (PDF, DOCX, or TXT)
        3. Click 'Analyze CVs'
        4. Download results
        """)

    # Main content
    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar")
        return

    # File upload
    uploaded_files = st.file_uploader(
        "Upload CVs (PDF, DOCX, or TXT)", 
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt']
    )

    if uploaded_files and st.button("Analyze CVs"):
        client = anthropic.Client(api_key=api_key)
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {file.name}...")
            cv_text = read_file_content(file)
            result = analyze_cv(client, cv_text)
            
            if result:
                result['filename'] = file.name
                results.append(result)
            
            progress_bar.progress((i + 1) / len(uploaded_files))

        if results:
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Split UK and non-UK candidates
            uk_mask = df['location'].apply(lambda x: x['is_uk'])
            uk_candidates = df[uk_mask].sort_values('overall_score', ascending=False)
            non_uk_candidates = df[~uk_mask].sort_values('overall_score', ascending=False)
            
            # Create Excel file in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                uk_candidates.to_excel(writer, sheet_name='UK Candidates', index=False)
                non_uk_candidates.to_excel(writer, sheet_name='Non-UK Candidates', index=False)
            
            # Offer download
            st.download_button(
                label="Download Analysis Results",
                data=output.getvalue(),
                file_name="cv_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Display summary
            st.header("Analysis Summary")
            
            # UK Candidates
            st.subheader("UK Candidates")
            for _, row in uk_candidates.iterrows():
                with st.expander(f"{row['filename']} - Score: {row['overall_score']:.1f}/100"):
                    st.write("Key Strengths:", ", ".join(row['key_strengths']))
                    st.write("Skills:")
                    skills = row['skills']
                    cols = st.columns(4)
                    cols[0].metric("Finance/Economics", f"{skills['finance_economics']}/10")
                    cols[1].metric("Analytical", f"{skills['analytical']}/10")
                    cols[2].metric("Excel", f"{skills['excel']}/10")
                    cols[3].metric("Python/SQL", f"{skills['python_sql']}/10")
            
            # Non-UK Candidates
            st.subheader("Non-UK Candidates")
            for _, row in non_uk_candidates.iterrows():
                with st.expander(f"{row['filename']} - Score: {row['overall_score']:.1f}/100"):
                    st.write("Location:", row['location']['location_details'])
                    st.write("Key Strengths:", ", ".join(row['key_strengths']))
                    st.write("Skills:")
                    skills = row['skills']
                    cols = st.columns(4)
                    cols[0].metric("Finance/Economics", f"{skills['finance_economics']}/10")
                    cols[1].metric("Analytical", f"{skills['analytical']}/10")
                    cols[2].metric("Excel", f"{skills['excel']}/10")
                    cols[3].metric("Python/SQL", f"{skills['python_sql']}/10")

if __name__ == "__main__":
    main()
