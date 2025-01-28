import streamlit as st
import pandas as pd
import PyPDF2
import docx
import anthropic
import json
from io import BytesIO
import hashlib
import plotly.graph_objects as go
from datetime import datetime
import pickle

def read_file_content(file_obj):
    """Extract text from uploaded file"""
    text = ""
    try:
        # Get file name from the BytesIO object's name if available
        file_name = getattr(file_obj, 'name', '')
        
        if file_name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file_obj)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file_name.endswith('.docx'):
            doc = docx.Document(file_obj)
            text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
        else:  # Assume text file
            text = file_obj.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
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
    
    Return ONLY a JSON string (no other text) in this exact format:
    {{
        "location": {{"is_uk": boolean, "location_details": "string"}},
        "skills": {{
            "finance_economics": number,
            "analytical": number,
            "excel": number,
            "python_sql": number,
            "identified_skills": ["skill1", "skill2"]
        }},
        "experience": {{
            "years_relevant": number,
            "ma_experience": number,
            "leadership": number,
            "autonomy_indicators": ["indicator1", "indicator2"]
        }},
        "cultural_fit": {{
            "learning_orientation": number,
            "impact_driven": number,
            "team_orientation": number,
            "supporting_evidence": ["evidence1", "evidence2"]
        }},
        "overall_score": number,
        "key_strengths": ["strength1", "strength2"],
        "potential_concerns": ["concern1", "concern2"]
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
        
        # Extract the JSON string from Claude's response
        response_content = response.content
        if isinstance(response_content, list):
            response_content = response_content[0].text
        
        # Try to parse the JSON
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse Claude's response as JSON: {str(e)}")
            st.text("Claude's response:")
            st.text(response_content)
            return None
            
    except Exception as e:
        st.error(f"Error in Claude analysis: {str(e)}")
        return None

class EnhancedCVAnalyzer:
    def __init__(self):
        # Initialize session state for persistence
        if 'processed_cvs' not in st.session_state:
            st.session_state.processed_cvs = {}
        if 'custom_weights' not in st.session_state:
            st.session_state.custom_weights = {
                'finance_economics': 1.0,
                'analytical': 1.0,
                'excel': 1.0,
                'python_sql': 0.5,
                'ma_experience': 1.0,
                'leadership': 0.8
            }
    
    def calculate_file_hash(self, file_content):
        """Generate unique hash for file to detect duplicates"""
        return hashlib.md5(file_content).hexdigest()
    
    def check_duplicate(self, file_hash):
        """Check if CV has been processed before"""
        return file_hash in st.session_state.processed_cvs
    
    def analyze_skills_gap(self, skills_dict):
        """Analyze skills gaps against requirements"""
        gaps = []
        if skills_dict['finance_economics'] < 7:
            gaps.append("Finance/Economics knowledge below threshold")
        if skills_dict['excel'] < 6:
            gaps.append("Excel proficiency needs improvement")
        if skills_dict['analytical'] < 7:
            gaps.append("Analytical skills need strengthening")
        if skills_dict.get('ma_experience', 0) < 6:
            gaps.append("Limited M&A experience")
        return gaps
    
    def detect_red_flags(self, cv_analysis):
        """Detect potential red flags in CV"""
        red_flags = []
        
        # Experience gaps
        if cv_analysis['experience']['years_relevant'] < 2:
            red_flags.append("Limited relevant experience")
        
        # Essential skills missing
        if cv_analysis['skills']['excel'] < 5:
            red_flags.append("Insufficient Excel proficiency")
        
        # Cultural fit concerns
        if cv_analysis['cultural_fit']['team_orientation'] < 5:
            red_flags.append("Potential team fit concerns")
            
        # Location considerations
        if not cv_analysis['location']['is_uk']:
            red_flags.append("Non-UK location - visa may be required")
            
        return red_flags

[... Rest of the code remains the same ...]

def main():
    st.title("Enhanced CV Analyzer")
    
    # Get API key
    api_key = st.sidebar.text_input("Enter your Anthropic API key", type="password")
    if not api_key:
        st.warning("Please enter your Anthropic API key")
        return
        
    client = anthropic.Client(api_key=api_key)
    
    analyzer = EnhancedCVAnalyzer()
    
    # Load previous session
    if st.sidebar.button("Load Previous Session"):
        if load_session_state():
            st.success("Previous session loaded successfully!")
        else:
            st.info("No previous session found")
    
    # Save current session
    if st.sidebar.button("Save Current Session"):
        if save_session_state():
            st.success("Session saved successfully!")
    
    # Custom scoring weights
    custom_scoring_ui()
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload CVs (PDF, DOCX, or TXT)",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt']
    )
    
    if uploaded_files and st.button("Analyze CVs"):
        results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            # Check for duplicates
            file_content = file.read()
            file_hash = analyzer.calculate_file_hash(file_content)
            
            if analyzer.check_duplicate(file_hash):
                st.warning(f"Duplicate CV detected: {file.name}")
                continue
            
            # Process CV
            file_obj = BytesIO(file_content)
            file_obj.name = file.name  # Add name attribute for file type detection
            cv_text = read_file_content(file_obj)
            analysis = analyze_cv(client, cv_text)
            
            if analysis:
                # Add additional analysis
                analysis['skills_gaps'] = analyzer.analyze_skills_gap(analysis['skills'])
                analysis['red_flags'] = analyzer.detect_red_flags(analysis)
                analysis['filename'] = file.name
                results.append(analysis)
                
                # Store in session state
                st.session_state.processed_cvs[file_hash] = analysis
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if results:
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Apply quick filters
            filtered_df = apply_quick_filters(df)
            
            # Display results
            st.header("Analysis Results")
            
            for _, row in filtered_df.iterrows():
                with st.expander(f"{row['filename']} - Score: {row['overall_score']:.1f}/100"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Skills:")
                        st.write(f"Finance/Economics: {row['skills']['finance_economics']}/10")
                        st.write(f"Excel: {row['skills']['excel']}/10")
                        st.write(f"Years Experience: {row['experience']['years_relevant']}")
                    
                    with col2:
                        st.write("Gaps and Flags:")
                        if row['skills_gaps']:
                            st.write("Skills Gaps:", ", ".join(row['skills_gaps']))
                        if row['red_flags']:
                            st.write("Red Flags:", ", ".join(row['red_flags']))
            
            # Export functionality
            st.header("Export Results")
            export_selected_candidates(filtered_df)

if __name__ == "__main__":
    main()
