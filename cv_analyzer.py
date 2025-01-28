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
        if skills_dict['ma_experience'] < 6:
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

def custom_scoring_ui():
    """UI for adjusting scoring weights"""
    st.sidebar.header("Customize Scoring Weights")
    
    weights = st.session_state.custom_weights
    new_weights = {}
    
    for skill, weight in weights.items():
        new_weights[skill] = st.sidebar.slider(
            f"{skill.replace('_', ' ').title()}", 
            0.0, 1.0, weight,
            help=f"Adjust importance of {skill}"
        )
    
    st.session_state.custom_weights = new_weights

def apply_quick_filters(df):
    """Apply quick filters to CV dataframe"""
    st.sidebar.header("Quick Filters")
    
    # Experience filter
    min_exp = st.sidebar.slider("Minimum Years Experience", 0, 15, 0)
    df = df[df['experience'].apply(lambda x: x['years_relevant'] >= min_exp)]
    
    # Location filter
    location_filter = st.sidebar.radio("Location", ["All", "UK Only", "Non-UK"])
    if location_filter == "UK Only":
        df = df[df['location'].apply(lambda x: x['is_uk'])]
    elif location_filter == "Non-UK":
        df = df[~df['location'].apply(lambda x: x['is_uk'])]
    
    # Skills filter
    min_finance = st.sidebar.slider("Min Finance Score", 0, 10, 0)
    min_excel = st.sidebar.slider("Min Excel Score", 0, 10, 0)
    df = df[
        (df['skills'].apply(lambda x: x['finance_economics'] >= min_finance)) &
        (df['skills'].apply(lambda x: x['excel'] >= min_excel))
    ]
    
    return df

def save_session_state():
    """Save current session state to file"""
    try:
        with open('session_state.pkl', 'wb') as f:
            pickle.dump(dict(st.session_state), f)
        return True
    except Exception as e:
        st.error(f"Error saving session: {str(e)}")
        return False

def load_session_state():
    """Load previous session state"""
    try:
        with open('session_state.pkl', 'rb') as f:
            saved_state = pickle.load(f)
            for key, value in saved_state.items():
                st.session_state[key] = value
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        st.error(f"Error loading session: {str(e)}")
        return False

def export_selected_candidates(selected_df):
    """Export selected candidates to CSV"""
    if not selected_df.empty:
        # Flatten nested dictionaries for CSV export
        export_df = pd.DataFrame()
        export_df['Filename'] = selected_df['filename']
        export_df['Overall Score'] = selected_df['overall_score']
        export_df['Location'] = selected_df['location'].apply(lambda x: x['location_details'])
        export_df['Finance Score'] = selected_df['skills'].apply(lambda x: x['finance_economics'])
        export_df['Excel Score'] = selected_df['skills'].apply(lambda x: x['excel'])
        export_df['Years Experience'] = selected_df['experience'].apply(lambda x: x['years_relevant'])
        export_df['Key Strengths'] = selected_df['key_strengths'].apply(lambda x: ', '.join(x))
        export_df['Skills Gaps'] = selected_df['skills_gaps'].apply(lambda x: ', '.join(x))
        export_df['Red Flags'] = selected_df['red_flags'].apply(lambda x: ', '.join(x))
        
        # Convert to CSV
        csv = export_df.to_csv(index=False)
        
        # Create download button
        st.download_button(
            "Download Selected Candidates",
            csv,
            "selected_candidates.csv",
            "text/csv",
            key='download-csv'
        )

def main():
    st.title("SAG CV Cruncher")
    
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
            cv_text = read_file_content(BytesIO(file_content))
            analysis = analyze_cv(cv_text)
            
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
