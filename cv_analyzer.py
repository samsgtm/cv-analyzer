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
    Analyze this CV for a finance role. Extract the email address if present and analyze based on these criteria:

    Skills (45% of total score):
    - Finance and economics knowledge
    - Excel proficiency
    - Analytical capabilities

    Experience (15% of total score):
    - Autonomy and leadership
    - Relevant industry experience
    
    Cultural Fit (25% of total score):
    - Learning mindset
    - Impact-orientation
    - Collaboration/team fit
    
    Location (15% of total score):
    - UK location preferred

    Return ONLY a JSON string (no other text) in this exact format:
    {{
        "email": "string",
        "location": {{"is_uk": boolean, "location_details": "string"}},
        "skills": {{
            "finance_economics": number (0-10),
            "analytical": number (0-10),
            "excel": number (0-10),
            "python_sql": number (0-10),
            "identified_skills": ["skill1", "skill2"]
        }},
        "experience": {{
            "years_relevant": number,
            "autonomy": number (0-10),
            "industry_relevance": number (0-10),
            "key_achievements": ["achievement1", "achievement2"]
        }},
        "cultural_fit": {{
            "learning_orientation": number (0-10),
            "impact_driven": number (0-10),
            "team_orientation": number (0-10),
            "supporting_evidence": ["evidence1", "evidence2"]
        }},
        "overall_score": number (0-10),
        "key_strengths": ["strength1", "strength2"],
        "potential_concerns": ["concern1", "concern2"]
    }}

    Base the overall_score (0-10) on these weightings:
    - Skills: 45% (finance_economics, excel, analytical)
    - Experience: 15% (autonomy, industry_relevance)
    - Cultural Fit: 25% (learning_orientation, impact_driven, team_orientation)
    - Location: 15% (10 for UK, 5 for non-UK)

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

def export_selected_candidates(filtered_df):
    """Export selected candidates to CSV with enhanced information"""
    if not filtered_df.empty:
        export_df = pd.DataFrame()
        export_df['Email'] = filtered_df['email']
        export_df['Filename'] = filtered_df['filename']
        export_df['Overall Score'] = filtered_df['overall_score']
        export_df['Location'] = filtered_df['location'].apply(lambda x: x['location_details'])
        
        # Detailed skills breakdown
        export_df['Finance/Economics Score'] = filtered_df['skills'].apply(lambda x: x['finance_economics'])
        export_df['Excel Score'] = filtered_df['skills'].apply(lambda x: x['excel'])
        export_df['Analytical Score'] = filtered_df['skills'].apply(lambda x: x['analytical'])
        
        # Experience details
        export_df['Years Experience'] = filtered_df['experience'].apply(lambda x: x['years_relevant'])
        export_df['Autonomy Score'] = filtered_df['experience'].apply(lambda x: x['autonomy'])
        export_df['Industry Relevance'] = filtered_df['experience'].apply(lambda x: x['industry_relevance'])
        
        # Cultural fit scores
        export_df['Learning Orientation'] = filtered_df['cultural_fit'].apply(lambda x: x['learning_orientation'])
        export_df['Impact Driven'] = filtered_df['cultural_fit'].apply(lambda x: x['impact_driven'])
        export_df['Team Orientation'] = filtered_df['cultural_fit'].apply(lambda x: x['team_orientation'])
        
        # Additional insights
        export_df['Key Strengths'] = filtered_df['key_strengths'].apply(lambda x: ', '.join(x))
        export_df['Skills Gaps'] = filtered_df['skills_gaps'].apply(lambda x: ', '.join(x))
        export_df['Red Flags'] = filtered_df['red_flags'].apply(lambda x: ', '.join(x))
        export_df['Key Achievements'] = filtered_df['experience'].apply(lambda x: ', '.join(x['key_achievements']))
        
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

def display_results(filtered_df):
    """Enhanced results display with sorting options"""
    st.header("Analysis Results")
    
    # Sorting options
    sort_by = st.selectbox(
        "Sort candidates by:",
        ["Overall Score", "Finance/Economics", "Excel", "Analytical", "Industry Relevance", "Learning Orientation"]
    )
    
    # Create sorting mapping
    sort_mapping = {
        "Overall Score": "overall_score",
        "Finance/Economics": lambda x: x['skills']['finance_economics'],
        "Excel": lambda x: x['skills']['excel'],
        "Analytical": lambda x: x['skills']['analytical'],
        "Industry Relevance": lambda x: x['experience']['industry_relevance'],
        "Learning Orientation": lambda x: x['cultural_fit']['learning_orientation']
    }
    
    # Sort dataframe
    if sort_by in sort_mapping:
        if isinstance(sort_mapping[sort_by], str):
            filtered_df = filtered_df.sort_values(sort_mapping[sort_by], ascending=False)
        else:
            filtered_df = filtered_df.sort_values(by=sort_mapping[sort_by], ascending=False)
    
    # Display results with enhanced visualization
    for _, row in filtered_df.iterrows():
        with st.expander(f"{row['filename']} - Score: {row['overall_score']:.1f}/10"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("Contact:")
                st.write(f"Email: {row['email']}")
                st.write(f"Location: {row['location']['location_details']}")
                
                st.write("\nSkills (45%):")
                st.write(f"Finance/Economics: {row['skills']['finance_economics']}/10")
                st.write(f"Excel: {row['skills']['excel']}/10")
                st.write(f"Analytical: {row['skills']['analytical']}/10")
            
            with col2:
                st.write("Experience (15%):")
                st.write(f"Years: {row['experience']['years_relevant']}")
                st.write(f"Autonomy: {row['experience']['autonomy']}/10")
                st.write(f"Industry Relevance: {row['experience']['industry_relevance']}/10")
                
                st.write("\nCultural Fit (25%):")
                st.write(f"Learning: {row['cultural_fit']['learning_orientation']}/10")
                st.write(f"Impact: {row['cultural_fit']['impact_driven']}/10")
                st.write(f"Team: {row['cultural_fit']['team_orientation']}/10")
            
            with col3:
                st.write("Key Insights:")
                if row['key_strengths']:
                    st.write("Strengths:", ", ".join(row['key_strengths']))
                if row['skills_gaps']:
                    st.write("Gaps:", ", ".join(row['skills_gaps']))
                if row['red_flags']:
                    st.write("Flags:", ", ".join(row['red_flags']))

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
