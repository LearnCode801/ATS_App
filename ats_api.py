import os
import io
import base64
import time
import json
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import pdf2image
import fitz  # PyMuPDF
import google.generativeai as genai
import re

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

# FastAPI app
app = FastAPI(title="ATS Resume Analyzer", 
              description="An AI-powered ATS resume analyzer that provides detailed feedback and scoring.",
              version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models
class AnalysisCategory(str, Enum):
    CONTENT = "content"
    FORMAT = "format"
    SECTIONS = "sections"
    SKILLS = "skills"
    STYLE = "style"

class Issue(BaseModel):
    message: str
    severity: str = "error"  # error, warning, info

class CategoryAnalysis(BaseModel):
    category: AnalysisCategory
    score: int
    issues_found: int
    issues: List[Issue] = []
    analysis: Optional[str] = None
    suggestions: Optional[List[str]] = None

class JobMatchAnalysis(BaseModel):
    overall_score: int
    ats_parse_rate: int
    categories: List[CategoryAnalysis]
    match_percentage: int

class ATSOnlyAnalysis(BaseModel):
    overall_score: int
    ats_parse_rate: int
    categories: List[CategoryAnalysis]
    potential_improvement: Optional[int] = None

# Enhanced ReAct Chain Prompts

# Step 1: Initial Extraction and Structure Analysis (shared by both modes)
STEP1_SYSTEM_PROMPT = """
You are an expert ATS (Applicant Tracking System) resume analyzer in the first stage of analysis. 
Your task is to extract all information from the resume and analyze its structure.

Perform a detailed extraction of:
1. Basic information (name, contact, etc.)
2. Education details
3. Work experience
4. Skills and qualifications
5. Structure and format analysis

Return your analysis in this exact JSON format:
```json
{
  "extraction": {
    "basic_info": {
      "name": "<extracted name>",
      "contact": "<extracted contact info>",
      "linkedin": "<extracted linkedin>",
      "other_links": "<any other links>"
    },
    "education": [
      {
        "degree": "<degree type>",
        "institution": "<institution name>",
        "dates": "<date range>",
        "details": "<additional details>"
      }
    ],
    "experience": [
      {
        "title": "<job title>",
        "company": "<company name>",
        "dates": "<date range>",
        "responsibilities": ["<responsibility 1>", "<responsibility 2>", "..."]
      }
    ],
    "skills": ["<skill 1>", "<skill 2>", "..."],
    "certifications": ["<cert 1>", "<cert 2>", "..."],
    "additional_sections": ["<section name 1>", "<section name 2>", "..."]
  },
  "structure_analysis": {
    "has_clear_sections": true/false,
    "formatting_consistency": true/false,
    "ats_readable": true/false,
    "issues": [
      {
        "type": "<issue type>",
        "description": "<issue description>"
      }
    ]
  },
  "ats_parse_rate": <percentage successfully parsed>
}
```
"""

# Step 2: Job Description Matching Analysis
STEP2_JOB_MATCHING_PROMPT = """
You are an expert ATS (Applicant Tracking System) resume analyzer in the second stage of analysis.
You will analyze how well the resume matches the job description.

Use this first-stage analysis: {{FIRST_STAGE_ANALYSIS}}

Now analyze the job description requirements carefully and compare with the resume content:
1. Identify key required skills in the job description
2. Map resume skills to job requirements
3. Identify experience relevance to the position
4. Find education match and gaps
5. Detect missing keywords and phrases

Return your analysis in this exact JSON format:
```json
{
  "job_requirements": {
    "key_skills": ["<skill 1>", "<skill 2>", "..."],
    "experience_needed": ["<experience 1>", "<experience 2>", "..."],
    "education_requirements": ["<education 1>", "<education 2>", "..."],
    "other_requirements": ["<other 1>", "<other 2>", "..."]
  },
  "matching_analysis": {
    "skills_match": {
      "matching": ["<skill 1>", "<skill 2>", "..."],
      "missing": ["<skill 1>", "<skill 2>", "..."],
      "match_percentage": <percentage>
    },
    "experience_match": {
      "relevant_experiences": ["<experience 1>", "<experience 2>", "..."],
      "missing_experiences": ["<experience 1>", "<experience 2>", "..."],
      "match_percentage": <percentage>
    },
    "education_match": {
      "meets_requirements": true/false,
      "gaps": ["<gap 1>", "<gap 2>", "..."],
      "match_percentage": <percentage>
    },
    "keyword_analysis": {
      "present": ["<keyword 1>", "<keyword 2>", "..."],
      "missing": ["<keyword 1>", "<keyword 2>", "..."]
    }
  },
  "overall_match_percentage": <percentage>
}
```
"""

# Step 2 Alternative: ATS Optimization Analysis
STEP2_ATS_ONLY_PROMPT = """
You are an expert ATS (Applicant Tracking System) resume analyzer focusing on ATS optimization.
Your task is to analyze the resume for ATS compatibility and provide detailed optimization feedback.

Use this first-stage analysis: {{FIRST_STAGE_ANALYSIS}}

Perform a detailed ATS optimization analysis:
1. Evaluate formatting for ATS compatibility
2. Assess keyword optimization and density
3. Check for proper section labels and organization
4. Look for problematic elements (tables, images, headers/footers)
5. Analyze readability by ATS systems

Return your analysis in this exact JSON format:
```json
{
  "ats_optimization": {
    "format_issues": [
      {
        "issue": "<formatting issue>",
        "impact": "<impact on ATS parsing>",
        "recommendation": "<recommendation to fix>"
      }
    ],
    "keyword_analysis": {
      "well_optimized_sections": ["<section 1>", "<section 2>", "..."],
      "poorly_optimized_sections": ["<section 1>", "<section 2>", "..."],
      "recommendations": ["<recommendation 1>", "<recommendation 2>", "..."]
    },
    "section_analysis": {
      "well_structured": ["<section 1>", "<section 2>", "..."],
      "needs_improvement": ["<section 1>", "<section 2>", "..."],
      "missing_important": ["<section 1>", "<section 2>", "..."]
    },
    "problematic_elements": [
      {
        "element": "<problematic element>",
        "issue": "<why it's problematic>",
        "solution": "<how to fix it>"
      }
    ],
    "readability_score": <percentage>,
    "overall_ats_effectiveness": <percentage>
  }
}
```
"""

# Step 3: Comprehensive Feedback for Job Matching
STEP3_JOB_MATCHING_PROMPT = """
You are an expert ATS (Applicant Tracking System) resume analyzer in the final stage of analysis.
Based on the previous analyses, provide comprehensive feedback and scoring.

Use first-stage analysis: {{FIRST_STAGE_ANALYSIS}}
Use second-stage analysis: {{SECOND_STAGE_ANALYSIS}}

Now provide a detailed evaluation across all categories with actionable feedback:
1. Calculate scores for each category based on previous analyses
2. Identify specific issues with severity ratings
3. Provide detailed analysis for each category
4. Generate specific improvement suggestions
5. Calculate overall score

Format your response following this exact structure:
```json
{
  "overall_score": <score from 0-100>,
  "ats_parse_rate": <percentage successfully parsed>,
  "categories": [
    {
      "category": "content",
      "score": <score from 0-100>,
      "issues_found": <number of issues>,
      "issues": [
        {
          "message": "<specific issue>",
          "severity": "<error/warning/info>"
        }
      ],
      "analysis": "<detailed analysis>",
      "suggestions": ["<improvement 1>", "<improvement 2>", "..."]
    },
    {
      "category": "format",
      "score": <score from 0-100>,
      "issues_found": <number of issues>,
      "issues": [
        {
          "message": "<specific issue>",
          "severity": "<error/warning/info>"
        }
      ],
      "analysis": "<detailed analysis>",
      "suggestions": ["<improvement 1>", "<improvement 2>", "..."]
    },
    {
      "category": "sections",
      "score": <score from 0-100>,
      "issues_found": <number of issues>,
      "issues": [
        {
          "message": "<specific issue>",
          "severity": "<error/warning/info>"
        }
      ],
      "analysis": "<detailed analysis>",
      "suggestions": ["<improvement 1>", "<improvement 2>", "..."]
    },
    {
      "category": "skills",
      "score": <score from 0-100>,
      "issues_found": <number of issues>,
      "issues": [
        {
          "message": "<specific issue>",
          "severity": "<error/warning/info>"
        }
      ],
      "analysis": "<detailed analysis>",
      "suggestions": ["<improvement 1>", "<improvement 2>", "..."]
    },
    {
      "category": "style",
      "score": <score from 0-100>,
      "issues_found": <number of issues>,
      "issues": [
        {
          "message": "<specific issue>",
          "severity": "<error/warning/info>"
        }
      ],
      "analysis": "<detailed analysis>",
      "suggestions": ["<improvement 1>", "<improvement 2>", "..."]
    }
  ],
  "match_percentage": <percentage match from stage 2>
}
```
"""

# Step 3 Alternative: ATS-Only Comprehensive Feedback
STEP3_ATS_ONLY_PROMPT = """
You are an expert ATS (Applicant Tracking System) resume analyzer in the final stage of analysis.
Based on the previous analyses, provide comprehensive ATS optimization feedback and scoring.

Use first-stage analysis: {{FIRST_STAGE_ANALYSIS}}
Use ATS optimization analysis: {{SECOND_STAGE_ANALYSIS}}

Now provide a detailed evaluation across all categories with actionable feedback focused on ATS optimization:
1. Calculate scores for each category based on previous analyses
2. Identify specific issues with severity ratings
3. Provide detailed analysis for each category
4. Generate specific ATS optimization suggestions
5. Calculate overall ATS-readiness score

Format your response following this exact structure:
```json
{
  "overall_score": <score from 0-100>,
  "ats_parse_rate": <percentage successfully parsed>,
  "categories": [
    {
      "category": "content",
      "score": <score from 0-100>,
      "issues_found": <number of issues>,
      "issues": [
        {
          "message": "<specific issue>",
          "severity": "<error/warning/info>"
        }
      ],
      "analysis": "<detailed analysis>",
      "suggestions": ["<improvement 1>", "<improvement 2>", "..."]
    },
    {
      "category": "format",
      "score": <score from 0-100>,
      "issues_found": <number of issues>,
      "issues": [
        {
          "message": "<specific issue>",
          "severity": "<error/warning/info>"
        }
      ],
      "analysis": "<detailed analysis>",
      "suggestions": ["<improvement 1>", "<improvement 2>", "..."]
    },
    {
      "category": "sections",
      "score": <score from 0-100>,
      "issues_found": <number of issues>,
      "issues": [
        {
          "message": "<specific issue>",
          "severity": "<error/warning/info>"
        }
      ],
      "analysis": "<detailed analysis>",
      "suggestions": ["<improvement 1>", "<improvement 2>", "..."]
    },
    {
      "category": "skills",
      "score": <score from 0-100>,
      "issues_found": <number of issues>,
      "issues": [
        {
          "message": "<specific issue>",
          "severity": "<error/warning/info>"
        }
      ],
      "analysis": "<detailed analysis>",
      "suggestions": ["<improvement 1>", "<improvement 2>", "..."]
    },
    {
      "category": "style",
      "score": <score from 0-100>,
      "issues_found": <number of issues>,
      "issues": [
        {
          "message": "<specific issue>",
          "severity": "<error/warning/info>"
        }
      ],
      "analysis": "<detailed analysis>",
      "suggestions": ["<improvement 1>", "<improvement 2>", "..."]
    }
  ],
  "potential_improvement": <percentage potential improvement>
}
```
"""

def extract_text_from_pdf(pdf_bytes):
    """Extract text from all pages of a PDF"""
    text = ""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

def convert_pdf_to_images(pdf_bytes, max_pages=5):
    """Convert PDF pages to images, limited to max_pages"""
    try:
        images = pdf2image.convert_from_bytes(pdf_bytes)
        # Limit to max_pages
        return images[:max_pages]
    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        return []

def prepare_vision_content(pdf_bytes, job_description=None, prompt=None, include_images=True):
    """Prepare content for Gemini Vision API"""
    # Extract text
    text_content = extract_text_from_pdf(pdf_bytes)
    
    # Create content parts list with prompt text first
    content_parts = [
        {
            "text": f"{prompt}\n\nResume text content: {text_content[:10000]}..."
        }
    ]
    
    # Add job description if provided
    if job_description:
        content_parts[0]["text"] += f"\n\nJOB DESCRIPTION:\n{job_description}"
    
    # Add image parts if requested (up to 3 pages)
    if include_images:
        # Convert first few pages to images
        images = convert_pdf_to_images(pdf_bytes)
        
        if not images:
            print("Warning: Failed to extract images from PDF")
        else:
            for i, img in enumerate(images[:3]):
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                content_parts.append({
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img_byte_arr).decode()
                })

    return content_parts

def parse_json_from_response(response_text):
    """Extract and parse JSON from model response"""
    try:
        # Find JSON pattern in the response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find anything that looks like JSON
            json_match = re.search(r'\{\s*"[^"]+"\s*:', response_text)
            if json_match:
                # Find opening brace
                start_pos = json_match.start()
                # Balance braces to find the end
                brace_count = 0
                end_pos = start_pos
                in_string = False
                escape_next = False
                
                for i in range(start_pos, len(response_text)):
                    char = response_text[i]
                    
                    if escape_next:
                        escape_next = False
                        continue
                        
                    if char == '\\':
                        escape_next = True
                        continue
                        
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                
                if end_pos > start_pos:
                    json_str = response_text[start_pos:end_pos]
                else:
                    raise ValueError("Could not find complete JSON in response")
            else:
                raise ValueError("Could not find JSON in response")
        
        # Parse the JSON string
        data = json.loads(json_str)
        return data
    except Exception as e:
        print(f"Error parsing JSON response: {str(e)}")
        print(f"Raw response: {response_text}")
        raise ValueError("Failed to parse analysis results")

def get_gemini_response(content_parts, temperature=0.2):
    """Get response from Gemini model with error handling and retries"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = model.generate_content(
                contents=content_parts,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            retry_count += 1
            print(f"Error calling Gemini API (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count >= max_retries:
                raise
            # Add exponential backoff
            time.sleep(2 ** retry_count)
    
    raise RuntimeError("Failed to get response from Gemini after multiple attempts")

def analyze_resume_ats_only(pdf_bytes):
    """Analyze resume using multi-step ReAct chaining for ATS-only analysis"""
    try:
        print("Starting ATS-only analysis process")
        
        # Step 1: Initial extraction and structure analysis
        print("Step 1: Initial extraction and structure analysis")
        step1_prompt = STEP1_SYSTEM_PROMPT
        step1_content = prepare_vision_content(pdf_bytes, None, step1_prompt, include_images=True)
        step1_response = get_gemini_response(step1_content, temperature=0.1)
        step1_analysis = parse_json_from_response(step1_response)
        print("Step 1 completed")
        
        # Step 2: ATS optimization analysis
        print("Step 2: ATS optimization analysis")
        step2_prompt = STEP2_ATS_ONLY_PROMPT.replace(
            "{{FIRST_STAGE_ANALYSIS}}", 
            json.dumps(step1_analysis, indent=2)
        )
            
        step2_content = prepare_vision_content(pdf_bytes, None, step2_prompt, include_images=False)
        step2_response = get_gemini_response(step2_content, temperature=0.2)
        step2_analysis = parse_json_from_response(step2_response)
        print("Step 2 completed")
        
        # Step 3: ATS optimization feedback
        print("Step 3: ATS optimization feedback")
        step3_prompt = STEP3_ATS_ONLY_PROMPT.replace(
            "{{FIRST_STAGE_ANALYSIS}}", 
            json.dumps(step1_analysis, indent=2)
        ).replace(
            "{{SECOND_STAGE_ANALYSIS}}",
            json.dumps(step2_analysis, indent=2)
        )
            
        step3_content = prepare_vision_content(pdf_bytes, None, step3_prompt, include_images=False)
        step3_response = get_gemini_response(step3_content, temperature=0.3)
        final_analysis = parse_json_from_response(step3_response)
        print("Step 3 completed")
        
        # Remove any debug info before returning to client
        client_response = {k: v for k, v in final_analysis.items() if not k.startswith('_')}
        return client_response
            
    except Exception as e:
        print(f"Error in analyze_resume_ats_only: {str(e)}")
        raise

def analyze_resume_job_matching(pdf_bytes, job_description):
    """Analyze resume using multi-step ReAct chaining for job matching analysis"""
    try:
        print("Starting job matching analysis process")
        
        # Step 1: Initial extraction and structure analysis
        print("Step 1: Initial extraction and structure analysis")
        step1_prompt = STEP1_SYSTEM_PROMPT
        step1_content = prepare_vision_content(pdf_bytes, None, step1_prompt, include_images=True)
        step1_response = get_gemini_response(step1_content, temperature=0.1)
        step1_analysis = parse_json_from_response(step1_response)
        print("Step 1 completed")
        
        # Step 2: Job description matching analysis
        print("Step 2: Job description matching analysis")
        step2_prompt = STEP2_JOB_MATCHING_PROMPT.replace(
            "{{FIRST_STAGE_ANALYSIS}}", 
            json.dumps(step1_analysis, indent=2)
        )
            
        step2_content = prepare_vision_content(pdf_bytes, job_description, step2_prompt, include_images=False)
        step2_response = get_gemini_response(step2_content, temperature=0.2)
        step2_analysis = parse_json_from_response(step2_response)
        print("Step 2 completed")
        
        # Step 3: Comprehensive feedback and scoring
        print("Step 3: Comprehensive feedback and scoring")
        step3_prompt = STEP3_JOB_MATCHING_PROMPT.replace(
            "{{FIRST_STAGE_ANALYSIS}}", 
            json.dumps(step1_analysis, indent=2)
        ).replace(
            "{{SECOND_STAGE_ANALYSIS}}",
            json.dumps(step2_analysis, indent=2)
        )
            
        step3_content = prepare_vision_content(pdf_bytes, job_description, step3_prompt, include_images=False)
        step3_response = get_gemini_response(step3_content, temperature=0.3)
        final_analysis = parse_json_from_response(step3_response)
        print("Step 3 completed")
        
        # Remove any debug info before returning to client
        client_response = {k: v for k, v in final_analysis.items() if not k.startswith('_')}
        return client_response
            
    except Exception as e:
        print(f"Error in analyze_resume_job_matching: {str(e)}")
        raise

@app.post("/analyze/job-matching/", response_model=JobMatchAnalysis)
async def api_analyze_resume_job_matching(
    resume_file: UploadFile = File(...), 
    job_description: str = Form(...)
):
    """
    Analyze a resume by matching it against a job description.
    This endpoint provides detailed feedback on how well the resume matches the specified job requirements.
    """
    # Validate file type
    if not resume_file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate job description
    if not job_description or job_description.strip() == "":
        raise HTTPException(status_code=400, detail="Job description is required")
    
    try:
        # Read file content
        pdf_bytes = await resume_file.read()
        
        # Analyze resume with job description
        start_time = time.time()
        analysis_result = analyze_resume_job_matching(pdf_bytes, job_description)
        print(f"Job matching analysis completed in {time.time() - start_time:.2f} seconds")
        
        return JSONResponse(content=analysis_result)
    
    except Exception as e:
        print(f"Error in job matching analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/ats-only/", response_model=ATSOnlyAnalysis)
async def api_analyze_resume_ats_only(
    resume_file: UploadFile = File(...)
):
    """
    Analyze a resume for ATS compatibility without job matching.
    This endpoint provides feedback on ATS optimization and readability.
    """
    # Validate file type
    if not resume_file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read file content
        pdf_bytes = await resume_file.read()
        
        # Analyze resume without job description
        start_time = time.time()
        analysis_result = analyze_resume_ats_only(pdf_bytes)
        print(f"ATS-only analysis completed in {time.time() - start_time:.2f} seconds")
        
        return JSONResponse(content=analysis_result)
    
    except Exception as e:
        print(f"Error in ATS-only analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "ATS Resume Analyzer API", 
        "version": "2.0.0",
        "endpoints": [
            {
                "path": "/analyze/job-matching/",
                "description": "Analyze resume against a specific job description"
            },
            {
                "path": "/analyze/ats-only/", 
                "description": "ATS-only optimization analysis without job description"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





