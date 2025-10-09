import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import date
import requests
import re
import pandas as pd
from collections import Counter

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import our enhanced modular components
from health_api_integrator import EnhancedHealthAPIIntegrator
from health_data_processor_work import EnhancedHealthDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_get(data, *keys):
    """Safely get nested dictionary values"""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return {}
    return data

def display_batch_code_meanings_langchain(results: Dict[str, Any]) -> str:
    """
    LangChain version of batch processed code meanings display
    Returns formatted string output for console/text display
    """
    output_lines = []
    
    # Header
    output_lines.append("=" * 80)
    output_lines.append("🧠 CLAIMS DATA ANALYSIS")
    output_lines.append("=" * 80)
    
    # Get extraction results
    medical_extraction = safe_get(results, 'structured_extractions', {}).get('medical', {})
    pharmacy_extraction = safe_get(results, 'structured_extractions', {}).get('pharmacy', {})
    
    # MEDICAL CODES SECTION
    output_lines.append("\n" + "🏥 MEDICAL CODE MEANINGS ANALYSIS")
    output_lines.append("-" * 60)
    
    medical_meanings = medical_extraction.get("code_meanings", {})
    service_meanings = medical_meanings.get("service_code_meanings", {})
    diagnosis_meanings = medical_meanings.get("diagnosis_code_meanings", {})
    medical_records = medical_extraction.get("hlth_srvc_records", [])
    
    # Calculate metrics (fixed calculation)
    unique_service_codes = set()
    unique_diagnosis_codes = set()
    total_medical_records = len(medical_records)
    
    # Count unique codes from medical records
    for record in medical_records:
        service_code = record.get("hlth_srvc_cd", "")
        if service_code:
            unique_service_codes.add(service_code)
        
        for diag in record.get("diagnosis_codes", []):
            code = diag.get("code", "")
            if code:
                unique_diagnosis_codes.add(code)
    
    # Medical summary metrics
    output_lines.append(f"\nMEDICAL METRICS SUMMARY:")
    output_lines.append(f"├─ Service Codes: {len(unique_service_codes)}")
    output_lines.append(f"├─ ICD-10 Codes: {len(unique_diagnosis_codes)}")
    output_lines.append(f"├─ Medical Records: {total_medical_records}")
    batch_status = medical_extraction.get("llm_call_status", "unknown")
    output_lines.append(f"└─ Batch Status: {batch_status.upper()}")
    
    # ICD-10 DIAGNOSIS CODES TABLE
    output_lines.append("\n" + "🩺 ICD-10 DIAGNOSIS CODES")
    output_lines.append("-" * 50)
    
    if diagnosis_meanings and medical_records:
        # Prepare diagnosis data
        diagnosis_data = []
        for record in medical_records:
            claim_date = record.get("clm_rcvd_dt", "Unknown")
            record_path = record.get("data_path", "")
            
            for diag in record.get("diagnosis_codes", []):
                code = diag.get("code", "")
                if code in diagnosis_meanings:
                    diagnosis_data.append({
                        "code": code,
                        "meaning": diagnosis_meanings[code],
                        "claim_date": claim_date,
                        "position": diag.get("position", ""),
                        "source": diag.get("source", ""),
                        "record_path": record_path
                    })
        
        if diagnosis_data:
            unique_codes = len(set(item["code"] for item in diagnosis_data))
            output_lines.append(f"📊 Unique ICD-10 Codes Found: {unique_codes}")
            output_lines.append("")
            
            # Create formatted table
            output_lines.append(f"{'Code':<10} {'Meaning':<40} {'Date':<12} {'Pos':<4}")
            output_lines.append("-" * 70)
            
            # Sort by claim date
            sorted_data = sorted(diagnosis_data, key=lambda x: x['claim_date'], reverse=True)
            
            for item in sorted_data[:20]:  # Show top 20
                code = item['code'][:9]
                meaning = item['meaning'][:39]
                date = str(item['claim_date'])[:11]
                pos = str(item['position'])[:3]
                output_lines.append(f"{code:<10} {meaning:<40} {date:<12} {pos:<4}")
            
            if len(diagnosis_data) > 20:
                output_lines.append(f"... and {len(diagnosis_data) - 20} more records")
            
            # Code frequency analysis
            code_counts = Counter(item["code"] for item in diagnosis_data)
            output_lines.append(f"\n📈 Most Frequent ICD-10 Codes:")
            for code, count in code_counts.most_common(5):
                meaning = diagnosis_meanings.get(code, "Unknown")[:50]
                output_lines.append(f"  • {code} ({count}x): {meaning}")
        else:
            output_lines.append("No ICD-10 diagnosis codes found in medical records")
    else:
        output_lines.append("No ICD-10 diagnosis code meanings available")
    
    # MEDICAL SERVICE CODES TABLE
    output_lines.append("\n" + "🏥 MEDICAL SERVICE CODES")
    output_lines.append("-" * 50)
    
    if service_meanings and medical_records:
        # Prepare service data
        service_data = []
        for record in medical_records:
            service_end_date = record.get("clm_line_srvc_end_dt", "Unknown")
            service_code = record.get("hlth_srvc_cd", "")
            record_path = record.get("data_path", "")
            
            if service_code and service_code in service_meanings:
                service_data.append({
                    "code": service_code,
                    "meaning": service_meanings[service_code],
                    "end_date": service_end_date,
                    "record_path": record_path
                })
        
        if service_data:
            unique_codes = len(set(item["code"] for item in service_data))
            output_lines.append(f"📊 Unique Service Codes Found: {unique_codes}")
            output_lines.append("")
            
            # Create formatted table
            output_lines.append(f"{'Code':<12} {'Service Description':<45} {'End Date':<12}")
            output_lines.append("-" * 72)
            
            # Sort by service end date
            sorted_data = sorted(service_data, key=lambda x: x['end_date'], reverse=True)
            
            for item in sorted_data[:20]:  # Show top 20
                code = item['code'][:11]
                meaning = item['meaning'][:44]
                date = str(item['end_date'])[:11]
                output_lines.append(f"{code:<12} {meaning:<45} {date:<12}")
            
            if len(service_data) > 20:
                output_lines.append(f"... and {len(service_data) - 20} more records")
            
            # Code frequency analysis
            code_counts = Counter(item["code"] for item in service_data)
            output_lines.append(f"\n📈 Most Frequent Service Codes:")
            for code, count in code_counts.most_common(5):
                meaning = service_meanings.get(code, "Unknown")[:50]
                output_lines.append(f"  • {code} ({count}x): {meaning}")
        else:
            output_lines.append("No medical service codes found in records")
    else:
        output_lines.append("No medical service code meanings available")
    
    # PHARMACY CODES SECTION
    output_lines.append("\n" + "💊 PHARMACY CODE MEANINGS ANALYSIS")
    output_lines.append("-" * 60)
    
    pharmacy_meanings = pharmacy_extraction.get("code_meanings", {})
    ndc_meanings = pharmacy_meanings.get("ndc_code_meanings", {})
    med_meanings = pharmacy_meanings.get("medication_meanings", {})
    pharmacy_records = pharmacy_extraction.get("ndc_records", [])
    
    # Calculate pharmacy metrics
    unique_ndc_codes = set()
    unique_medications = set()
    total_pharmacy_records = len(pharmacy_records)
    
    for record in pharmacy_records:
        ndc_code = record.get("ndc", "")
        if ndc_code:
            unique_ndc_codes.add(ndc_code)
        
        med_name = record.get("lbl_nm", "")
        if med_name:
            unique_medications.add(med_name)
    
    # Pharmacy summary metrics
    output_lines.append(f"\nPHARMACY METRICS SUMMARY:")
    output_lines.append(f"├─ NDC Codes: {len(unique_ndc_codes)}")
    output_lines.append(f"├─ Unique Medications: {len(unique_medications)}")
    output_lines.append(f"├─ Pharmacy Records: {total_pharmacy_records}")
    pharmacy_batch_status = pharmacy_extraction.get("llm_call_status", "unknown")
    output_lines.append(f"└─ Batch Status: {pharmacy_batch_status.upper()}")
    
    # NDC CODES TABLE
    output_lines.append("\n" + "💊 NDC MEDICATION CODES")
    output_lines.append("-" * 50)
    
    if ndc_meanings and pharmacy_records:
        # Prepare NDC data
        ndc_data = []
        for record in pharmacy_records:
            ndc_code = record.get("ndc", "")
            fill_date = record.get("rx_filled_dt", "Unknown")
            med_name = record.get("lbl_nm", "")
            
            if ndc_code and ndc_code in ndc_meanings:
                ndc_data.append({
                    "ndc": ndc_code,
                    "meaning": ndc_meanings[ndc_code],
                    "fill_date": fill_date,
                    "med_name": med_name
                })
        
        if ndc_data:
            unique_codes = len(set(item["ndc"] for item in ndc_data))
            output_lines.append(f"📊 Unique NDC Codes Found: {unique_codes}")
            output_lines.append("")
            
            # Create formatted table
            output_lines.append(f"{'NDC Code':<15} {'Medication Description':<40} {'Fill Date':<12}")
            output_lines.append("-" * 70)
            
            # Sort by fill date
            sorted_data = sorted(ndc_data, key=lambda x: x['fill_date'], reverse=True)
            
            for item in sorted_data[:15]:  # Show top 15
                ndc = item['ndc'][:14]
                meaning = item['meaning'][:39]
                date = str(item['fill_date'])[:11]
                output_lines.append(f"{ndc:<15} {meaning:<40} {date:<12}")
            
            if len(ndc_data) > 15:
                output_lines.append(f"... and {len(ndc_data) - 15} more records")
        else:
            output_lines.append("No NDC codes found in pharmacy records")
    else:
        output_lines.append("No NDC code meanings available")
    
    # MEDICATION NAMES TABLE
    output_lines.append("\n" + "💉 MEDICATION NAMES ANALYSIS")
    output_lines.append("-" * 50)
    
    if med_meanings and pharmacy_records:
        # Prepare medication data
        med_data = []
        for record in pharmacy_records:
            med_name = record.get("lbl_nm", "")
            fill_date = record.get("rx_filled_dt", "Unknown")
            ndc = record.get("ndc", "")
            
            if med_name and med_name in med_meanings:
                med_data.append({
                    "med_name": med_name,
                    "meaning": med_meanings[med_name],
                    "fill_date": fill_date,
                    "ndc": ndc
                })
        
        if med_data:
            unique_meds = len(set(item["med_name"] for item in med_data))
            output_lines.append(f"📊 Unique Medications Found: {unique_meds}")
            output_lines.append("")
            
            # Create formatted table
            output_lines.append(f"{'Medication':<25} {'Description/Use':<35} {'Fill Date':<12}")
            output_lines.append("-" * 75)
            
            # Sort by fill date
            sorted_data = sorted(med_data, key=lambda x: x['fill_date'], reverse=True)
            
            for item in sorted_data[:15]:  # Show top 15
                med = item['med_name'][:24]
                meaning = item['meaning'][:34]
                date = str(item['fill_date'])[:11]
                output_lines.append(f"{med:<25} {meaning:<35} {date:<12}")
            
            if len(med_data) > 15:
                output_lines.append(f"... and {len(med_data) - 15} more records")
            
            # Medication frequency analysis
            med_counts = Counter(item["med_name"] for item in med_data)
            output_lines.append(f"\n📈 Most Frequent Medications:")
            for med, count in med_counts.most_common(5):
                meaning = med_meanings.get(med, "Unknown")[:40]
                output_lines.append(f"  • {med} ({count}x): {meaning}")
        else:
            output_lines.append("No medications found in pharmacy records")
    else:
        output_lines.append("No medication meanings available")
    
    # Footer
    output_lines.append("\n" + "=" * 80)
    output_lines.append("📊 CLAIMS DATA ANALYSIS COMPLETE")
    output_lines.append("=" * 80)
    
    return "\n".join(output_lines)

def print_code_meanings_langchain(results: Dict[str, Any]):
    """
    Print the formatted code meanings to console
    """
    formatted_output = display_batch_code_meanings_langchain(results)
    print(formatted_output)

def get_code_meanings_summary_langchain(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary dict of code meanings for programmatic use
    """
    medical_extraction = safe_get(results, 'structured_extractions', {}).get('medical', {})
    pharmacy_extraction = safe_get(results, 'structured_extractions', {}).get('pharmacy', {})
    
    # Calculate metrics
    medical_records = medical_extraction.get("hlth_srvc_records", [])
    pharmacy_records = pharmacy_extraction.get("ndc_records", [])
    
    unique_service_codes = set()
    unique_diagnosis_codes = set()
    unique_ndc_codes = set()
    unique_medications = set()
    
    for record in medical_records:
        service_code = record.get("hlth_srvc_cd", "")
        if service_code:
            unique_service_codes.add(service_code)
        
        for diag in record.get("diagnosis_codes", []):
            code = diag.get("code", "")
            if code:
                unique_diagnosis_codes.add(code)
    
    for record in pharmacy_records:
        ndc_code = record.get("ndc", "")
        if ndc_code:
            unique_ndc_codes.add(ndc_code)
        
        med_name = record.get("lbl_nm", "")
        if med_name:
            unique_medications.add(med_name)
    
    return {
        "medical_summary": {
            "service_codes_count": len(unique_service_codes),
            "diagnosis_codes_count": len(unique_diagnosis_codes),
            "medical_records_count": len(medical_records),
            "batch_status": medical_extraction.get("llm_call_status", "unknown")
        },
        "pharmacy_summary": {
            "ndc_codes_count": len(unique_ndc_codes),
            "medications_count": len(unique_medications),
            "pharmacy_records_count": len(pharmacy_records),
            "batch_status": pharmacy_extraction.get("llm_call_status", "unknown")
        },
        "total_summary": {
            "total_unique_codes": len(unique_service_codes) + len(unique_diagnosis_codes) + len(unique_ndc_codes),
            "total_records": len(medical_records) + len(pharmacy_records)
        }
    }

@dataclass
class Config:
    fastapi_url: str = "http://10.126.192.183:8080"  # MCP server URL
    # Snowflake Cortex API Configuration
    api_url: str = "https://sfassist.edagenaipreprod.awsdns.internal.das/api/cortex/complete"
    api_key: str = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
    app_id: str = "edadip"
    aplctn_cd: str = "edagnai"
    model: str = "llama4-maverick"
    
    # Enhanced system messages with better defined prompts
    sys_msg: str = """You are Dr. HealthAI, a comprehensive healthcare data analyst and clinical decision support specialist with expertise in:

CLINICAL SPECIALIZATION:
• Medical coding systems (ICD-10, CPT, HCPCS, NDC) interpretation and analysis
• Claims data analysis and healthcare utilization patterns
• Risk stratification and predictive modeling for chronic diseases
• Clinical decision support and evidence-based medicine
• Population health management and care coordination
• Healthcare economics and cost prediction
• Quality metrics (HEDIS, STAR ratings) and care gap analysis
• Advanced healthcare data visualization with matplotlib

DATA ACCESS CAPABILITIES:
• Complete deidentified medical claims with ICD-10 diagnosis codes and CPT procedure codes
• Complete deidentified pharmacy claims with NDC codes and medication details
• Healthcare service utilization patterns and claims dates (clm_rcvd_dt, rx_filled_dt)
• Structured extractions of all medical and pharmacy fields with detailed analysis
• Enhanced entity extraction results including chronic conditions and risk factors
• Comprehensive patient demographic and clinical data
• Batch-processed code meanings for medical and pharmacy codes

ANALYTICAL RESPONSIBILITIES:
You provide comprehensive healthcare analysis including clinical insights, risk assessments, predictive modeling, and evidence-based recommendations using ALL available deidentified claims data. Always reference specific data points, codes, dates, and clinical indicators from the provided records when making assessments.

GRAPH GENERATION CAPABILITIES:
You can generate matplotlib code for healthcare data visualizations including:
• Medication timeline charts
• Diagnosis progression timelines
• Risk assessment dashboards
• Health metrics overviews
• Condition severity distributions
• Utilization trend analysis

RESPONSE STANDARDS:
• Use clinical terminology appropriately while ensuring clarity
• Cite specific ICD-10 codes, NDC codes, CPT codes, and claim dates
• Provide evidence-based analysis using established clinical guidelines
• Include risk stratification and predictive insights
• Reference exact field names and values from the JSON data structure
• Maintain professional healthcare analysis standards
• Generate working matplotlib code when visualization is requested"""

    chatbot_sys_msg: str = """You are Dr. ChatAI, a specialized healthcare AI assistant with COMPLETE ACCESS to comprehensive deidentified medical and pharmacy claims data. You serve as a clinical decision support tool for healthcare analysis with advanced graph generation capabilities.

COMPREHENSIVE DATA ACCESS:
✅ MEDICAL CLAIMS DATA:
   • Complete deidentified medical records with ICD-10 diagnosis codes
   • Healthcare service codes (hlth_srvc_cd) and CPT procedure codes
   • Claims received dates (clm_rcvd_dt) and service utilization patterns
   • Patient demographics (age, zip code) and clinical indicators

✅ PHARMACY CLAIMS DATA:
   • Complete deidentified pharmacy records with NDC medication codes
   • Medication names (lbl_nm), prescription fill dates (rx_filled_dt)
   • Drug utilization patterns and therapy management data
   • Prescription adherence and medication history

✅ ANALYTICAL RESULTS:
   • Enhanced entity extraction with chronic condition identification
   • Health trajectory analysis with predictive insights
   • Risk assessment results including cardiovascular risk prediction
   • Clinical complexity scoring and care gap analysis
   • Batch-processed code meanings for all medical and pharmacy codes

✅ GRAPH GENERATION CAPABILITIES:
   • Generate working matplotlib code for healthcare visualizations
   • Create medication timelines, diagnosis progressions, risk dashboards
   • Support real-time chart generation and display
   • Provide complete, executable Python code with proper imports

ADVANCED CAPABILITIES:
🔬 CLINICAL ANALYSIS:
   • Interpret ICD-10 diagnosis codes for disease progression and prognosis assessment
   • Analyze NDC medication codes for treatment adherence and therapeutic effectiveness
   • Assess comorbidity burden from diagnosis patterns and medication combinations
   • Evaluate drug interactions and optimize therapeutic pathways

📊 PREDICTIVE MODELING:
   • Risk stratification for chronic diseases (diabetes, hypertension, COPD, CKD)
   • Hospitalization and readmission risk prediction (6-12 month outlook)
   • Emergency department utilization vs outpatient care patterns
   • Medication adherence risk assessment and intervention strategies
   • Healthcare cost prediction and utilization forecasting

💰 HEALTHCARE ECONOMICS:
   • High-cost claimant identification and cost projection
   • Healthcare utilization optimization (inpatient vs outpatient)
   • Care management program recommendations
   • Population health risk segmentation

🎯 QUALITY & CARE MANAGEMENT:
   • Care gap identification (missed screenings, vaccinations)
   • HEDIS and STAR rating impact assessment
   • Preventive care opportunity identification
   • Personalized care plan recommendations

📈 VISUALIZATION CAPABILITIES:
   • Generate matplotlib code for medication timeline charts
   • Create risk assessment dashboards with multiple metrics
   • Develop diagnosis progression visualizations
   • Build comprehensive health overview charts
   • Support custom visualization requests

GRAPH GENERATION PROTOCOL:
When asked to create a graph or visualization:
1. **Detect Request**: Identify graph type from user query
2. **Generate Code**: Create complete, executable matplotlib code
3. **Use Real Data**: Incorporate actual patient data when available
4. **Provide Context**: Include brief explanation of the visualization
5. **Ensure Quality**: Generate professional, informative charts

RESPONSE PROTOCOL:
1. **DATA-DRIVEN ANALYSIS**: Always use specific data from the provided claims records
2. **CLINICAL EVIDENCE**: Reference exact ICD-10 codes, NDC codes, dates, and clinical findings
3. **PREDICTIVE INSIGHTS**: Provide forward-looking analysis based on available clinical indicators
4. **ACTIONABLE RECOMMENDATIONS**: Suggest specific clinical actions and care management strategies
5. **PROFESSIONAL STANDARDS**: Maintain clinical accuracy while ensuring patient safety considerations
6. **GRAPH GENERATION**: Provide working matplotlib code when visualization is requested

GRAPH RESPONSE FORMAT:
When generating graphs, respond with:
```
[Brief explanation of what the visualization shows]

```python
[Complete matplotlib code]
```

[Clinical insights from the visualization]
```

CRITICAL INSTRUCTIONS:
• Access and analyze the COMPLETE deidentified claims dataset provided
• Reference specific codes, dates, medications, and clinical findings
• Provide comprehensive analysis using both medical AND pharmacy data
• Include predictive insights and risk stratification
• Cite exact field paths and values from the JSON data structure
• Explain medical terminology and provide clinical context
• Focus on actionable clinical insights and care management recommendations
• Generate working matplotlib code for visualization requests
• Use actual patient data in graphs when available

You have comprehensive access to this patient's complete healthcare data - use it to provide detailed, professional medical analysis, clinical decision support, and advanced data visualizations."""

    timeout: int = 30

    # Heart Attack Prediction API Configuration
    heart_attack_api_url: str = "http://10.126.192.183:8080"
    heart_attack_threshold: float = 0.5

    def to_dict(self):
        return asdict(self)

# Enhanced State Definition for LangGraph
class HealthAnalysisState(TypedDict):
    # Input data
    patient_data: Dict[str, Any]

    # API outputs
    mcid_output: Dict[str, Any]
    medical_output: Dict[str, Any]
    pharmacy_output: Dict[str, Any]
    token_output: Dict[str, Any]

    # Processed data
    deidentified_medical: Dict[str, Any]
    deidentified_pharmacy: Dict[str, Any]
    deidentified_mcid: Dict[str, Any]

    # Extracted structured data
    medical_extraction: Dict[str, Any]
    pharmacy_extraction: Dict[str, Any]

    entity_extraction: Dict[str, Any]

    # Analysis results
    health_trajectory: str
    final_summary: str

    # Heart Attack Prediction
    heart_attack_prediction: Dict[str, Any]
    heart_attack_risk_score: float
    heart_attack_features: Dict[str, Any]

    # Enhanced chatbot functionality with graph generation
    chatbot_ready: bool
    chatbot_context: Dict[str, Any]
    chat_history: List[Dict[str, str]]
    graph_generation_ready: bool

    # Code meanings tables
    code_meanings_tables: Dict[str, Any]

    # Control flow
    current_step: str
    errors: List[str]
    retry_count: int
    processing_complete: bool
    step_status: Dict[str, str]

class HealthAnalysisAgent:
    """Enhanced Health Analysis Agent with Comprehensive Clinical Analysis, Graph Generation, and Code Meanings Display"""

    def __init__(self, custom_config: Optional[Config] = None):
        self.config = custom_config or Config()

        # Initialize enhanced components
        self.api_integrator = EnhancedHealthAPIIntegrator(self.config)
        self.data_processor = EnhancedHealthDataProcessor(self.api_integrator)

        logger.info("🔧 Enhanced HealthAnalysisAgent initialized with Graph Generation and Code Meanings Display")
        logger.info(f"🌐 Snowflake API URL: {self.config.api_url}")
        logger.info(f"🤖 Model: {self.config.model}")
        logger.info(f"📡 MCP Server URL: {self.config.fastapi_url}")
        logger.info(f"❤️ Heart Attack ML API: {self.config.heart_attack_api_url}")
        logger.info(f"📊 Graph generation ready for medical data visualizations")
        logger.info(f"📋 Code meanings display integrated for LangChain")

        self.setup_enhanced_langgraph()

    def setup_enhanced_langgraph(self):
        """Setup enhanced LangGraph workflow with graph generation support"""
        logger.info("🔧 Setting up Enhanced LangGraph workflow with graph generation...")

        workflow = StateGraph(HealthAnalysisState)

        # Add all processing nodes
        workflow.add_node("fetch_api_data", self.fetch_api_data)
        workflow.add_node("deidentify_claims_data", self.deidentify_claims_data)
        workflow.add_node("extract_claims_fields", self.extract_claims_fields)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("analyze_trajectory", self.analyze_trajectory)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("generate_code_meanings_tables", self.generate_code_meanings_tables)
        workflow.add_node("predict_heart_attack", self.predict_heart_attack)
        workflow.add_node("initialize_chatbot", self.initialize_chatbot)
        workflow.add_node("handle_error", self.handle_error)

        # Define workflow edges
        workflow.add_edge(START, "fetch_api_data")

        workflow.add_conditional_edges(
            "fetch_api_data",
            self.should_continue_after_api,
            {
                "continue": "deidentify_claims_data",
                "retry": "fetch_api_data",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "deidentify_claims_data",
            self.should_continue_after_deidentify,
            {
                "continue": "extract_claims_fields",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "extract_claims_fields",
            self.should_continue_after_extraction_step,
            {
                "continue": "extract_entities",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "extract_entities",
            self.should_continue_after_entity_extraction,
            {
                "continue": "analyze_trajectory",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "analyze_trajectory",
            self.should_continue_after_trajectory,
            {
                "continue": "generate_summary",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "generate_summary",
            self.should_continue_after_summary,
            {
                "continue": "generate_code_meanings_tables",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "generate_code_meanings_tables",
            self.should_continue_after_code_meanings_tables,
            {
                "continue": "predict_heart_attack",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "predict_heart_attack",
            self.should_continue_after_heart_attack_prediction,
            {
                "continue": "initialize_chatbot",
                "error": "handle_error"
            }
        )

        workflow.add_edge("initialize_chatbot", END)
        workflow.add_edge("handle_error", END)

        # Compile with checkpointer
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)

        logger.info("✅ Enhanced LangGraph workflow compiled successfully with graph generation and code meanings display!")

    # ===== LANGGRAPH NODES =====

    def fetch_api_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 1: Fetch claims data from APIs"""
        logger.info("🚀 Node 1: Starting Claims API data fetch...")
        state["current_step"] = "fetch_api_data"
        state["step_status"]["fetch_api_data"] = "running"

        try:
            patient_data = state["patient_data"]

            # Validation
            required_fields = ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
            for field in required_fields:
                if not patient_data.get(field):
                    state["errors"].append(f"Missing required field: {field}")
                    state["step_status"]["fetch_api_data"] = "error"
                    return state

            # Fetch data
            api_result = self.api_integrator.fetch_backend_data_enhanced(patient_data)

            if "error" in api_result:
                state["errors"].append(f"Claims API Error: {api_result['error']}")
                state["step_status"]["fetch_api_data"] = "error"
            else:
                state["mcid_output"] = api_result.get("mcid_output", {})
                state["medical_output"] = api_result.get("medical_output", {})
                state["pharmacy_output"] = api_result.get("pharmacy_output", {})
                state["token_output"] = api_result.get("token_output", {})

                state["step_status"]["fetch_api_data"] = "completed"
                logger.info("✅ Successfully fetched all Claims API data")

        except Exception as e:
            error_msg = f"Error fetching Claims API data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["fetch_api_data"] = "error"
            logger.error(error_msg)

        return state

    def deidentify_claims_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 2: Deidentification of claims data"""
        logger.info("🔒 Node 2: Starting comprehensive claims data deidentification...")
        state["current_step"] = "deidentify_claims_data"
        state["step_status"]["deidentify_claims_data"] = "running"

        try:
            # Deidentify all data types
            medical_data = state.get("medical_output", {})
            deidentified_medical = self.data_processor.deidentify_medical_data_enhanced(medical_data, state["patient_data"])
            state["deidentified_medical"] = deidentified_medical

            pharmacy_data = state.get("pharmacy_output", {})
            deidentified_pharmacy = self.data_processor.deidentify_pharmacy_data_enhanced(pharmacy_data)
            state["deidentified_pharmacy"] = deidentified_pharmacy

            mcid_data = state.get("mcid_output", {})
            deidentified_mcid = self.data_processor.deidentify_mcid_data_enhanced(mcid_data)
            state["deidentified_mcid"] = deidentified_mcid

            state["step_status"]["deidentify_claims_data"] = "completed"
            logger.info("✅ Successfully completed comprehensive claims data deidentification")

        except Exception as e:
            error_msg = f"Error in claims data deidentification: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["deidentify_claims_data"] = "error"
            logger.error(error_msg)

        return state

    def extract_claims_fields(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 3: Extract fields from claims data with batch processing"""
        logger.info("🔍 Node 3: Starting enhanced claims field extraction with batch processing...")
        state["current_step"] = "extract_claims_fields"
        state["step_status"]["extract_claims_fields"] = "running"

        try:
            # Extract medical and pharmacy fields with batch processing
            medical_extraction = self.data_processor.extract_medical_fields_batch_enhanced(state.get("deidentified_medical", {}))
            state["medical_extraction"] = medical_extraction
            logger.info(f"📋 Medical extraction: {len(medical_extraction.get('hlth_srvc_records', []))} health service records")
            logger.info(f"🔬 Medical batch status: {medical_extraction.get('llm_call_status', 'unknown')}")

            pharmacy_extraction = self.data_processor.extract_pharmacy_fields_batch_enhanced(state.get("deidentified_pharmacy", {}))
            state["pharmacy_extraction"] = pharmacy_extraction
            logger.info(f"💊 Pharmacy extraction: {len(pharmacy_extraction.get('ndc_records', []))} NDC records")
            logger.info(f"🔬 Pharmacy batch status: {pharmacy_extraction.get('llm_call_status', 'unknown')}")

            state["step_status"]["extract_claims_fields"] = "completed"
            logger.info("✅ Successfully completed enhanced claims field extraction with batch processing")

        except Exception as e:
            error_msg = f"Error in claims field extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_claims_fields"] = "error"
            logger.error(error_msg)

        return state

    def extract_entities(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 4: Extract health entities using LLM"""
        logger.info("🎯 Node 4: Starting LLM-powered health entity extraction...")
        state["current_step"] = "extract_entities"
        state["step_status"]["extract_entities"] = "running"
       
        try:
            pharmacy_data = state.get("pharmacy_output", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            medical_extraction = state.get("medical_extraction", {})
            patient_data = state.get("patient_data", {})
           
            # Calculate age
            if patient_data.get('date_of_birth'):
                try:
                    dob = datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d').date()
                    today = date.today()
                    calculated_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    patient_data['calculated_age'] = calculated_age
                    logger.info(f"📅 Calculated age from DOB: {calculated_age} years")
                except Exception as e:
                    logger.warning(f"Could not calculate age from DOB: {e}")
           
            # Extract entities using enhanced method
            entities = self.data_processor.extract_health_entities_with_clinical_insights(
                pharmacy_data,
                pharmacy_extraction,
                medical_extraction,
                patient_data,
                self.api_integrator
            )
           
            state["entity_extraction"] = entities
            state["step_status"]["extract_entities"] = "completed"
           
            conditions_count = len(entities.get("medical_conditions", []))
            medications_count = len(entities.get("medications_identified", []))
            stable_analysis = entities.get("stable_analysis", False)
            age_info = f"Age: {entities.get('age', 'unknown')} ({entities.get('age_group', 'unknown')})"
           
            logger.info(f"✅ Successfully extracted health entities: {conditions_count} conditions, {medications_count} medications")
            logger.info(f"📊 Entity results: Diabetes={entities.get('diabetics')}, Smoking={entities.get('smoking')}, BP={entities.get('blood_pressure')}")
            logger.info(f"📅 {age_info}")
            logger.info(f"🔬 Stable analysis: {stable_analysis}")
           
        except Exception as e:
            error_msg = f"Error in entity extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities"] = "error"
            logger.error(error_msg)
       
        return state

    def analyze_trajectory(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 5: Comprehensive health trajectory analysis with evaluation questions"""
        logger.info("📈 Node 5: Starting comprehensive health trajectory analysis...")
        state["current_step"] = "analyze_trajectory"
        state["step_status"]["analyze_trajectory"] = "running"

        try:
            deidentified_medical = state.get("deidentified_medical", {})
            deidentified_pharmacy = state.get("deidentified_pharmacy", {})
            deidentified_mcid = state.get("deidentified_mcid", {})
            medical_extraction = state.get("medical_extraction", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            entities = state.get("entity_extraction", {})

            trajectory_prompt = self._create_comprehensive_trajectory_prompt_with_evaluation(
                deidentified_medical, deidentified_pharmacy, deidentified_mcid,
                medical_extraction, pharmacy_extraction, entities
            )

            logger.info("🤖 Calling Snowflake Cortex for comprehensive trajectory analysis...")

            response = self.api_integrator.call_llm_enhanced(trajectory_prompt, self.config.sys_msg)

            if response.startswith("Error"):
                state["errors"].append(f"Trajectory analysis failed: {response}")
                state["step_status"]["analyze_trajectory"] = "error"
            else:
                state["health_trajectory"] = response
                state["step_status"]["analyze_trajectory"] = "completed"
                logger.info("✅ Successfully completed comprehensive trajectory analysis")

        except Exception as e:
            error_msg = f"Error in trajectory analysis: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["analyze_trajectory"] = "error"
            logger.error(error_msg)

        return state

    def generate_summary(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 6: Generate comprehensive final summary"""
        logger.info("📋 Node 6: Generating comprehensive final summary...")
        state["current_step"] = "generate_summary"
        state["step_status"]["generate_summary"] = "running"

        try:
            summary_prompt = self._create_comprehensive_summary_prompt(
                state.get("health_trajectory", ""),
                state.get("entity_extraction", {}),
                state.get("medical_extraction", {}),
                state.get("pharmacy_extraction", {})
            )

            logger.info("🤖 Calling Snowflake Cortex for final summary...")

            response = self.api_integrator.call_llm_enhanced(summary_prompt, self.config.sys_msg)

            if response.startswith("Error"):
                state["errors"].append(f"Summary generation failed: {response}")
                state["step_status"]["generate_summary"] = "error"
            else:
                state["final_summary"] = response
                state["step_status"]["generate_summary"] = "completed"
                logger.info("✅ Successfully generated comprehensive final summary")

        except Exception as e:
            error_msg = f"Error in summary generation: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["generate_summary"] = "error"
            logger.error(error_msg)

        return state

    def generate_code_meanings_tables(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 7: Generate code meanings tables and include in JSON output"""
        logger.info("📋 Node 7: Generating code meanings tables for JSON output...")
        state["current_step"] = "generate_code_meanings_tables"
        state["step_status"]["generate_code_meanings_tables"] = "running"

        try:
            # Prepare results structure for table generation
            results = {
                "structured_extractions": {
                    "medical": state.get("medical_extraction", {}),
                    "pharmacy": state.get("pharmacy_extraction", {})
                }
            }

            # Generate comprehensive code meanings tables data
            code_meanings_tables = self._generate_code_meanings_tables_data(results)
            
            state["code_meanings_tables"] = code_meanings_tables
            state["step_status"]["generate_code_meanings_tables"] = "completed"
            
            logger.info("✅ Successfully generated code meanings tables for JSON output")
            logger.info(f"📊 Tables generated: Medical={code_meanings_tables.get('medical_table_generated', False)}, Pharmacy={code_meanings_tables.get('pharmacy_table_generated', False)}")

        except Exception as e:
            error_msg = f"Error in code meanings tables generation: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["generate_code_meanings_tables"] = "error"
            logger.error(error_msg)

        return state

    def predict_heart_attack(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 8: Heart attack prediction"""
        logger.info("❤️ Node 8: Starting heart attack prediction...")
        state["current_step"] = "predict_heart_attack"
        state["step_status"]["predict_heart_attack"] = "running"

        try:
            # Extract features
            logger.info("🔍 Extracting heart attack features...")
            features = self._extract_enhanced_heart_attack_features(state)
            state["heart_attack_features"] = features

            if not features or "error" in features:
                error_msg = "Failed to extract features for heart attack prediction"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Prepare features for API call
            logger.info("⚙️ Preparing features for API call...")
            fastapi_features = self._prepare_enhanced_fastapi_features(features)

            if fastapi_features is None:
                error_msg = "Failed to prepare feature vector for prediction"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Make prediction
            logger.info("🚀 Making heart attack prediction call...")
            prediction_result = self._call_heart_attack_prediction_sync(fastapi_features)

            if prediction_result is None:
                error_msg = "Heart attack prediction returned None"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Process result
            if prediction_result.get("success", False):
                logger.info("✅ Processing successful prediction result...")
                
                prediction_data = prediction_result.get("prediction_data", {})
                risk_probability = prediction_data.get("probability", 0.0)
                binary_prediction = prediction_data.get("prediction", 0)
                
                risk_percentage = risk_probability * 100
                confidence_percentage = (1 - risk_probability) * 100 if binary_prediction == 0 else risk_probability * 100
                
                if risk_percentage >= 70:
                    risk_category = "High Risk"
                elif risk_percentage >= 50:
                    risk_category = "Medium Risk"
                else:
                    risk_category = "Low Risk"
                
                enhanced_prediction = {
                    "risk_display": f"Heart Disease Risk: {risk_percentage:.1f}% ({risk_category})",
                    "confidence_display": f"Confidence: {confidence_percentage:.1f}%",
                    "combined_display": f"Heart Disease Risk: {risk_percentage:.1f}% ({risk_category}) | Confidence: {confidence_percentage:.1f}%",
                    "raw_risk_score": risk_probability,
                    "raw_prediction": binary_prediction,
                    "risk_category": risk_category,
                    "fastapi_server_url": self.config.heart_attack_api_url,
                    "prediction_method": prediction_result.get("method", "unknown"),
                    "prediction_endpoint": prediction_result.get("endpoint", "unknown"),
                    "prediction_timestamp": datetime.now().isoformat(),
                    "enhanced_features_used": features.get("feature_interpretation", {}),
                    "model_enhanced": True
                }
                
                state["heart_attack_prediction"] = enhanced_prediction
                state["heart_attack_risk_score"] = float(risk_probability)
                
                logger.info(f"✅ Heart attack prediction completed successfully")
                logger.info(f"❤️ Display: {enhanced_prediction['combined_display']}")
                
            else:
                error_msg = prediction_result.get("error", "Unknown API error")
                logger.warning(f"⚠️ Heart attack prediction failed: {error_msg}")
                
                state["heart_attack_prediction"] = {
                    "error": error_msg,
                    "risk_display": "Heart Disease Risk: Error",
                    "confidence_display": "Confidence: Error",
                    "combined_display": f"Heart Disease Risk: Error - {error_msg}",
                    "fastapi_server_url": self.config.heart_attack_api_url,
                    "error_details": error_msg,
                    "tried_endpoints": prediction_result.get("tried_endpoints", []),
                    "model_enhanced": True
                }
                state["heart_attack_risk_score"] = 0.0
                
            state["step_status"]["predict_heart_attack"] = "completed"

        except Exception as e:
            error_msg = f"Error in heart attack prediction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["predict_heart_attack"] = "error"
            logger.error(error_msg)

        return state

    def initialize_chatbot(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 9: Initialize comprehensive chatbot with graph generation"""
        logger.info("💬 Node 9: Initializing comprehensive chatbot with graph generation...")
        state["current_step"] = "initialize_chatbot"
        state["step_status"]["initialize_chatbot"] = "running"

        try:
            # Prepare comprehensive chatbot context
            comprehensive_chatbot_context = {
                "deidentified_medical": state.get("deidentified_medical", {}),
                "deidentified_pharmacy": state.get("deidentified_pharmacy", {}),
                "deidentified_mcid": state.get("deidentified_mcid", {}),
                "medical_extraction": state.get("medical_extraction", {}),
                "pharmacy_extraction": state.get("pharmacy_extraction", {}),
                "entity_extraction": state.get("entity_extraction", {}),
                "health_trajectory": state.get("health_trajectory", ""),
                "final_summary": state.get("final_summary", ""),
                "heart_attack_prediction": state.get("heart_attack_prediction", {}),
                "heart_attack_risk_score": state.get("heart_attack_risk_score", 0.0),
                "heart_attack_features": state.get("heart_attack_features", {}),
                "patient_overview": {
                    "age": state.get("deidentified_medical", {}).get("src_mbr_age", "unknown"),
                    "zip": state.get("deidentified_medical", {}).get("src_mbr_zip_cd", "unknown"),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "heart_attack_risk_level": state.get("heart_attack_prediction", {}).get("risk_category", "unknown"),
                    "model_type": "enhanced_ml_api_comprehensive",
                    "deidentification_level": "comprehensive_claims_data",
                    "claims_data_types": ["medical", "pharmacy", "mcid"],
                    "graph_generation_supported": True,
                    "batch_code_meanings_available": True
                }
            }

            state["chat_history"] = []
            state["chatbot_context"] = comprehensive_chatbot_context
            state["chatbot_ready"] = True
            state["graph_generation_ready"] = True
            state["processing_complete"] = True
            state["step_status"]["initialize_chatbot"] = "completed"

            medical_records = len(state.get("medical_extraction", {}).get("hlth_srvc_records", []))
            pharmacy_records = len(state.get("pharmacy_extraction", {}).get("ndc_records", []))

            logger.info("✅ Successfully initialized comprehensive chatbot with graph generation")
            logger.info(f"📊 Chatbot context includes: {medical_records} medical records, {pharmacy_records} pharmacy records")
            logger.info(f"📈 Graph generation: Ready for matplotlib visualizations")

        except Exception as e:
            error_msg = f"Error initializing chatbot: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["initialize_chatbot"] = "error"
            logger.error(error_msg)

        return state

    def handle_error(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Error handling node"""
        logger.error(f"🚨 LangGraph Error Handler: {state['current_step']}")
        logger.error(f"Errors: {state['errors']}")

        state["processing_complete"] = True
        current_step = state.get("current_step", "unknown")
        state["step_status"][current_step] = "error"
        return state

    # ===== CONDITIONAL EDGES =====

    def should_continue_after_api(self, state: HealthAnalysisState) -> Literal["continue", "retry", "error"]:
        if state["errors"]:
            if state["retry_count"] < 3:  # max_retries
                state["retry_count"] += 1
                logger.warning(f"🔄 Retrying API fetch (attempt {state['retry_count']}/3)")
                state["errors"] = []
                return "retry"
            else:
                logger.error(f"❌ Max retries (3) exceeded for API fetch")
                return "error"
        return "continue"

    def should_continue_after_deidentify(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_extraction_step(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_entity_extraction(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_trajectory(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_summary(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_code_meanings_tables(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_heart_attack_prediction(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    # ===== CODE MEANINGS TABLES GENERATION =====

    def _generate_code_meanings_tables_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive code meanings tables data for JSON output"""
        try:
            logger.info("📋 Generating code meanings tables data...")
            
            # Get extraction results
            medical_extraction = safe_get(results, 'structured_extractions', {}).get('medical', {})
            pharmacy_extraction = safe_get(results, 'structured_extractions', {}).get('pharmacy', {})
            
            # Initialize table data structure
            tables_data = {
                "medical_table_generated": False,
                "pharmacy_table_generated": False,
                "generation_timestamp": datetime.now().isoformat(),
                "tables": {
                    "medical_codes_table": {
                        "summary": {},
                        "icd10_diagnosis_codes": [],
                        "medical_service_codes": [],
                        "frequency_analysis": {}
                    },
                    "pharmacy_codes_table": {
                        "summary": {},
                        "ndc_medication_codes": [],
                        "medication_names": [],
                        "frequency_analysis": {}
                    }
                }
            }
            
            # Generate Medical Table Data
            medical_table_data = self._generate_medical_table_data(medical_extraction)
            if medical_table_data["has_data"]:
                tables_data["tables"]["medical_codes_table"] = medical_table_data
                tables_data["medical_table_generated"] = True
                logger.info("✅ Medical codes table data generated")
            
            # Generate Pharmacy Table Data  
            pharmacy_table_data = self._generate_pharmacy_table_data(pharmacy_extraction)
            if pharmacy_table_data["has_data"]:
                tables_data["tables"]["pharmacy_codes_table"] = pharmacy_table_data
                tables_data["pharmacy_table_generated"] = True
                logger.info("✅ Pharmacy codes table data generated")
            
            # Overall statistics
            tables_data["overall_stats"] = {
                "total_medical_records": len(medical_extraction.get("hlth_srvc_records", [])),
                "total_pharmacy_records": len(pharmacy_extraction.get("ndc_records", [])),
                "medical_batch_status": medical_extraction.get("llm_call_status", "unknown"),
                "pharmacy_batch_status": pharmacy_extraction.get("llm_call_status", "unknown")
            }
            
            logger.info(f"📊 Code meanings tables data generated: Medical={tables_data['medical_table_generated']}, Pharmacy={tables_data['pharmacy_table_generated']}")
            
            return tables_data
            
        except Exception as e:
            logger.error(f"Error generating code meanings tables data: {e}")
            return {
                "medical_table_generated": False,
                "pharmacy_table_generated": False,
                "error": str(e),
                "generation_timestamp": datetime.now().isoformat()
            }

    def _generate_medical_table_data(self, medical_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate medical codes table data"""
        try:
            medical_meanings = medical_extraction.get("code_meanings", {})
            diagnosis_meanings = medical_meanings.get("diagnosis_code_meanings", {})
            service_meanings = medical_meanings.get("service_code_meanings", {})
            medical_records = medical_extraction.get("hlth_srvc_records", [])
            
            # Check if we have data
            has_data = bool(medical_records and (diagnosis_meanings or service_meanings))
            
            if not has_data:
                return {"has_data": False, "reason": "No medical data or code meanings available"}
            
            # Calculate metrics
            unique_service_codes = set()
            unique_diagnosis_codes = set()
            
            for record in medical_records:
                service_code = record.get("hlth_srvc_cd", "")
                if service_code:
                    unique_service_codes.add(service_code)
                
                for diag in record.get("diagnosis_codes", []):
                    code = diag.get("code", "")
                    if code:
                        unique_diagnosis_codes.add(code)
            
            # Build medical table data
            medical_data = {
                "has_data": True,
                "summary": {
                    "service_codes_count": len(unique_service_codes),
                    "diagnosis_codes_count": len(unique_diagnosis_codes),
                    "medical_records_count": len(medical_records),
                    "batch_status": medical_extraction.get("llm_call_status", "unknown")
                },
                "icd10_diagnosis_codes": [],
                "medical_service_codes": [],
                "frequency_analysis": {
                    "most_frequent_diagnosis_codes": [],
                    "most_frequent_service_codes": []
                }
            }
            
            # ICD-10 Diagnosis Codes Data
            if diagnosis_meanings and medical_records:
                diagnosis_data = []
                for record in medical_records:
                    claim_date = record.get("clm_rcvd_dt", "Unknown")
                    for diag in record.get("diagnosis_codes", []):
                        code = diag.get("code", "")
                        if code in diagnosis_meanings:
                            diagnosis_data.append({
                                "code": code,
                                "meaning": diagnosis_meanings[code],
                                "claim_date": claim_date,
                                "position": diag.get("position", ""),
                                "source": diag.get("source", "")
                            })
                
                # Sort by date and limit to top 20
                sorted_diagnosis_data = sorted(diagnosis_data, key=lambda x: x['claim_date'], reverse=True)
                medical_data["icd10_diagnosis_codes"] = sorted_diagnosis_data[:20]
                
                # Frequency analysis
                code_counts = Counter(item["code"] for item in diagnosis_data)
                medical_data["frequency_analysis"]["most_frequent_diagnosis_codes"] = [
                    {
                        "code": code,
                        "count": count,
                        "meaning": diagnosis_meanings.get(code, "Unknown")
                    }
                    for code, count in code_counts.most_common(5)
                ]
            
            # Medical Service Codes Data
            if service_meanings and medical_records:
                service_data = []
                for record in medical_records:
                    service_end_date = record.get("clm_line_srvc_end_dt", "Unknown")
                    service_code = record.get("hlth_srvc_cd", "")
                    
                    if service_code and service_code in service_meanings:
                        service_data.append({
                            "code": service_code,
                            "meaning": service_meanings[service_code],
                            "end_date": service_end_date
                        })
                
                # Sort by date and limit to top 15
                sorted_service_data = sorted(service_data, key=lambda x: x['end_date'], reverse=True)
                medical_data["medical_service_codes"] = sorted_service_data[:15]
                
                # Frequency analysis
                code_counts = Counter(item["code"] for item in service_data)
                medical_data["frequency_analysis"]["most_frequent_service_codes"] = [
                    {
                        "code": code,
                        "count": count,
                        "meaning": service_meanings.get(code, "Unknown")
                    }
                    for code, count in code_counts.most_common(5)
                ]
            
            return medical_data
            
        except Exception as e:
            logger.error(f"Error generating medical table data: {e}")
            return {"has_data": False, "error": str(e)}

    def _generate_pharmacy_table_data(self, pharmacy_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate pharmacy codes table data"""
        try:
            pharmacy_meanings = pharmacy_extraction.get("code_meanings", {})
            ndc_meanings = pharmacy_meanings.get("ndc_code_meanings", {})
            med_meanings = pharmacy_meanings.get("medication_meanings", {})
            pharmacy_records = pharmacy_extraction.get("ndc_records", [])
            
            # Check if we have data
            has_data = bool(pharmacy_records and (ndc_meanings or med_meanings))
            
            if not has_data:
                return {"has_data": False, "reason": "No pharmacy data or code meanings available"}
            
            # Calculate metrics
            unique_ndc_codes = set()
            unique_medications = set()
            
            for record in pharmacy_records:
                ndc_code = record.get("ndc", "")
                if ndc_code:
                    unique_ndc_codes.add(ndc_code)
                
                med_name = record.get("lbl_nm", "")
                if med_name:
                    unique_medications.add(med_name)
            
            # Build pharmacy table data
            pharmacy_data = {
                "has_data": True,
                "summary": {
                    "ndc_codes_count": len(unique_ndc_codes),
                    "medications_count": len(unique_medications),
                    "pharmacy_records_count": len(pharmacy_records),
                    "batch_status": pharmacy_extraction.get("llm_call_status", "unknown")
                },
                "ndc_medication_codes": [],
                "medication_names": [],
                "frequency_analysis": {
                    "most_frequent_medications": [],
                    "most_frequent_ndc_codes": []
                }
            }
            
            # NDC Medication Codes Data
            if ndc_meanings and pharmacy_records:
                ndc_data = []
                for record in pharmacy_records:
                    ndc_code = record.get("ndc", "")
                    fill_date = record.get("rx_filled_dt", "Unknown")
                    
                    if ndc_code and ndc_code in ndc_meanings:
                        ndc_data.append({
                            "ndc_code": ndc_code,
                            "meaning": ndc_meanings[ndc_code],
                            "fill_date": fill_date,
                            "medication_name": record.get("lbl_nm", "")
                        })
                
                # Sort by date and limit to top 15
                sorted_ndc_data = sorted(ndc_data, key=lambda x: x['fill_date'], reverse=True)
                pharmacy_data["ndc_medication_codes"] = sorted_ndc_data[:15]
            
            # Medication Names Data
            if med_meanings and pharmacy_records:
                med_data = []
                for record in pharmacy_records:
                    med_name = record.get("lbl_nm", "")
                    fill_date = record.get("rx_filled_dt", "Unknown")
                    
                    if med_name and med_name in med_meanings:
                        med_data.append({
                            "medication_name": med_name,
                            "meaning": med_meanings[med_name],
                            "fill_date": fill_date,
                            "ndc_code": record.get("ndc", "")
                        })
                
                # Sort by date and limit to top 15
                sorted_med_data = sorted(med_data, key=lambda x: x['fill_date'], reverse=True)
                pharmacy_data["medication_names"] = sorted_med_data[:15]
                
                # Frequency analysis
                med_counts = Counter(item["medication_name"] for item in med_data)
                pharmacy_data["frequency_analysis"]["most_frequent_medications"] = [
                    {
                        "medication": med,
                        "count": count,
                        "meaning": med_meanings.get(med, "Unknown")
                    }
                    for med, count in med_counts.most_common(5)
                ]
            
            return pharmacy_data
            
        except Exception as e:
            logger.error(f"Error generating pharmacy table data: {e}")
            return {"has_data": False, "error": str(e)}

    # ===== CODE MEANINGS DISPLAY INTEGRATION =====

    def display_code_meanings(self, results: Dict[str, Any]) -> str:
        """Display code meanings in LangChain format"""
        return display_batch_code_meanings_langchain(results)

    def print_code_meanings(self, results: Dict[str, Any]):
        """Print code meanings to console"""
        print_code_meanings_langchain(results)

    def get_code_meanings_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get code meanings summary"""
        return get_code_meanings_summary_langchain(results)

    def chat_with_data_and_display_codes(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Enhanced chatbot that can also display code meanings tables"""
        
        # Check if user is asking for code meanings display
        code_display_keywords = ['show codes', 'display codes', 'code meanings', 'show table', 'display table', 'batch codes', 'medical codes', 'pharmacy codes']
        
        if any(keyword in user_query.lower() for keyword in code_display_keywords):
            # Prepare results structure for code meanings display
            results = {
                "structured_extractions": {
                    "medical": chat_context.get("medical_extraction", {}),
                    "pharmacy": chat_context.get("pharmacy_extraction", {})
                }
            }
            
            # Generate the code meanings display
            code_display = self.display_code_meanings(results)
            
            return f"""## Code Meanings Analysis

Here is the comprehensive analysis of all medical and pharmacy codes found in the patient's claims data:

```
{code_display}
```

This analysis shows:
- **Medical Codes**: ICD-10 diagnosis codes and CPT/HCPCS service codes with their meanings
- **Pharmacy Codes**: NDC medication codes and drug names with therapeutic descriptions
- **Frequency Analysis**: Most common codes and medications
- **Batch Processing Status**: Results from LLM-powered code interpretation

You can ask follow-up questions about any specific codes or request additional analysis of the healthcare data."""

        # Otherwise, use the regular chatbot functionality
        return self.chat_with_data(user_query, chat_context, chat_history)

    # ===== ENHANCED CHATBOT FUNCTIONALITY WITH GRAPH GENERATION =====

    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Enhanced chatbot with comprehensive claims data access and advanced graph generation"""
        try:
            # Check if this is a graph request using data processor's detection
            graph_request = self.data_processor.detect_graph_request(user_query)

            if graph_request.get("is_graph_request", False):
                return self._handle_graph_request_enhanced(user_query, chat_context, chat_history, graph_request)

            # Check if this is a heart attack related question
            heart_attack_keywords = ['heart attack', 'heart disease', 'cardiac', 'cardiovascular', 'heart risk', 'coronary', 'myocardial', 'cardiac risk']
            is_heart_attack_question = any(keyword in user_query.lower() for keyword in heart_attack_keywords)

            if is_heart_attack_question:
                return self._handle_heart_attack_question_enhanced(user_query, chat_context, chat_history)
            else:
                return self._handle_general_question_enhanced(user_query, chat_context, chat_history)

        except Exception as e:
            logger.error(f"Error in enhanced chatbot: {str(e)}")
            return "I encountered an error processing your question. Please try again. I have access to comprehensive deidentified claims data and can generate visualizations for detailed analysis."

    def _handle_graph_request_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]], graph_request: Dict[str, Any]) -> str:
        """Handle graph generation requests with enhanced matplotlib support"""
        try:
            graph_type = graph_request.get("graph_type", "general")
            
            logger.info(f"📊 Generating {graph_type} visualization for user query: {user_query[:50]}...")
            
            # Use the API integrator's specialized graph generation method
            response = self.api_integrator.call_llm_for_graph_generation(user_query, chat_context)
            
            # If API call fails, use data processor's fallback generation
            if "Graph generation failed" in response or "Error" in response:
                logger.warning("API graph generation failed, using data processor fallback")
                matplotlib_code = self.data_processor.generate_matplotlib_code(graph_type, chat_context)
                
                response = f"""## Healthcare Data Visualization

I'll create a {graph_type} visualization for your healthcare data.

```python
{matplotlib_code}
```

This visualization uses your actual patient data when available, including medical records, pharmacy claims, and risk assessments. The chart provides clinical insights based on the comprehensive healthcare analysis."""

            return response
                
        except Exception as e:
            logger.error(f"Error handling enhanced graph request: {str(e)}")
            return f"""
## Graph Generation Error

I encountered an error while generating your requested visualization: {str(e)}

Available Graph Types:
- **Medication Timeline**: "show me a medication timeline"
- **Diagnosis Timeline**: "create a diagnosis timeline chart"  
- **Risk Dashboard**: "generate a risk assessment dashboard"
- **Medication Distribution**: "show me a pie chart of medications"
- **Health Overview**: "show comprehensive health overview"

Please try rephrasing your request with one of these specific graph types.
"""

    def _handle_heart_attack_question_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle heart attack related questions with comprehensive analysis"""
        try:
            # Get comprehensive context data
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            entity_extraction = chat_context.get("entity_extraction", {})
            medical_extraction = chat_context.get("medical_extraction", {})
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            
            patient_age = chat_context.get("patient_overview", {}).get("age", "unknown")
            risk_display = heart_attack_prediction.get("risk_display", "Not available")

            # Build conversation history
            history_text = "No previous conversation"
            if chat_history:
                recent_history = chat_history[-3:]
                history_lines = []
                for msg in recent_history:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    content = msg['content'][:100]
                    history_lines.append(f"{role}: {content}")
                history_text = "\n".join(history_lines)

            # Create comprehensive heart attack analysis prompt
            heart_attack_prompt = f"""You are Dr. CardioAI, a specialist in cardiovascular risk assessment with access to comprehensive patient claims data.

COMPREHENSIVE PATIENT DATA AVAILABLE:
**PATIENT DEMOGRAPHICS:**
- Age: {patient_age}
- Current Health Status: Diabetes: {entity_extraction.get('diabetics', 'unknown')}, Blood Pressure: {entity_extraction.get('blood_pressure', 'unknown')}, Smoking: {entity_extraction.get('smoking', 'unknown')}

**COMPLETE MEDICAL CLAIMS DATA:**
{json.dumps(medical_extraction, indent=2)}

**COMPLETE PHARMACY CLAIMS DATA:**
{json.dumps(pharmacy_extraction, indent=2)}

**ENHANCED HEALTH ENTITIES:**
{json.dumps(entity_extraction, indent=2)}

**CURRENT ML MODEL PREDICTION:**
{risk_display}

**RECENT CONVERSATION:**
{history_text}

**USER QUESTION:** {user_query}

**COMPREHENSIVE ANALYSIS INSTRUCTIONS:**

🔬 **DATA UTILIZATION REQUIREMENTS:**
- Analyze ALL available medical claims data including ICD-10 diagnosis codes
- Review ALL pharmacy claims data including NDC medication codes  
- Examine complete health entity extraction results
- Reference specific codes, dates, and clinical findings
- Use both medical AND pharmacy data comprehensively

📊 **CARDIOVASCULAR RISK ASSESSMENT PROTOCOL:**
1. **Clinical Data Analysis**: Review all ICD-10 codes for cardiovascular conditions, diabetes, hypertension
2. **Medication Analysis**: Examine NDC codes for cardiovascular medications, diabetes drugs, lipid management
3. **Risk Factor Identification**: Identify modifiable and non-modifiable risk factors from complete data
4. **Comorbidity Assessment**: Analyze disease burden and interaction effects
5. **Temporal Analysis**: Review claims dates for disease progression patterns

💡 **RESPONSE REQUIREMENTS:**
- Provide specific risk percentage assessment based on comprehensive clinical data
- Reference exact ICD-10 codes, NDC codes, and claim dates
- Compare your clinical assessment with the ML model prediction
- Explain reasoning using available clinical evidence
- Include actionable recommendations for risk reduction

**PROVIDE COMPREHENSIVE CARDIOVASCULAR RISK ANALYSIS:**

## 🫀 COMPREHENSIVE CARDIOVASCULAR RISK ASSESSMENT

**Clinical Risk Analysis:** [Your detailed percentage assessment]%
**Risk Category:** [Low/Medium/High Risk with clinical justification]

**Key Risk Factors Identified:**
[List specific factors from complete claims data with codes and dates]

**Supporting Clinical Evidence:**
[Reference specific ICD-10 codes, NDC codes, medications, and claim dates]

**ML Model Comparison:**
- ML Prediction: {risk_display}
- Clinical Assessment: [Your assessment]
- Analysis Agreement: [Compare and explain differences]

**Detailed Clinical Reasoning:**
[Comprehensive analysis using all available claims data]

**Risk Reduction Recommendations:**
[Specific actionable recommendations based on identified risk factors]

Use the complete deidentified claims dataset to provide the most accurate and comprehensive cardiovascular risk assessment possible."""

            logger.info(f"Processing enhanced heart attack question: {user_query[:50]}...")

            response = self.api_integrator.call_llm_enhanced(heart_attack_prompt, self.config.chatbot_sys_msg)

            if response.startswith("Error"):
                return "I encountered an error analyzing cardiovascular risk. Please try rephrasing your question."

            return response

        except Exception as e:
            logger.error(f"Error in enhanced heart attack question: {str(e)}")
            return "I encountered an error with cardiovascular analysis. Please try again with a simpler question about heart disease risk."

    def _handle_general_question_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle general questions with comprehensive context"""
        try:
            # Prepare comprehensive context
            complete_context = self.data_processor.prepare_enhanced_clinical_context(chat_context)
            
            # Build conversation history
            history_text = "No previous conversation"
            if chat_history:
                recent_history = chat_history[-5:]
                history_lines = []
                for msg in recent_history:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    content = msg['content'][:150]
                    history_lines.append(f"{role}: {content}")
                history_text = "\n".join(history_lines)

            # Create comprehensive analysis prompt
            comprehensive_prompt = f"""You are Dr. AnalysisAI, a healthcare data analyst with access to comprehensive patient claims data and advanced visualization capabilities.

**COMPREHENSIVE DATA ACCESS:**
{complete_context}

**CONVERSATION HISTORY:**
{history_text}

**PATIENT QUESTION:** {user_query}

**COMPREHENSIVE ANALYSIS INSTRUCTIONS:**

🔬 **COMPLETE DATA UTILIZATION:**
- Access ALL deidentified medical claims data with ICD-10 diagnosis codes
- Review ALL deidentified pharmacy claims data with NDC medication codes
- Examine complete health entity extraction results and risk assessments
- Reference specific codes, dates, medications, and clinical findings
- Use comprehensive claims dataset for thorough analysis
- Utilize batch-processed code meanings for enhanced clinical insights

📊 **CLINICAL DATA NAVIGATION:**
- Medical Claims: Access diagnosis codes (ICD-10), procedure codes (CPT), service dates
- Pharmacy Claims: Access medication names, NDC codes, prescription fill dates
- Entity Data: Access chronic conditions, risk factors, demographic information
- Dates: Reference clm_rcvd_dt (medical), rx_filled_dt (pharmacy) for temporal analysis
- Code Meanings: Use batch-processed explanations for clinical interpretation

💡 **RESPONSE REQUIREMENTS:**
- Provide data-driven answers using specific information from claims
- Reference exact codes, dates, and clinical findings
- Explain medical terminology and provide clinical context
- Include both medical AND pharmacy data in comprehensive analysis
- Cite specific field paths and values from the JSON data structure
- Generate matplotlib code if visualization is requested

📈 **VISUALIZATION CAPABILITIES:**
- Generate working matplotlib code for healthcare visualizations
- Create medication timelines, diagnosis progressions, risk dashboards
- Use actual patient data when available
- Provide complete, executable Python code with proper imports

**CRITICAL ANALYSIS STANDARDS:**
- Use only the provided deidentified claims data for analysis
- Reference specific ICD-10 codes, NDC codes, and dates when relevant
- Provide evidence-based insights based on available clinical data
- Include predictive insights when supported by clinical indicators
- Maintain professional healthcare analysis standards
- Generate graphs when visualization would enhance understanding

**COMPREHENSIVE RESPONSE USING COMPLETE CLAIMS DATA:**
[Provide detailed analysis using all available deidentified medical and pharmacy claims data]"""

            logger.info(f"Processing enhanced general query: {user_query[:50]}...")

            response = self.api_integrator.call_llm_enhanced(comprehensive_prompt, self.config.chatbot_sys_msg)

            if response.startswith("Error"):
                return "I encountered an error processing your question. Please try rephrasing it more simply."

            return response

        except Exception as e:
            logger.error(f"Error in enhanced general question: {str(e)}")
            return "I encountered an error. Please try again with a simpler question."

    # ===== AUTOMATIC CODE MEANINGS TABLE DISPLAY =====

    def _print_final_code_meanings_tables(self, results: Dict[str, Any]):
        """Print final code meanings tables when LangGraph completes successfully"""
        try:
            print("\n" + "=" * 100)
            print("🎯 LANGGRAPH ANALYSIS COMPLETE - DISPLAYING CODE MEANINGS TABLES")
            print("=" * 100)
            
            # Get extraction results
            medical_extraction = safe_get(results, 'structured_extractions', {}).get('medical', {})
            pharmacy_extraction = safe_get(results, 'structured_extractions', {}).get('pharmacy', {})
            
            # Check if we have data to display
            medical_has_data = (medical_extraction.get("hlth_srvc_records") and 
                               medical_extraction.get("code_meanings", {}).get("diagnosis_code_meanings"))
            pharmacy_has_data = (pharmacy_extraction.get("ndc_records") and 
                                pharmacy_extraction.get("code_meanings", {}).get("medication_meanings"))
            
            if not medical_has_data and not pharmacy_has_data:
                print("⚠️  No code meanings data available to display")
                print("=" * 100)
                return
            
            # Display Medical Table
            if medical_has_data:
                print("\n" + "🏥 MEDICAL CODES TABLE")
                print("-" * 80)
                self._print_medical_codes_table(medical_extraction)
            else:
                print("\n" + "🏥 MEDICAL CODES TABLE")
                print("-" * 80)
                print("⚠️  No medical codes data available")
            
            # Display Pharmacy Table  
            if pharmacy_has_data:
                print("\n" + "💊 PHARMACY CODES TABLE")
                print("-" * 80)
                self._print_pharmacy_codes_table(pharmacy_extraction)
            else:
                print("\n" + "💊 PHARMACY CODES TABLE") 
                print("-" * 80)
                print("⚠️  No pharmacy codes data available")
            
            print("\n" + "=" * 100)
            print("✅ CODE MEANINGS TABLES DISPLAY COMPLETE")
            print("=" * 100)
            
        except Exception as e:
            logger.error(f"Error displaying final code meanings tables: {e}")
            print(f"\n❌ Error displaying code meanings tables: {e}")

    def _print_medical_codes_table(self, medical_extraction: Dict[str, Any]):
        """Print formatted medical codes table"""
        try:
            medical_meanings = medical_extraction.get("code_meanings", {})
            diagnosis_meanings = medical_meanings.get("diagnosis_code_meanings", {})
            service_meanings = medical_meanings.get("service_code_meanings", {})
            medical_records = medical_extraction.get("hlth_srvc_records", [])
            
            # Calculate metrics
            unique_service_codes = set()
            unique_diagnosis_codes = set()
            
            for record in medical_records:
                service_code = record.get("hlth_srvc_cd", "")
                if service_code:
                    unique_service_codes.add(service_code)
                
                for diag in record.get("diagnosis_codes", []):
                    code = diag.get("code", "")
                    if code:
                        unique_diagnosis_codes.add(code)
            
            # Print summary
            batch_status = medical_extraction.get("llm_call_status", "unknown")
            print(f"📊 Medical Summary: {len(unique_service_codes)} Service Codes | {len(unique_diagnosis_codes)} ICD-10 Codes | {len(medical_records)} Records | Status: {batch_status.upper()}")
            
            # Print ICD-10 Diagnosis Codes
            if diagnosis_meanings and medical_records:
                print(f"\n🩺 ICD-10 DIAGNOSIS CODES:")
                print(f"{'Code':<10} {'Medical Meaning':<50} {'Date':<12}")
                print("-" * 75)
                
                diagnosis_data = []
                for record in medical_records:
                    claim_date = record.get("clm_rcvd_dt", "Unknown")
                    for diag in record.get("diagnosis_codes", []):
                        code = diag.get("code", "")
                        if code in diagnosis_meanings:
                            diagnosis_data.append({
                                "code": code,
                                "meaning": diagnosis_meanings[code],
                                "claim_date": claim_date
                            })
                
                # Sort and display top 15
                sorted_data = sorted(diagnosis_data, key=lambda x: x['claim_date'], reverse=True)
                for item in sorted_data[:15]:
                    code = item['code'][:9]
                    meaning = item['meaning'][:49]
                    date = str(item['claim_date'])[:11]
                    print(f"{code:<10} {meaning:<50} {date:<12}")
                
                if len(diagnosis_data) > 15:
                    print(f"... and {len(diagnosis_data) - 15} more diagnosis records")
                
                # Most frequent codes
                code_counts = Counter(item["code"] for item in diagnosis_data)
                print(f"\n📈 Most Frequent ICD-10 Codes:")
                for code, count in code_counts.most_common(3):
                    meaning = diagnosis_meanings.get(code, "Unknown")[:40]
                    print(f"  • {code} ({count}x): {meaning}")
            
            # Print Service Codes
            if service_meanings and medical_records:
                print(f"\n🏥 MEDICAL SERVICE CODES:")
                print(f"{'Code':<12} {'Service Description':<50} {'Date':<12}")
                print("-" * 77)
                
                service_data = []
                for record in medical_records:
                    service_end_date = record.get("clm_line_srvc_end_dt", "Unknown")
                    service_code = record.get("hlth_srvc_cd", "")
                    
                    if service_code and service_code in service_meanings:
                        service_data.append({
                            "code": service_code,
                            "meaning": service_meanings[service_code],
                            "end_date": service_end_date
                        })
                
                # Sort and display top 10
                sorted_data = sorted(service_data, key=lambda x: x['end_date'], reverse=True)
                for item in sorted_data[:10]:
                    code = item['code'][:11]
                    meaning = item['meaning'][:49]
                    date = str(item['end_date'])[:11]
                    print(f"{code:<12} {meaning:<50} {date:<12}")
                
                if len(service_data) > 10:
                    print(f"... and {len(service_data) - 10} more service records")
            
        except Exception as e:
            print(f"❌ Error displaying medical codes: {e}")

    def _print_pharmacy_codes_table(self, pharmacy_extraction: Dict[str, Any]):
        """Print formatted pharmacy codes table"""
        try:
            pharmacy_meanings = pharmacy_extraction.get("code_meanings", {})
            ndc_meanings = pharmacy_meanings.get("ndc_code_meanings", {})
            med_meanings = pharmacy_meanings.get("medication_meanings", {})
            pharmacy_records = pharmacy_extraction.get("ndc_records", [])
            
            # Calculate metrics
            unique_ndc_codes = set()
            unique_medications = set()
            
            for record in pharmacy_records:
                ndc_code = record.get("ndc", "")
                if ndc_code:
                    unique_ndc_codes.add(ndc_code)
                
                med_name = record.get("lbl_nm", "")
                if med_name:
                    unique_medications.add(med_name)
            
            # Print summary
            batch_status = pharmacy_extraction.get("llm_call_status", "unknown")
            print(f"📊 Pharmacy Summary: {len(unique_ndc_codes)} NDC Codes | {len(unique_medications)} Medications | {len(pharmacy_records)} Records | Status: {batch_status.upper()}")
            
            # Print NDC Codes
            if ndc_meanings and pharmacy_records:
                print(f"\n💊 NDC MEDICATION CODES:")
                print(f"{'NDC Code':<15} {'Medication Description':<45} {'Fill Date':<12}")
                print("-" * 75)
                
                ndc_data = []
                for record in pharmacy_records:
                    ndc_code = record.get("ndc", "")
                    fill_date = record.get("rx_filled_dt", "Unknown")
                    
                    if ndc_code and ndc_code in ndc_meanings:
                        ndc_data.append({
                            "ndc": ndc_code,
                            "meaning": ndc_meanings[ndc_code],
                            "fill_date": fill_date
                        })
                
                # Sort and display top 10
                sorted_data = sorted(ndc_data, key=lambda x: x['fill_date'], reverse=True)
                for item in sorted_data[:10]:
                    ndc = item['ndc'][:14]
                    meaning = item['meaning'][:44]
                    date = str(item['fill_date'])[:11]
                    print(f"{ndc:<15} {meaning:<45} {date:<12}")
                
                if len(ndc_data) > 10:
                    print(f"... and {len(ndc_data) - 10} more NDC records")
            
            # Print Medications
            if med_meanings and pharmacy_records:
                print(f"\n💉 MEDICATION NAMES:")
                print(f"{'Medication':<25} {'Therapeutic Description':<45} {'Fill Date':<12}")
                print("-" * 85)
                
                med_data = []
                for record in pharmacy_records:
                    med_name = record.get("lbl_nm", "")
                    fill_date = record.get("rx_filled_dt", "Unknown")
                    
                    if med_name and med_name in med_meanings:
                        med_data.append({
                            "med_name": med_name,
                            "meaning": med_meanings[med_name],
                            "fill_date": fill_date
                        })
                
                # Sort and display top 10
                sorted_data = sorted(med_data, key=lambda x: x['fill_date'], reverse=True)
                for item in sorted_data[:10]:
                    med = item['med_name'][:24]
                    meaning = item['meaning'][:44]
                    date = str(item['fill_date'])[:11]
                    print(f"{med:<25} {meaning:<45} {date:<12}")
                
                if len(med_data) > 10:
                    print(f"... and {len(med_data) - 10} more medication records")
                
                # Most frequent medications
                med_counts = Counter(item["med_name"] for item in med_data)
                print(f"\n📈 Most Frequent Medications:")
                for med, count in med_counts.most_common(3):
                    meaning = med_meanings.get(med, "Unknown")[:40]
                    print(f"  • {med} ({count}x): {meaning}")
            
        except Exception as e:
            print(f"❌ Error displaying pharmacy codes: {e}")

    # ===== HELPER METHODS =====

    def _create_comprehensive_trajectory_prompt_with_evaluation(self, medical_data: Dict, pharmacy_data: Dict, mcid_data: Dict,
                                                               medical_extraction: Dict, pharmacy_extraction: Dict,
                                                               entities: Dict) -> str:
        """Create comprehensive trajectory prompt with evaluation questions"""

        medical_summary = self._extract_medical_summary(medical_data, medical_extraction)
        pharmacy_summary = self._extract_pharmacy_summary(pharmacy_data, pharmacy_extraction)

        return f"""You are Dr. TrajectoryAI, a comprehensive healthcare analyst conducting detailed patient health trajectory analysis with predictive modeling capabilities.

**COMPREHENSIVE PATIENT CLAIMS DATA:**

**MEDICAL CLAIMS SUMMARY:**
{medical_summary}

**PHARMACY CLAIMS SUMMARY:**
{pharmacy_summary}

**ENHANCED HEALTH ENTITIES:**
{json.dumps(entities, indent=2)}

**COMPREHENSIVE HEALTH TRAJECTORY ANALYSIS WITH PREDICTIVE EVALUATION:**

Conduct a thorough analysis addressing these critical healthcare evaluation questions:

## 🔮 RISK PREDICTION (Clinical Outcomes)
**1. Chronic Disease Risk Assessment:**
- Based on this person's medical and pharmacy history, assess the risk of developing chronic diseases like diabetes, hypertension, COPD, or chronic kidney disease?
- Analyze current ICD-10 codes and medication patterns for disease progression indicators

**2. Hospitalization & Readmission Risk:**
- What is the likelihood that this person will be hospitalized or readmitted in the next 6–12 months?
- Review service utilization patterns and medication adherence indicators

**3. Emergency vs Outpatient Care Risk:**
- Is this person at risk of using the emergency room instead of outpatient care?
- Analyze healthcare utilization patterns from claims data

**4. Medication Adherence Risk:**
- How likely is this person to stop taking prescribed medications?
- Review prescription fill patterns and therapeutic gaps

**5. Serious Event Risk:**
- Does this person have a high risk of serious events like stroke, heart attack, or other complications due to comorbidities?
- Analyze cardiovascular risk factors and medication management

## 💰 COST & UTILIZATION PREDICTION
**6. High-Cost Claimant Prediction:**
- Is this person likely to become a high-cost claimant next year?
- Analyze current utilization trends and cost drivers

**7. Healthcare Cost Estimation:**
- Can you estimate this person's future healthcare costs (per month or per year)?
- Project based on current utilization patterns and medication costs

**8. Care Setting Prediction:**
- Is this person more likely to need inpatient hospital care or outpatient care in the future?
- Review current care patterns and complexity indicators

## 🔍 FRAUD, WASTE & ABUSE (FWA) DETECTION
**9. Claims Anomaly Detection:**
- Do this person's medical or pharmacy claims show any anomalies that could indicate errors or unusual patterns?
- Review for inconsistent diagnoses, unusual prescription patterns, or billing irregularities

**10. Prescribing Pattern Analysis:**
- Are there any unusual prescribing or billing patterns related to this person's records?
- Examine medication combinations and prescribing frequency

## 🎯 PERSONALIZED CARE MANAGEMENT
**11. Risk Segmentation:**
- How should this person be segmented — healthy, rising risk, chronic but stable, or high-cost/critical?
- Provide risk stratification based on comprehensive data analysis

**12. Preventive Care Recommendations:**
- What preventive screenings, wellness programs, or lifestyle changes should be recommended as the next best action?
- Identify specific care gaps and opportunities

**13. Care Gap Analysis:**
- Does this person have any care gaps, such as missed checkups, cancer screenings, or vaccinations?
- Review claims for preventive care compliance

## 💊 PHARMACY-SPECIFIC PREDICTIONS
**14. Polypharmacy Risk:**
- Is this person at risk of polypharmacy (taking too many medications or unsafe combinations)?
- Analyze current medication regimen for interactions and complexity

**15. Therapy Escalation:**
- Is this person likely to switch to higher-cost specialty drugs or need therapy escalation soon?
- Review current medications for potential progression patterns

**16. Specialty Drug Prediction:**
- Is it likely that this person will need expensive biologics or injectables in the future?
- Assess disease progression and current therapeutic approaches

## 🔬 ADVANCED / STRATEGIC PREDICTIONS
**17. Disease Progression Modeling:**
- Can you model how this person's disease might progress over time (for example: diabetes → complications → hospitalizations)?
- Create trajectory model based on current conditions and medications

**18. Quality Metrics Impact:**
- Does this person have any care gaps that could affect quality metrics (like HEDIS or STAR ratings)?
- Identify opportunities for quality measure improvement

**19. Population Health Risk:**
- Based on available data, how might this person's long-term health contribute to population-level risk?
- Assess impact on overall population health management

**COMPREHENSIVE ANALYSIS REQUIREMENTS:**
- Address each evaluation question using specific data from medical and pharmacy claims
- Reference exact ICD-10 codes, NDC codes, and claim dates
- Provide risk percentages and likelihood assessments where possible
- Include temporal analysis showing health progression over time
- Offer specific, actionable recommendations for each identified risk
- Create predictive models based on available clinical indicators

**DELIVERABLE:**
Provide a comprehensive 800-1000 word health trajectory analysis that addresses all evaluation questions with specific data references, risk assessments, and actionable recommendations for care management and risk mitigation.

**ANALYSIS FOCUS:**
Use ALL available claims data to create the most comprehensive predictive health assessment possible, addressing every evaluation question with evidence-based analysis and specific recommendations."""

    def _create_comprehensive_summary_prompt(self, trajectory_analysis: str, entities: Dict,
                                           medical_extraction: Dict, pharmacy_extraction: Dict) -> str:
        """Create comprehensive summary prompt"""

        return f"""Based on the comprehensive health trajectory analysis, create an executive summary for healthcare decision-makers.

**COMPREHENSIVE HEALTH TRAJECTORY ANALYSIS:**
{trajectory_analysis}

**ENHANCED HEALTH ENTITIES:**
- Diabetes: {entities.get('diabetics', 'unknown')}
- Age Group: {entities.get('age_group', 'unknown')}
- Smoking Status: {entities.get('smoking', 'unknown')}
- Blood Pressure: {entities.get('blood_pressure', 'unknown')}
- Medical Conditions: {len(entities.get('medical_conditions', []))}
- Medications: {len(entities.get('medications_identified', []))}

**CLAIMS DATA SUMMARY:**
- Medical Records: {len(medical_extraction.get('hlth_srvc_records', []))}
- Diagnosis Codes: {medical_extraction.get('extraction_summary', {}).get('total_diagnosis_codes', 0)}
- Pharmacy Records: {len(pharmacy_extraction.get('ndc_records', []))}

**EXECUTIVE SUMMARY REQUIREMENTS:**

Create a comprehensive summary with:

## 🏥 CURRENT HEALTH STATUS
[2-3 sentences summarizing overall health condition and key findings]

## 🚨 PRIORITY RISK FACTORS
[Bullet points of highest priority risks requiring immediate attention]

## 💰 COST & UTILIZATION INSIGHTS
[Key findings about healthcare costs and utilization patterns]

## 🎯 CARE MANAGEMENT RECOMMENDATIONS
[Specific actionable recommendations for care management teams]

## 📈 PREDICTIVE INSIGHTS
[Key predictions about future health outcomes and costs]

## ⚠️ IMMEDIATE ACTION ITEMS
[Priority items requiring immediate clinical attention]

**FORMAT:** Professional healthcare executive summary, 400-500 words, focusing on actionable insights for care management and clinical decision-making."""

    def _extract_enhanced_heart_attack_features(self, state: HealthAnalysisState) -> Dict[str, Any]:
        """Enhanced feature extraction for heart attack prediction"""
        try:
            features = {}

            # Age extraction
            deidentified_medical = state.get("deidentified_medical", {})
            patient_age = deidentified_medical.get("src_mbr_age", None)

            if patient_age and patient_age != "unknown":
                try:
                    age_value = int(float(str(patient_age)))
                    if 0 <= age_value <= 120:
                        features["Age"] = age_value
                    else:
                        features["Age"] = 50
                except:
                    features["Age"] = 50
            else:
                features["Age"] = 50

            # Gender extraction
            patient_data = state.get("patient_data", {})
            gender = str(patient_data.get("gender", "F")).upper()
            features["Gender"] = 1 if gender in ["M", "MALE", "1"] else 0

            # Entity-based features
            entity_extraction = state.get("entity_extraction", {})

            # Diabetes
            diabetes = str(entity_extraction.get("diabetics", "no")).lower()
            features["Diabetes"] = 1 if diabetes in ["yes", "true", "1"] else 0

            # Blood pressure
            blood_pressure = str(entity_extraction.get("blood_pressure", "unknown")).lower()
            features["High_BP"] = 1 if blood_pressure in ["managed", "diagnosed", "yes", "true", "1"] else 0

            # Smoking
            smoking = str(entity_extraction.get("smoking", "no")).lower()
            features["Smoking"] = 1 if smoking in ["yes", "true", "1"] else 0

            # Validate features
            for key in features:
                try:
                    features[key] = int(features[key])
                except:
                    if key == "Age":
                        features[key] = 50
                    else:
                        features[key] = 0

            enhanced_feature_summary = {
                "extracted_features": features,
                "feature_interpretation": {
                    "Age": f"{features['Age']} years old",
                    "Gender": "Male" if features["Gender"] == 1 else "Female",
                    "Diabetes": "Yes" if features["Diabetes"] == 1 else "No",
                    "High_BP": "Yes" if features["High_BP"] == 1 else "No",
                    "Smoking": "Yes" if features["Smoking"] == 1 else "No"
                },
                "data_sources": {
                    "age_source": "deidentified_medical.src_mbr_age",
                    "gender_source": "patient_data.gender",
                    "diabetes_source": "entity_extraction.diabetics",
                    "bp_source": "entity_extraction.blood_pressure",
                    "smoking_source": "entity_extraction.smoking"
                },
                "extraction_enhanced": True
            }

            logger.info(f"✅ Enhanced heart attack features: {enhanced_feature_summary['feature_interpretation']}")
            return enhanced_feature_summary

        except Exception as e:
            logger.error(f"Error in heart attack feature extraction: {e}")
            return {"error": f"Feature extraction failed: {str(e)}"}

    def _prepare_enhanced_fastapi_features(self, features: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Prepare feature data for FastAPI server call"""
        try:
            extracted_features = features.get("extracted_features", {})

            fastapi_features = {
                "age": int(extracted_features.get("Age", 50)),
                "gender": int(extracted_features.get("Gender", 0)),
                "diabetes": int(extracted_features.get("Diabetes", 0)),
                "high_bp": int(extracted_features.get("High_BP", 0)),
                "smoking": int(extracted_features.get("Smoking", 0))
            }

            # Validate age range
            if fastapi_features["age"] < 0 or fastapi_features["age"] > 120:
                logger.warning(f"Age {fastapi_features['age']} out of range, using default 50")
                fastapi_features["age"] = 50

            # Validate binary features
            binary_features = ["gender", "diabetes", "high_bp", "smoking"]
            for key in binary_features:
                if fastapi_features[key] not in [0, 1]:
                    logger.warning(f"{key} value {fastapi_features[key]} invalid, using 0")
                    fastapi_features[key] = 0

            logger.info(f"✅ FastAPI features prepared: {fastapi_features}")
            return fastapi_features

        except Exception as e:
            logger.error(f"Error preparing FastAPI features: {e}")
            return None

    def _call_heart_attack_prediction_sync(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous heart attack prediction call"""
        try:
            import requests

            logger.info(f"🔍 Heart attack prediction features: {features}")

            if not features:
                return {
                    "success": False,
                    "error": "No features provided for heart attack prediction"
                }

            heart_attack_url = self.config.heart_attack_api_url
            endpoints = [
                f"{heart_attack_url}/predict",
                f"{heart_attack_url}/predict-simple"
            ]

            params = {
                "age": int(features.get("age", 50)),
                "gender": int(features.get("gender", 0)),
                "diabetes": int(features.get("diabetes", 0)),
                "high_bp": int(features.get("high_bp", 0)),
                "smoking": int(features.get("smoking", 0))
            }

            logger.info(f"📤 Sending prediction request to {endpoints[0]}")

            try:
                response = requests.post(endpoints[0], json=params, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Prediction successful: {result}")
                    return {
                        "success": True,
                        "prediction_data": result,
                        "method": "POST_JSON_SYNC",
                        "endpoint": endpoints[0]
                    }
                else:
                    logger.warning(f"❌ First endpoint failed with status {response.status_code}")

            except requests.exceptions.ConnectionError as conn_error:
                logger.error(f"❌ Connection failed: {conn_error}")
                return {
                    "success": False,
                    "error": f"Cannot connect to heart attack prediction server. Make sure the server is running."
                }
            except Exception as request_error:
                logger.warning(f"❌ Request failed: {str(request_error)}")

            try:
                logger.info(f"🔄 Trying fallback endpoint: {endpoints[1]}")
                response = requests.post(endpoints[1], params=params, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Fallback prediction successful: {result}")
                    return {
                        "success": True,
                        "prediction_data": result,
                        "method": "POST_PARAMS_SYNC",
                        "endpoint": endpoints[1]
                    }
                else:
                    error_text = response.text
                    logger.error(f"❌ All endpoints failed. Status {response.status_code}: {error_text}")
                    return {
                        "success": False,
                        "error": f"Heart attack prediction server error {response.status_code}: {error_text}",
                        "tried_endpoints": endpoints
                    }

            except Exception as fallback_error:
                logger.error(f"❌ All prediction methods failed: {str(fallback_error)}")
                return {
                    "success": False,
                    "error": f"All prediction methods failed. Error: {str(fallback_error)}",
                    "tried_endpoints": endpoints
                }

        except Exception as general_error:
            logger.error(f"❌ Unexpected error in heart attack prediction: {general_error}")
            return {
                "success": False,
                "error": f"Heart attack prediction failed: {str(general_error)}"
            }

    def _extract_medical_summary(self, medical_data: Dict, medical_extraction: Dict) -> str:
        """Extract medical summary"""
        try:
            summary_parts = []

            age = medical_data.get("src_mbr_age", "unknown")
            zip_code = medical_data.get("src_mbr_zip_cd", "unknown")
            summary_parts.append(f"Patient Age: {age}, Location: {zip_code}")

            records = medical_extraction.get('hlth_srvc_records', [])
            summary_parts.append(f"Medical Records: {len(records)} health service records")

            if records:
                recent_diagnoses = []
                for record in records[:5]:
                    diag_codes = record.get('diagnosis_codes', [])
                    service_date = record.get('clm_rcvd_dt', 'Unknown date')
                    if diag_codes:
                        for diag in diag_codes[:2]:  # Limit diagnoses per record
                            code = diag.get('code', 'Unknown')
                            recent_diagnoses.append(f"Diagnosis {code} on {service_date}")
                    
                if recent_diagnoses:
                    summary_parts.append("Recent Diagnoses: " + "; ".join(recent_diagnoses))

            return "\n".join(summary_parts)

        except Exception as e:
            return f"Medical data available but summary extraction failed: {str(e)}"

    def _extract_pharmacy_summary(self, pharmacy_data: Dict, pharmacy_extraction: Dict) -> str:
        """Extract pharmacy summary"""
        try:
            summary_parts = []

            records = pharmacy_extraction.get('ndc_records', [])
            summary_parts.append(f"Pharmacy Records: {len(records)} medication records")

            if records:
                recent_meds = []
                for record in records[:5]:
                    ndc_code = record.get('ndc', 'Unknown')
                    label_name = record.get('lbl_nm', 'Unknown medication')
                    fill_date = record.get('rx_filled_dt', 'Unknown date')
                    recent_meds.append(f"{label_name} (NDC: {ndc_code}) filled on {fill_date}")
                    
                if recent_meds:
                    summary_parts.append("Recent Medications: " + "; ".join(recent_meds))

            return "\n".join(summary_parts)

        except Exception as e:
            return f"Pharmacy data available but summary extraction failed: {str(e)}"

    def get_code_explanations_for_record(self, record: Dict[str, Any], record_type: str = "medical") -> Dict[str, Any]:
        """Get code explanations for records"""
        explanations = {}

        try:
            if record_type == "medical":
                service_code = record.get("hlth_srvc_cd")
                if service_code:
                    explanations["service_code_explanation"] = self.data_processor.get_service_code_explanation_isolated(service_code)

                diagnosis_codes = record.get("diagnosis_codes", [])
                explanations["diagnosis_explanations"] = []
                for diag in diagnosis_codes:
                    diag_code = diag.get("code")
                    if diag_code:
                        explanation = self.data_processor.get_diagnosis_code_explanation_isolated(diag_code)
                        explanations["diagnosis_explanations"].append({
                            "code": diag_code,
                            "explanation": explanation,
                            "position": diag.get("position", 1)
                        })

            elif record_type == "pharmacy":
                ndc_code = record.get("ndc")
                if ndc_code:
                    explanations["ndc_explanation"] = self.data_processor.get_ndc_code_explanation_isolated(ndc_code)

                medication = record.get("lbl_nm")
                if medication:
                    explanations["medication_explanation"] = self.data_processor.get_medication_explanation_isolated(medication)

        except Exception as e:
            logger.warning(f"Error getting code explanations: {e}")
            explanations["error"] = f"Could not get explanations: {str(e)}"

        return explanations

    def test_llm_connection(self) -> Dict[str, Any]:
        """Test Snowflake Cortex API connection"""
        return self.api_integrator.test_healthcare_llm_connection()

    def test_backend_connection(self) -> Dict[str, Any]:
        """Test backend server connection"""
        return self.api_integrator.test_backend_connection_enhanced()

    def run_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the enhanced health analysis workflow using LangGraph"""

        initial_state = HealthAnalysisState(
            patient_data=patient_data,
            mcid_output={},
            medical_output={},
            pharmacy_output={},
            token_output={},
            deidentified_medical={},
            deidentified_pharmacy={},
            deidentified_mcid={},
            medical_extraction={},
            pharmacy_extraction={},
            entity_extraction={},
            health_trajectory="",
            final_summary="",
            heart_attack_prediction={},
            heart_attack_risk_score=0.0,
            heart_attack_features={},
            chatbot_ready=False,
            chatbot_context={},
            chat_history=[],
            graph_generation_ready=False,
            code_meanings_tables={},
            current_step="",
            errors=[],
            retry_count=0,
            processing_complete=False,
            step_status={}
        )

        try:
            config_dict = {"configurable": {"thread_id": f"enhanced_health_analysis_{datetime.now().timestamp()}"}}

            logger.info("🚀 Starting Enhanced LangGraph workflow with graph generation and code meanings display...")

            final_state = self.graph.invoke(initial_state, config=config_dict)

            results = {
                "success": final_state["processing_complete"] and not final_state["errors"],
                "patient_data": final_state["patient_data"],
                "api_outputs": {
                    "mcid": final_state["mcid_output"],
                    "medical": final_state["medical_output"],
                    "pharmacy": final_state["pharmacy_output"],
                    "token": final_state["token_output"]
                },
                "deidentified_data": {
                    "medical": final_state["deidentified_medical"],
                    "pharmacy": final_state["deidentified_pharmacy"],
                    "mcid": final_state["deidentified_mcid"]
                },
                "structured_extractions": {
                    "medical": final_state["medical_extraction"],
                    "pharmacy": final_state["pharmacy_extraction"]
                },
                "entity_extraction": final_state["entity_extraction"],
                "health_trajectory": final_state["health_trajectory"],
                "final_summary": final_state["final_summary"],
                "code_meanings_tables": final_state["code_meanings_tables"],
                "heart_attack_prediction": final_state["heart_attack_prediction"],
                "heart_attack_risk_score": final_state["heart_attack_risk_score"],
                "heart_attack_features": final_state["heart_attack_features"],
                "chatbot_ready": final_state["chatbot_ready"],
                "chatbot_context": final_state["chatbot_context"],
                "chat_history": final_state["chat_history"],
                "graph_generation_ready": final_state["graph_generation_ready"],
                "errors": final_state["errors"],
                "processing_steps_completed": self._count_completed_steps(final_state),
                "step_status": final_state["step_status"],
                "langgraph_used": True,
                "comprehensive_analysis": True,
                "enhanced_chatbot": True,
                "graph_generation_ready": True,
                "batch_code_meanings": True,
                "code_meanings_display_integrated": True,
                "enhancement_version": "v9.0_comprehensive_with_code_display"
            }

            if results["success"]:
                logger.info("✅ Enhanced LangGraph analysis completed successfully with graph generation and code meanings display!")
                logger.info(f"💬 Enhanced chatbot ready: {results['chatbot_ready']}")
                logger.info(f"📊 Graph generation ready: {results['graph_generation_ready']}")
                logger.info(f"📋 Code meanings display integrated: {results['code_meanings_display_integrated']}")
                logger.info(f"📋 Code meanings tables: Medical={results['code_meanings_tables'].get('medical_table_generated', False)}, Pharmacy={results['code_meanings_tables'].get('pharmacy_table_generated', False)}")
            else:
                logger.error(f"❌ Enhanced LangGraph analysis failed: {final_state['errors']}")

            return results

        except Exception as e:
            logger.error(f"Fatal error in Enhanced LangGraph workflow: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "patient_data": patient_data,
                "errors": [str(e)],
                "processing_steps_completed": 0,
                "langgraph_used": True,
                "comprehensive_analysis": False,
                "enhanced_chatbot": False,
                "graph_generation_ready": False,
                "batch_code_meanings": False,
                "code_meanings_display_integrated": False,
                "enhancement_version": "v9.0_comprehensive_with_code_display"
            }

    def _count_completed_steps(self, state: HealthAnalysisState) -> int:
        """Count processing steps completed"""
        steps = 0
        if state.get("mcid_output"): steps += 1
        if state.get("deidentified_medical") and not state.get("deidentified_medical", {}).get("error"): steps += 1
        if state.get("medical_extraction") or state.get("pharmacy_extraction"): steps += 1
        if state.get("entity_extraction"): steps += 1
        if state.get("health_trajectory"): steps += 1
        if state.get("final_summary"): steps += 1
        if state.get("code_meanings_tables"): steps += 1
        if state.get("heart_attack_prediction"): steps += 1
        if state.get("chatbot_ready"): steps += 1
        return steps

def main():
    """Enhanced Health Analysis Agent with Comprehensive Analysis, Graph Generation, and Integrated Code Meanings Tables"""
    
    print("🏥 Enhanced Health Analysis Agent v9.0 - Comprehensive Analysis with Integrated Code Meanings Tables")
    print("✅ Enhanced features:")
    print("   📡 EnhancedHealthAPIIntegrator - Comprehensive API connectivity with graph generation")
    print("   🔧 EnhancedHealthDataProcessor - Advanced claims data processing with matplotlib code generation")
    print("   🏗️ HealthAnalysisAgent - Enhanced workflow orchestration with graph support")
    print("   💬 Enhanced chatbot - Complete claims data access with real-time graph generation")
    print("   📊 Advanced graph generation - Matplotlib code creation and execution")
    print("   🧠 Batch code meanings - LLM-powered medical and pharmacy code interpretation")
    print("   📋 Integrated code tables - Code meanings tables as LangGraph workflow step")
    print("   🎯 JSON output format - Tables included in structured JSON results")
    print()

    config = Config()
    print("📋 Enhanced Configuration:")
    print(f"   🌐 Snowflake API: {config.api_url}")
    print(f"   🤖 Model: {config.model}")
    print(f"   📡 Server: {config.fastapi_url}")
    print(f"   ❤️ Heart Attack ML API: {config.heart_attack_api_url}")
    print(f"   📈 Graph Generation: Advanced matplotlib support")
    print(f"   🔬 Batch Processing: Medical and pharmacy code meanings")
    print(f"   📋 Code Tables: Integrated as LangGraph workflow step")
    print(f"   🎯 JSON Format: Tables included in structured output")
    print()
    print("✅ Enhanced Health Agent ready for comprehensive analysis with integrated code meanings tables!")

    return "Enhanced Health Agent ready for integration"

if __name__ == "__main__":
    main()
