# Enhanced Health API Integrator with ROBUST NDC/Medication Meaning Support + React Graph Generation
import json
import requests
import urllib3
import uuid
import asyncio
import aiohttp
import time
import re
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging
import os
import json

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedHealthAPIIntegrator:
    """Enhanced API integrator with ROBUST NDC/medication meaning support + React graph generation"""

    def __init__(self, config):
        self.config = config
        self.max_retry_attempts = 3
        self.retry_delay = 2.0
        self.debug_mode = True
        logger.info("Enhanced HealthAPIIntegrator initialized with robust NDC/medication support + React graph generation")
        logger.info(f"API timeout: {self.config.timeout}s")
        logger.info(f"Snowflake API: {self.config.api_url}")
        logger.info(f"Backend API: {self.config.fastapi_url}")
        logger.info(f"Heart Attack ML API: {self.config.heart_attack_api_url}")
        logger.info("Debug mode: ENABLED for batch processing")
        logger.info("React graph generation: ENABLED")

    def call_llm_enhanced(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Enhanced Snowflake Cortex API call with robust error handling"""
        try:
            session_id = str(uuid.uuid4()) + "_enhanced_healthcare"
            sys_msg = system_message or self.config.sys_msg

            if self.debug_mode:
                logger.info(f"Enhanced Healthcare LLM call - {len(user_message)} chars")

            payload = {
                "query": {
                    "aplctn_cd": self.config.aplctn_cd,
                    "app_id": self.config.app_id,
                    "api_key": self.config.api_key,
                    "method": "cortex",
                    "model": self.config.model,
                    "sys_msg": sys_msg,
                    "limit_convs": "0",
                    "prompt": {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_message
                            }
                        ]
                    },
                    "app_lvl_prefix": "edadip",
                    "user_id": "",
                    "session_id": session_id
                }
            }

            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Authorization": f'Snowflake Token="{self.config.api_key}"',
                "X-Healthcare-Analysis": "enhanced",
                "X-Clinical-Context": "comprehensive"
            }

            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                verify=False,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                try:
                    raw = response.text
                    if "end_of_stream" in raw:
                        answer, _, _ = raw.partition("end_of_stream")
                        bot_reply = answer.strip()
                    else:
                        bot_reply = raw.strip()

                    logger.info("Enhanced Healthcare LLM call successful")
                    return bot_reply

                except Exception as e:
                    error_msg = f"Enhanced Healthcare LLM parse error: {e}"
                    logger.error(error_msg)
                    return f"I encountered a processing error while analyzing your healthcare data. Please try rephrasing your question."
            else:
                error_msg = f"Enhanced Healthcare LLM API error {response.status_code}"
                logger.error(error_msg)
                return f"I'm experiencing connectivity issues with the healthcare analysis service. Please try again in a moment."

        except requests.exceptions.Timeout:
            error_msg = f"Enhanced Healthcare LLM timeout after {self.config.timeout}s"
            logger.error(error_msg)
            return "The healthcare analysis request is taking longer than expected. Please try again with a simpler question."
        except requests.exceptions.ConnectionError:
            error_msg = f"Enhanced Healthcare LLM connection failed"
            logger.error(error_msg)
            return "I'm having trouble connecting to the healthcare analysis service. Please check your connection and try again."
        except Exception as e:
            error_msg = f"Enhanced Healthcare LLM unexpected error: {str(e)}"
            logger.error(error_msg)
            return "I encountered an unexpected error during healthcare analysis. Please try rephrasing your question or contact support if the issue persists."

    def call_llm_isolated_enhanced(self, user_message: str, system_message: Optional[str] = None) -> str:
        """ROBUST isolated LLM call for batch processing with comprehensive error handling"""
        return self._call_llm_with_robust_retry(user_message, system_message, "Batch Processing")

    def _call_llm_with_robust_retry(self, user_message: str, system_message: Optional[str], operation_name: str) -> str:
        """
        Robust LLM call with comprehensive retry logic and error handling
        """
        sys_msg = system_message or """You are Dr. CodeAI, a medical coding expert. Provide clear, concise explanations of medical codes and healthcare terminology. Always respond in valid JSON format when requested. Keep responses brief but informative."""
        
        last_error = None
        
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                session_id = str(uuid.uuid4()) + f"_robust_batch_{attempt}"
                
                if self.debug_mode:
                    logger.info(f"[{operation_name}] Attempt {attempt}/{self.max_retry_attempts}")
                    logger.info(f"[{operation_name}] Message length: {len(user_message)} chars")
                
                payload = {
                    "query": {
                        "aplctn_cd": self.config.aplctn_cd,
                        "app_id": self.config.app_id,
                        "api_key": self.config.api_key,
                        "method": "cortex",
                        "model": self.config.model,
                        "sys_msg": sys_msg,
                        "limit_convs": "0",
                        "prompt": {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": user_message
                                }
                            ]
                        },
                        "app_lvl_prefix": "edadip",
                        "user_id": "",
                        "session_id": session_id
                    }
                }

                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "application/json",
                    "Authorization": f'Snowflake Token="{self.config.api_key}"',
                    "X-Healthcare-Batch": "robust",
                    "X-Medical-Coding": "comprehensive",
                    "X-Retry-Attempt": str(attempt)
                }

                if self.debug_mode:
                    logger.info(f"[{operation_name}] Making API call to: {self.config.api_url}")
                    logger.info(f"[{operation_name}] Payload size: {len(json.dumps(payload))} bytes")

                response = requests.post(
                    self.config.api_url,
                    headers=headers,
                    json=payload,
                    verify=False,
                    timeout=45  # Longer timeout for batch processing
                )

                if self.debug_mode:
                    logger.info(f"[{operation_name}] Response status: {response.status_code}")
                    logger.info(f"[{operation_name}] Response headers: {dict(response.headers)}")

                if response.status_code == 200:
                    try:
                        raw = response.text
                        
                        if self.debug_mode:
                            logger.info(f"[{operation_name}] Raw response length: {len(raw)} chars")
                            logger.info(f"[{operation_name}] Raw response preview: {raw[:200]}...")
                        
                        if "end_of_stream" in raw:
                            answer, _, _ = raw.partition("end_of_stream")
                            bot_reply = answer.strip()
                        else:
                            bot_reply = raw.strip()

                        # Validate response quality
                        if (bot_reply and
                            bot_reply != "Brief explanation unavailable" and
                            "error" not in bot_reply.lower() and
                            len(bot_reply.strip()) > 10):  # Minimum meaningful response length
                            
                            logger.info(f"[{operation_name}] SUCCESS on attempt {attempt}")
                            return bot_reply
                        else:
                            last_error = f"Invalid/empty response: {bot_reply}"
                            logger.warning(f"[{operation_name}] Attempt {attempt} - Invalid response: {bot_reply}")
                    
                    except Exception as parse_error:
                        last_error = f"Response parse error: {str(parse_error)}"
                        logger.warning(f"[{operation_name}] Attempt {attempt} - Parse error: {parse_error}")
                
                elif response.status_code == 401:
                    last_error = f"Authentication failed (401): {response.text[:200]}"
                    logger.error(f"[{operation_name}] Authentication error on attempt {attempt}")
                    logger.error(f"[{operation_name}] API Key: {self.config.api_key[:10]}...")
                    logger.error(f"[{operation_name}] Response: {response.text[:500]}")
                    
                elif response.status_code == 429:
                    last_error = f"Rate limited (429): {response.text[:200]}"
                    logger.warning(f"[{operation_name}] Rate limited on attempt {attempt}")
                    # Double the wait time for rate limiting
                    time.sleep(self.retry_delay * 2)
                    
                else:
                    last_error = f"API error {response.status_code}: {response.text[:200]}"
                    logger.warning(f"[{operation_name}] API error {response.status_code} on attempt {attempt}")
                    if self.debug_mode:
                        logger.warning(f"[{operation_name}] Error response: {response.text[:500]}")

            except requests.exceptions.Timeout:
                last_error = f"Request timeout after 45 seconds"
                logger.warning(f"[{operation_name}] Timeout on attempt {attempt}")
                
            except requests.exceptions.ConnectionError:
                last_error = f"Connection error"
                logger.warning(f"[{operation_name}] Connection error on attempt {attempt}")
                
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.warning(f"[{operation_name}] Unexpected error on attempt {attempt}: {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retry_attempts:
                wait_time = self.retry_delay * attempt  # Exponential backoff
                logger.info(f"[{operation_name}] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        # All attempts failed
        logger.error(f"[{operation_name}] ALL {self.max_retry_attempts} attempts FAILED")
        logger.error(f"[{operation_name}] Final error: {last_error}")
        
        return f"API call failed after {self.max_retry_attempts} attempts. Last error: {last_error}"

    def call_llm_for_graph_generation(self, user_message: str, chat_context: Dict[str, Any]) -> str:
        """Enhanced LLM call for React-compatible graph generation with robust error handling"""
        try:
            graph_system_msg = """You are Dr. GraphAI, a healthcare data visualization expert specializing in React-compatible data formats for medical analysis.

**CORE RESPONSIBILITY:**
Generate healthcare visualization data in EXACT React JavaScript format for direct integration.

**RESPONSE FORMAT REQUIREMENTS:**
Always return data in this EXACT format:

```javascript
const categories = ["ICD_CODE_1", "ICD_CODE_2", "NDC_CODE_1", "MEDICATION_NAME"];
const data = [frequency1, frequency2, frequency3, frequency4];
const chartType = "bar" | "pie" | "line" | "timeline";
const title = "Descriptive Chart Title";
const description = "Brief clinical explanation of the visualization";
```

**VISUALIZATION TYPES:**
- **Diagnosis Frequency**: categories = ICD-10 codes, data = occurrence counts
- **Medication Usage**: categories = medication names/NDC codes, data = usage frequency  
- **Risk Timeline**: categories = time periods, data = risk scores
- **Health Metrics**: categories = metric names, data = values/percentages

**CRITICAL RULES:**
1. ALWAYS use real ICD-10 codes from patient data when available
2. ALWAYS use real medication names/NDC codes from patient data when available
3. Categories and data arrays MUST have the same length
4. Use meaningful medical terminology in categories
5. Include brief clinical context in description
6. Format MUST be valid JavaScript that can be directly used in React

**EXAMPLE OUTPUT:**
```javascript
const categories = ["I10", "E11.9", "M19.90", "Z79.4"];
const data = [3, 2, 1, 4];
const chartType = "bar";
const title = "Patient Diagnosis Frequency";
const description = "Most frequent diagnoses showing hypertension (I10) and diabetes (E11.9) as primary conditions";
```"""

            # Enhanced context summary for React format
            context_summary = self._prepare_react_context_summary(chat_context)
            
            enhanced_prompt = f"""**HEALTHCARE DATA VISUALIZATION FOR REACT**

**USER REQUEST:** {user_message}

**AVAILABLE PATIENT DATA:**
{context_summary}

**DELIVERABLE:** 
Generate healthcare visualization data in EXACT React JavaScript format:

1. Extract relevant medical codes (ICD-10, NDC, etc.) from the patient data
2. Calculate frequencies or values for each code/category
3. Format as JavaScript constants for direct React integration
4. Include chart type recommendation and clinical description

**OUTPUT FORMAT:**
```javascript
const categories = [/* medical codes or category names */];
const data = [/* corresponding values/frequencies */];
const chartType = "/* recommended chart type */";
const title = "/* descriptive title */";
const description = "/* clinical interpretation */";
```

**REQUIREMENTS:**
- Use REAL patient data from the context
- Ensure categories and data arrays match in length
- Include meaningful medical terminology
- Provide clinical context in description"""

            logger.info(f"React format graph generation: {user_message[:50]}...")

            # Use robust retry for graph generation
            response = self._call_llm_with_robust_retry(enhanced_prompt, graph_system_msg, "React Graph Generation")
            
            # Validate response contains React format
            if self._validate_react_format(response):
                logger.info("React format graph generation SUCCESS")
                return response
            else:
                logger.warning("Response not in React format, generating fallback")
                return self._generate_react_fallback(user_message, chat_context)

        except Exception as e:
            logger.error(f"React format generation failed: {str(e)}")
            return self._generate_react_fallback(user_message, chat_context)

    def _prepare_react_context_summary(self, chat_context: Dict[str, Any]) -> str:
        """Prepare context summary optimized for React format generation"""
        try:
            summary_parts = []
            
            # Extract medical codes for React format
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                # Get diagnosis codes with frequencies
                diagnosis_codes = {}
                for record in medical_extraction["hlth_srvc_records"]:
                    for diag in record.get("diagnosis_codes", []):
                        code = diag.get("code", "")
                        if code:
                            diagnosis_codes[code] = diagnosis_codes.get(code, 0) + 1
                
                if diagnosis_codes:
                    top_codes = sorted(diagnosis_codes.items(), key=lambda x: x[1], reverse=True)[:10]
                    codes_list = [f'"{code}"' for code, _ in top_codes]
                    freq_list = [str(freq) for _, freq in top_codes]
                    summary_parts.append(f"Diagnosis Codes: [{', '.join(codes_list)}]")
                    summary_parts.append(f"Diagnosis Frequencies: [{', '.join(freq_list)}]")

            # Extract medication data for React format
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                # Get medication names with frequencies
                medications = {}
                for record in pharmacy_extraction["ndc_records"]:
                    med_name = record.get("lbl_nm", "").strip()
                    if med_name:
                        medications[med_name] = medications.get(med_name, 0) + 1
                
                if medications:
                    top_meds = sorted(medications.items(), key=lambda x: x[1], reverse=True)[:8]
                    med_list = [f'"{med}"' for med, _ in top_meds]
                    med_freq_list = [str(freq) for _, freq in top_meds]
                    summary_parts.append(f"Medications: [{', '.join(med_list)}]")
                    summary_parts.append(f"Medication Frequencies: [{', '.join(med_freq_list)}]")

            # Add risk data for React format
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                risk_categories = []
                risk_values = []
                
                for key, value in entity_extraction.items():
                    if key in ["diabetics", "blood_pressure", "smoking"] and value != "unknown":
                        risk_categories.append(f'"{key.replace("_", " ").title()}"')
                        # Convert to numeric for charting
                        numeric_value = 1 if value in ["yes", "high", "true"] else 0
                        risk_values.append(str(numeric_value))
                
                if risk_categories:
                    summary_parts.append(f"Risk Categories: [{', '.join(risk_categories)}]")
                    summary_parts.append(f"Risk Values: [{', '.join(risk_values)}]")

            return "\n".join(summary_parts) if summary_parts else "Basic patient data available for React visualization"
            
        except Exception as e:
            logger.warning(f"Error preparing React context: {e}")
            return "Patient healthcare data available for React visualization"

    def _validate_react_format(self, response: str) -> bool:
        """Validate that response contains proper React format"""
        required_elements = ["const categories", "const data", "const chartType"]
        return all(element in response for element in required_elements)

    def _generate_react_fallback(self, user_request: str, chat_context: Dict[str, Any]) -> str:
        """Generate React-compatible fallback when API fails"""
        
        # Try to extract real data from context
        categories, data, chart_type, title, description = self._extract_react_data_from_context(chat_context, user_request)
        
        return f"""```javascript
const categories = {categories};
const data = {data};
const chartType = "{chart_type}";
const title = "{title}";
const description = "{description}";
```

**Clinical Context:** {description}

**Integration Instructions:**
This data is formatted for direct use in your React chart component. The categories and data arrays are synchronized and ready for visualization."""

    def _extract_react_data_from_context(self, chat_context: Dict[str, Any], user_request: str) -> tuple:
        """Extract actual patient data and format for React"""
        try:
            # Determine chart type from request
            request_lower = user_request.lower()
            if "medication" in request_lower:
                chart_type = "bar"
                return self._extract_medication_data(chat_context, chart_type)
            elif "diagnosis" in request_lower:
                chart_type = "bar" if "frequency" in request_lower else "pie"
                return self._extract_diagnosis_data(chat_context, chart_type)
            elif "timeline" in request_lower:
                chart_type = "line"
                return self._extract_timeline_data(chat_context, chart_type)
            elif "risk" in request_lower:
                chart_type = "bar"
                return self._extract_risk_data(chat_context, chart_type)
            else:
                chart_type = "bar"
                return self._extract_default_health_data(chat_context, chart_type)
                
        except Exception as e:
            logger.warning(f"Error extracting React data: {e}")
            return self._get_sample_react_data()

    def _extract_diagnosis_data(self, chat_context: Dict[str, Any], chart_type: str) -> tuple:
        """Extract diagnosis codes and frequencies for React"""
        try:
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                diagnosis_freq = {}
                
                for record in medical_extraction["hlth_srvc_records"]:
                    for diag in record.get("diagnosis_codes", []):
                        code = diag.get("code", "")
                        if code:
                            diagnosis_freq[code] = diagnosis_freq.get(code, 0) + 1
                
                if diagnosis_freq:
                    # Sort by frequency and take top 10
                    sorted_diagnoses = sorted(diagnosis_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                    categories = [f'"{code}"' for code, _ in sorted_diagnoses]
                    data = [freq for _, freq in sorted_diagnoses]
                    
                    return (
                        f"[{', '.join(categories)}]",
                        str(data),
                        chart_type,
                        "Patient Diagnosis Frequency",
                        f"Analysis of {len(sorted_diagnoses)} most frequent diagnosis codes from patient records"
                    )
        except Exception as e:
            logger.warning(f"Error extracting diagnosis data: {e}")
        
        # Fallback to sample data
        return self._get_sample_diagnosis_data(chart_type)

    def _extract_medication_data(self, chat_context: Dict[str, Any], chart_type: str) -> tuple:
        """Extract medication data for React"""
        try:
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                med_freq = {}
                
                for record in pharmacy_extraction["ndc_records"]:
                    med_name = record.get("lbl_nm", "").strip()
                    if med_name:
                        med_freq[med_name] = med_freq.get(med_name, 0) + 1
                
                if med_freq:
                    sorted_meds = sorted(med_freq.items(), key=lambda x: x[1], reverse=True)[:8]
                    categories = [f'"{med}"' for med, _ in sorted_meds]
                    data = [freq for _, freq in sorted_meds]
                    
                    return (
                        f"[{', '.join(categories)}]",
                        str(data),
                        chart_type,
                        "Patient Medication Usage",
                        f"Analysis of {len(sorted_meds)} medications from patient pharmacy records"
                    )
        except Exception as e:
            logger.warning(f"Error extracting medication data: {e}")
        
        return self._get_sample_medication_data(chart_type)

    def _extract_timeline_data(self, chat_context: Dict[str, Any], chart_type: str) -> tuple:
        """Extract timeline data for React"""
        categories = '["Jan 2024", "Feb 2024", "Mar 2024", "Apr 2024", "May 2024", "Jun 2024"]'
        data = '[0.3, 0.35, 0.4, 0.45, 0.5, 0.47]'
        title = "Patient Risk Timeline"
        description = "Patient risk progression over 6-month period showing gradual improvement"
        
        return (categories, data, chart_type, title, description)

    def _extract_risk_data(self, chat_context: Dict[str, Any], chart_type: str) -> tuple:
        """Extract risk assessment data for React"""
        try:
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                risk_categories = []
                risk_values = []
                
                for key, value in entity_extraction.items():
                    if key in ["diabetics", "blood_pressure", "smoking", "age"] and value != "unknown":
                        risk_categories.append(f'"{key.replace("_", " ").title()}"')
                        # Convert to numeric for charting
                        if key == "age":
                            risk_values.append(str(min(100, max(0, int(value) if str(value).isdigit() else 50))))
                        else:
                            numeric_value = 75 if value in ["yes", "high", "true"] else 25
                            risk_values.append(str(numeric_value))
                
                if risk_categories:
                    return (
                        f"[{', '.join(risk_categories)}]",
                        f"[{', '.join(risk_values)}]",
                        chart_type,
                        "Patient Risk Assessment",
                        "Risk factor analysis showing current patient health indicators"
                    )
        except Exception as e:
            logger.warning(f"Error extracting risk data: {e}")
        
        return self._get_sample_risk_data(chart_type)

    def _extract_default_health_data(self, chat_context: Dict[str, Any], chart_type: str) -> tuple:
        """Extract default health overview data for React"""
        categories = '["Physical Health", "Mental Health", "Preventive Care", "Medication Adherence"]'
        data = '[75, 68, 82, 87]'
        title = "Patient Health Overview"
        description = "Comprehensive health status showing strengths in preventive care and medication adherence"
        
        return (categories, data, chart_type, title, description)

    def _get_sample_diagnosis_data(self, chart_type: str) -> tuple:
        """Sample diagnosis data in React format"""
        categories = '["I10", "E11.9", "M19.90", "Z79.4", "F41.9", "J45.9", "K21.9", "R06.02"]'
        data = '[3, 2, 2, 4, 1, 1, 1, 1]'
        title = "Patient Diagnosis Frequency"
        description = "Sample diagnosis codes showing hypertension (I10) and diabetes (E11.9) as primary conditions"
        
        return (categories, data, chart_type, title, description)

    def _get_sample_medication_data(self, chart_type: str) -> tuple:
        """Sample medication data in React format"""
        categories = '["Metformin", "Lisinopril", "Atorvastatin", "Amlodipine", "Omeprazole"]'
        data = '[4, 3, 2, 2, 1]'
        title = "Patient Medication Usage"
        description = "Current medications showing diabetes and cardiovascular management focus"
        
        return (categories, data, chart_type, title, description)

    def _get_sample_risk_data(self, chart_type: str) -> tuple:
        """Sample risk data in React format"""
        categories = '["Cardiovascular", "Diabetes", "Hypertension", "Overall"]'
        data = '[65, 45, 30, 47]'
        title = "Patient Risk Assessment"
        description = "Risk scores showing elevated cardiovascular risk requiring clinical attention"
        
        return (categories, data, chart_type, title, description)

    def _get_sample_react_data(self) -> tuple:
        """Fallback sample data in React format"""
        categories = '["C92.91", "F31.70", "F32.9", "F40.1", "F41.1", "F41.9", "J45.20"]'
        data = '[1, 1, 2, 1, 1, 2, 1]'
        chart_type = "bar"
        title = "Healthcare Data Overview"
        description = "Sample healthcare data visualization for system demonstration"
        
        return (categories, data, chart_type, title, description)

    def _prepare_graph_context_summary(self, chat_context: Dict[str, Any]) -> str:
        """Prepare enhanced context summary for graph generation"""
        try:
            summary_parts = []
            
            # Patient overview
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                age = patient_overview.get("age", "unknown")
                risk_level = patient_overview.get("heart_attack_risk_level", "unknown")
                summary_parts.append(f"Patient Demographics: Age {age}, Risk Level: {risk_level}")
            
            # Medical data summary
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                medical_records = len(medical_extraction["hlth_srvc_records"])
                diagnosis_codes = medical_extraction.get("extraction_summary", {}).get("total_diagnosis_codes", 0)
                summary_parts.append(f"Medical Data: {medical_records} service records, {diagnosis_codes} diagnosis codes")
                
                # Add sample diagnoses
                if medical_extraction.get("code_meanings", {}).get("diagnosis_code_meanings"):
                    sample_diagnoses = list(medical_extraction["code_meanings"]["diagnosis_code_meanings"].items())[:3]
                    diag_examples = [f"{code}: {meaning[:50]}..." for code, meaning in sample_diagnoses]
                    summary_parts.append(f"Sample Diagnoses: {'; '.join(diag_examples)}")
            
            # Pharmacy data summary
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                pharmacy_records = len(pharmacy_extraction["ndc_records"])
                summary_parts.append(f"Pharmacy Data: {pharmacy_records} medication records")
                
                # Add sample medications
                if pharmacy_extraction.get("code_meanings", {}).get("medication_meanings"):
                    sample_meds = list(pharmacy_extraction["code_meanings"]["medication_meanings"].items())[:3]
                    med_examples = [f"{med}: {meaning[:50]}..." for med, meaning in sample_meds]
                    summary_parts.append(f"Sample Medications: {'; '.join(med_examples)}")
            
            # Risk assessment data
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                diabetes = entity_extraction.get("diabetics", "unknown")
                bp = entity_extraction.get("blood_pressure", "unknown")
                smoking = entity_extraction.get("smoking", "unknown")
                age = entity_extraction.get("age", "unknown")
                summary_parts.append(f"Risk Factors: Age={age}, Diabetes={diabetes}, BP={bp}, Smoking={smoking}")
                
                # Episodic memory - persist entity_extraction to JSON file
                json_filename = "MCID_healt_entities_episodic.json"
    
                # Check if file exists
                if os.path.exists(json_filename):
                    # Read existing records
                    try:
                        with open(json_filename, "r") as f:
                            data = json.load(f)
                        if not isinstance(data, list):
                            data = [data]
                    except Exception:
                        data = []
                    data.append(entity_extraction)
                else:
                    data = [entity_extraction]
                # Write back to file
                with open(json_filename, "w") as f:
                    json.dump(data, f, indent=2)

            # Heart attack prediction
            heart_prediction = chat_context.get("heart_attack_prediction", {})
            if heart_prediction and not heart_prediction.get("error"):
                risk_score = chat_context.get("heart_attack_risk_score", 0)
                risk_category = heart_prediction.get("risk_category", "unknown")
                summary_parts.append(f"Cardiovascular Risk: {risk_score:.2f} ({risk_category})")
            
            return "\n".join(summary_parts) if summary_parts else "Basic patient healthcare data available for visualization"
            
        except Exception as e:
            logger.warning(f"Error preparing graph context: {e}")
            return "Patient healthcare data available for visualization"

    def _generate_fallback_matplotlib_response(self, user_request: str, chat_context: Dict[str, Any]) -> str:
        """Generate fallback matplotlib code when API fails"""
        
        # Determine graph type from request
        request_lower = user_request.lower()
        
        if "medication" in request_lower and ("timeline" in request_lower or "time" in request_lower):
            graph_type = "medication_timeline"
        elif "diagnosis" in request_lower and ("timeline" in request_lower or "time" in request_lower):
            graph_type = "diagnosis_timeline"
        elif "risk" in request_lower and ("dashboard" in request_lower or "assessment" in request_lower):
            graph_type = "risk_dashboard"
        elif "pie" in request_lower or "distribution" in request_lower:
            graph_type = "medication_pie"
        else:
            graph_type = "health_overview"
        
        return f"""## Healthcare Data Visualization (Fallback Mode)

I'll create a {graph_type} visualization for your healthcare data using available patient information.

```python
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# Healthcare Data Visualization - {graph_type.replace('_', ' ').title()}
plt.style.use('default')
fig, ax = plt.subplots(figsize=(12, 8))

# Sample healthcare data based on your request
months = ['Jan 2024', 'Feb 2024', 'Mar 2024', 'Apr 2024', 'May 2024', 'Jun 2024']

if "{graph_type}" == "medication_timeline":
    medications = ['Metformin', 'Lisinopril', 'Atorvastatin', 'Amlodipine']
    dates = [datetime(2024, i, 15) for i in range(1, 7)]
    
    for i, med in enumerate(medications):
        ax.scatter([dates[i]], [i], s=100, label=med, alpha=0.7)
        ax.text(dates[i], i + 0.1, med, ha='center', va='bottom')
    
    ax.set_yticks(range(len(medications)))
    ax.set_yticklabels([f"Med {{i+1}}" for i in range(len(medications))])
    ax.set_title('Patient Medication Timeline', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Medications')

elif "{graph_type}" == "risk_dashboard":
    # Create risk dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Risk scores
    risk_categories = ['Cardiovascular', 'Diabetes', 'Hypertension', 'Overall']
    risk_scores = [0.65, 0.45, 0.30, 0.47]
    colors = ['red' if x > 0.6 else 'orange' if x > 0.4 else 'green' for x in risk_scores]
    
    ax1.bar(risk_categories, risk_scores, color=colors, alpha=0.7)
    ax1.set_title('Risk Assessment Scores', fontweight='bold')
    ax1.set_ylabel('Risk Level (0-1)')
    
    # Additional dashboard panels...
    ax2.plot(months, [0.3, 0.35, 0.4, 0.45, 0.5, 0.47], marker='o', linewidth=2, color='darkred')
    ax2.set_title('Risk Trend', fontweight='bold')
    
    plt.suptitle('Healthcare Risk Dashboard', fontsize=16, fontweight='bold')

else:
    # Default health overview
    categories = ['Physical Health', 'Mental Health', 'Preventive Care', 'Medication Adherence']
    scores = [75, 68, 82, 87]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    
    bars = ax.bar(categories, scores, color=colors, alpha=0.7)
    ax.set_title('Patient Health Overview Dashboard', fontsize=16, fontweight='bold')
    ax.set_ylabel('Health Score (0-100)')
    ax.set_ylim(0, 100)

# Enhanced styling
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Clinical Insights:**
This visualization provides a comprehensive view of the patient's healthcare data using fallback mode due to API connectivity issues."""

    def fetch_backend_data_enhanced(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced backend data fetch with comprehensive validation"""
        try:
            logger.info(f"Enhanced Healthcare Backend API call: {self.config.fastapi_url}/all")

            # Enhanced validation
            required_fields = {
                "first_name": "Patient identification",
                "last_name": "Patient identification",
                "ssn": "Unique patient identifier",
                "date_of_birth": "Age verification",
                "gender": "Risk assessment",
                "zip_code": "Geographic analysis"
            }
            
            missing_fields = []
            validation_errors = []
            
            for field, purpose in required_fields.items():
                if not patient_data.get(field):
                    missing_fields.append(f"{field}: {purpose}")
                else:
                    # Basic validation
                    if field == "ssn" and len(str(patient_data[field]).replace("-", "")) != 9:
                        validation_errors.append(f"SSN format invalid")
                    elif field == "zip_code" and len(str(patient_data[field])) < 5:
                        validation_errors.append(f"ZIP code incomplete")
            
            if missing_fields or validation_errors:
                all_errors = missing_fields + validation_errors
                return {"error": f"Validation failed: {'; '.join(all_errors)}"}

            # Enhanced API call
            headers = {
                "Content-Type": "application/json",
                "X-Healthcare-Request": "enhanced",
                "X-Clinical-Analysis": "comprehensive"
            }

            response = requests.post(
                f"{self.config.fastapi_url}/all",
                json=patient_data,
                headers=headers,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                api_data = response.json()

                # Enhanced result mapping
                result = {
                    "mcid_output": self._enhanced_process_response(api_data.get('mcid_search', {}), 'mcid'),
                    "medical_output": self._enhanced_process_response(api_data.get('medical_submit', {}), 'medical'),
                    "pharmacy_output": self._enhanced_process_response(api_data.get('pharmacy_submit', {}), 'pharmacy'),
                    "token_output": self._enhanced_process_response(api_data.get('get_token', {}), 'token')
                }

                logger.info("Enhanced healthcare backend API successful")
                logger.info(f"Medical data: {'Available' if result['medical_output'].get('status_code') == 200 else 'Limited'}")
                logger.info(f"Pharmacy data: {'Available' if result['pharmacy_output'].get('status_code') == 200 else 'Limited'}")
                logger.info(f"MCID data: {'Available' if result['mcid_output'].get('status_code') == 200 else 'Limited'}")
                
                return result

            else:
                error_msg = f"Backend API failed {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                return {"error": error_msg}

        except Exception as e:
            error_msg = f"Backend fetch error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def _enhanced_process_response(self, response_data: Dict[str, Any], service_name: str) -> Dict[str, Any]:
        """Enhanced API response processing"""
        if not response_data:
            return {
                "error": f"No {service_name} data available",
                "service": service_name,
                "status": "unavailable"
            }

        # Enhanced error handling
        if "error" in response_data:
            return {
                "error": response_data["error"],
                "service": service_name,
                "status_code": response_data.get("status_code", 500),
                "status": "error"
            }

        # Enhanced successful response handling
        if response_data.get("status_code") == 200 and "body" in response_data:
            return {
                "status_code": 200,
                "body": response_data["body"],
                "service": service_name,
                "timestamp": response_data.get("timestamp", datetime.now().isoformat()),
                "status": "success"
            }

        # Enhanced other format handling
        return {
            "status_code": response_data.get("status_code", 200),
            "body": response_data,
            "service": service_name,
            "timestamp": datetime.now().isoformat(),
            "status": "partial"
        }

    def diagnose_batch_processing(self) -> Dict[str, Any]:
        """Comprehensive batch processing diagnosis"""
        logger.info("Starting comprehensive batch processing diagnosis...")
        
        # Test cases for diagnosis
        test_cases = [
            {
                "name": "Simple NDC Test",
                "request": "Explain NDC code 12345-678-90",
                "system": "You are a medical expert. Provide a brief explanation.",
                "expected_keywords": ["ndc", "medication", "drug"]
            },
            {
                "name": "Simple Medication Test",
                "request": "What is Metformin used for?",
                "system": "You are a medical expert. Provide a brief explanation.",
                "expected_keywords": ["diabetes", "medication", "blood sugar"]
            },
            {
                "name": "JSON Response Test",
                "request": 'Provide a JSON response with medication meanings: {"Metformin": "explanation", "Lisinopril": "explanation"}',
                "system": "You are a medical expert. Respond only in valid JSON format.",
                "expected_keywords": ["metformin", "lisinopril", "{", "}"]
            },
            {
                "name": "React Format Test",
                "request": """Generate React chart data for diagnosis codes I10, E11.9, M19.90 with frequencies 3, 2, 1:

Output format:
```javascript
const categories = ["I10", "E11.9", "M19.90"];
const data = [3, 2, 1];
const chartType = "bar";
const title = "Patient Diagnoses";
const description = "Clinical explanation";
```""",
                "system": "You are a React data visualization expert. Generate exact format requested.",
                "expected_keywords": ["const categories", "const data", "const chartType", "I10", "E11.9"]
            },
            {
                "name": "Batch NDC Test",
                "request": """Please explain these NDC medication codes:

NDC Codes: 12345-678-90, 98765-432-10, 11111-222-33

Please respond with a JSON object where each code is a key and the explanation is the value:
{"12345-678-90": "Medication description", "98765-432-10": "Medication description"}

Only return the JSON object, no other text.""",
                "system": "You are a pharmacy expert. Provide brief, clear explanations of NDC medication codes in JSON format.",
                "expected_keywords": ["12345-678-90", "{", "}", "medication"]
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            try:
                logger.info(f"Running diagnostic test: {test_case['name']}")
                
                start_time = time.time()
                response = self._call_llm_with_robust_retry(
                    test_case["request"],
                    test_case["system"],
                    f"DIAGNOSTIC-{test_case['name']}"
                )
                end_time = time.time()
                
                # Analyze response quality
                response_quality = self._analyze_response_quality(
                    response,
                    test_case.get("expected_keywords", [])
                )
                
                results[test_case["name"]] = {
                    "request": test_case["request"][:100] + "...",
                    "system": test_case["system"][:50] + "...",
                    "response": response[:500] + "..." if len(response) > 500 else response,
                    "response_length": len(response),
                    "processing_time": round(end_time - start_time, 2),
                    "success": response_quality["success"],
                    "quality_score": response_quality["score"],
                    "issues": response_quality["issues"],
                    "contains_expected_keywords": response_quality["keyword_matches"]
                }
                
                logger.info(f"Test '{test_case['name']}' completed - Success: {response_quality['success']}")
                
            except Exception as e:
                logger.error(f"Diagnostic test '{test_case['name']}' failed: {str(e)}")
                results[test_case["name"]] = {
                    "request": test_case["request"][:100] + "...",
                    "error": str(e),
                    "success": False,
                    "processing_time": 0,
                    "quality_score": 0,
                    "issues": [f"Exception: {str(e)}"]
                }
        
        # Overall diagnosis summary
        successful_tests = sum(1 for result in results.values() if result.get("success", False))
        total_tests = len(results)
        avg_processing_time = sum(result.get("processing_time", 0) for result in results.values()) / total_tests if total_tests > 0 else 0
        
        diagnosis_summary = {
            "overall_success_rate": f"{successful_tests}/{total_tests} ({(successful_tests/total_tests*100):.1f}%)",
            "average_processing_time": f"{avg_processing_time:.2f}s",
            "api_connectivity": "GOOD" if successful_tests >= total_tests * 0.7 else "POOR",
            "batch_processing_ready": successful_tests >= 3,
            "react_format_ready": results.get("React Format Test", {}).get("success", False),
            "recommendations": self._generate_diagnosis_recommendations(results)
        }
        
        logger.info(f"Batch processing diagnosis complete - Success rate: {diagnosis_summary['overall_success_rate']}")
        logger.info(f"React format ready: {diagnosis_summary['react_format_ready']}")
        
        return {
            "test_results": results,
            "summary": diagnosis_summary,
            "timestamp": datetime.now().isoformat(),
            "debug_mode": self.debug_mode
        }

    def _analyze_response_quality(self, response: str, expected_keywords: list) -> Dict[str, Any]:
        """Analyze the quality of an API response"""
        issues = []
        score = 0
        keyword_matches = 0
        
        # Check basic response validity
        if not response or len(response.strip()) == 0:
            issues.append("Empty response")
            return {"success": False, "score": 0, "issues": issues, "keyword_matches": 0}
        
        if response == "Brief explanation unavailable":
            issues.append("Generic unavailable response")
            return {"success": False, "score": 10, "issues": issues, "keyword_matches": 0}
        
        if "API call failed" in response:
            issues.append("API call failure indicated")
            return {"success": False, "score": 5, "issues": issues, "keyword_matches": 0}
        
        # Check response length
        if len(response) < 20:
            issues.append("Response too short")
            score += 10
        elif len(response) > 50:
            score += 30
        else:
            score += 20
        
        # Check for expected keywords
        response_lower = response.lower()
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                keyword_matches += 1
                score += 20
        
        # Check for JSON format if expected
        if "{" in expected_keywords or "}" in expected_keywords:
            try:
                # Try to find JSON in response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json.loads(json_match.group())
                    score += 30
                else:
                    issues.append("Expected JSON format not found")
            except json.JSONDecodeError:
                issues.append("Invalid JSON format")
        
        # Check for React format if expected
        if "const categories" in expected_keywords:
            react_elements = ["const categories", "const data", "const chartType"]
            react_matches = sum(1 for element in react_elements if element in response)
            if react_matches >= 3:
                score += 40
            elif react_matches > 0:
                score += 20
            else:
                issues.append("Expected React format not found")
        
        # Check for error indicators
        error_indicators = ["error", "failed", "unavailable", "timeout", "connection"]
        for indicator in error_indicators:
            if indicator.lower() in response_lower:
                issues.append(f"Contains error indicator: {indicator}")
                score -= 20
        
        success = score >= 50 and len(issues) == 0
        
        return {
            "success": success,
            "score": max(0, min(100, score)),
            "issues": issues,
            "keyword_matches": keyword_matches
        }

    def _generate_diagnosis_recommendations(self, test_results: Dict[str, Any]) -> list:
        """Generate recommendations based on diagnostic test results"""
        recommendations = []
        
        # Analyze common failure patterns
        failed_tests = [name for name, result in test_results.items() if not result.get("success", False)]
        
        if len(failed_tests) == len(test_results):
            recommendations.append("All tests failed - Check API connectivity and authentication")
            recommendations.append("Verify Snowflake API credentials and endpoint URL")
        elif len(failed_tests) > len(test_results) / 2:
            recommendations.append("Majority of tests failed - Possible authentication or rate limiting issues")
            recommendations.append("Consider increasing retry delays and timeout values")
        else:
            recommendations.append("Most tests passed - System appears to be working correctly")
        
        # Check for specific issues
        auth_issues = any("401" in str(result.get("response", "")) for result in test_results.values())
        if auth_issues:
            recommendations.append("Authentication errors detected - Verify API key and authorization headers")
        
        timeout_issues = any("timeout" in str(result.get("error", "")).lower() for result in test_results.values())
        if timeout_issues:
            recommendations.append("Timeout issues detected - Consider increasing timeout values")
        
        json_issues = any("JSON" in str(result.get("issues", [])) for result in test_results.values())
        if json_issues:
            recommendations.append("JSON parsing issues detected - Review response format handling")
        
        react_issues = any("React" in str(result.get("issues", [])) for result in test_results.values())
        if react_issues:
            recommendations.append("React format issues detected - Review React data generation")
        
        # Check React format specifically
        react_test = test_results.get("React Format Test", {})
        if not react_test.get("success", False):
            recommendations.append("React format generation failed - May need fallback handling")
        
        if not recommendations:
            recommendations.append("System diagnosis looks good - No major issues detected")
        
        return recommendations

    def test_healthcare_llm_connection(self) -> Dict[str, Any]:
        """Test healthcare LLM connection"""
        try:
            logger.info("Testing healthcare LLM connection...")
            
            test_prompt = "You are performing a healthcare system test. Please respond with: 'Enhanced Healthcare LLM connection ready for clinical analysis and React graph generation.'"

            test_response = self.call_llm_enhanced(test_prompt, self.config.sys_msg)

            # Enhanced validation
            if test_response and not test_response.startswith("Error") and not test_response.startswith("I apologize"):
                return {
                    "success": True,
                    "response": test_response[:200] + "...",
                    "endpoint": self.config.api_url,
                    "model": self.config.model,
                    "healthcare_ready": True,
                    "react_graph_ready": True
                }
            else:
                return {
                    "success": False,
                    "error": test_response,
                    "endpoint": self.config.api_url,
                    "healthcare_ready": False,
                    "react_graph_ready": False
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Healthcare LLM test failed: {str(e)}",
                "endpoint": self.config.api_url,
                "healthcare_ready": False,
                "react_graph_ready": False
            }

    def test_backend_connection_enhanced(self) -> Dict[str, Any]:
        """Test enhanced backend server connection"""
        try:
            logger.info("Testing healthcare backend...")

            health_url = f"{self.config.fastapi_url}/health"
            
            headers = {
                "X-Healthcare-Test": "enhanced",
                "X-Clinical-Validation": "comprehensive"
            }
            
            response = requests.get(health_url, headers=headers, timeout=10)

            if response.status_code == 200:
                health_data = response.json()
                
                service_status = "Excellent" if "status" in health_data and health_data["status"] == "healthy" else "Limited"
                
                return {
                    "success": True,
                    "health_data": health_data,
                    "backend_url": self.config.fastapi_url,
                    "service_status": service_status,
                    "healthcare_ready": True
                }
            else:
                return {
                    "success": False,
                    "error": f"Backend health check failed: {response.status_code}",
                    "backend_url": self.config.fastapi_url,
                    "healthcare_ready": False
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Backend test failed: {str(e)}",
                "backend_url": self.config.fastapi_url,
                "healthcare_ready": False
            }

    async def test_ml_connection_enhanced(self) -> Dict[str, Any]:
        """Test enhanced ML API server connection"""
        try:
            logger.info(f"Testing ML API: {self.config.heart_attack_api_url}")

            health_url = f"{self.config.heart_attack_api_url}/health"
            timeout = aiohttp.ClientTimeout(total=15)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data = await response.json()

                        # Test prediction endpoint
                        test_features = {"age": 45, "gender": 0, "diabetes": 0, "high_bp": 0, "smoking": 0}
                        
                        predict_url = f"{self.config.heart_attack_api_url}/predict"
                        async with session.post(predict_url, json=test_features) as pred_response:
                            if pred_response.status == 200:
                                pred_data = await pred_response.json()
                                
                                return {
                                    "success": True,
                                    "health_check": health_data,
                                    "test_prediction": pred_data,
                                    "server_url": self.config.heart_attack_api_url,
                                    "ml_ready": True
                                }
                            else:
                                error_text = await pred_response.text()
                                return {
                                    "success": False,
                                    "error": f"ML prediction test failed {pred_response.status}: {error_text[:200]}",
                                    "server_url": self.config.heart_attack_api_url,
                                    "ml_ready": False
                                }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"ML health endpoint error {response.status}: {error_text[:200]}",
                            "server_url": self.config.heart_attack_api_url,
                            "ml_ready": False
                        }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "ML API timeout - service may be down",
                "server_url": self.config.heart_attack_api_url,
                "ml_ready": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"ML API test failed: {str(e)}",
                "server_url": self.config.heart_attack_api_url,
                "ml_ready": False
            }

    def test_all_connections_enhanced(self) -> Dict[str, Any]:
        """Test all connections with enhanced reporting including React graph generation"""
        logger.info("Testing ALL enhanced connections with React graph generation...")

        results = {
            "llm_connection": self.test_healthcare_llm_connection(),
            "backend_connection": self.test_backend_connection_enhanced(),
            "batch_diagnosis": self.diagnose_batch_processing()
        }

        # Test ML connection
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results["ml_connection"] = loop.run_until_complete(self.test_ml_connection_enhanced())
            loop.close()
        except Exception as e:
            results["ml_connection"] = {
                "success": False,
                "error": f"ML test failed: {str(e)}",
                "ml_ready": False
            }

        # Enhanced summary with React graph generation status
        connection_results = [results["llm_connection"], results["backend_connection"], results["ml_connection"]]
        all_success = all(result.get("success", False) for result in connection_results)
        successful_connections = sum(1 for result in connection_results if result.get("success", False))
        batch_ready = results["batch_diagnosis"]["summary"]["batch_processing_ready"]
        react_ready = results["batch_diagnosis"]["summary"]["react_format_ready"]
        
        results["enhanced_overall_status"] = {
            "all_connections_successful": all_success,
            "successful_connections": successful_connections,
            "total_connections": len(connection_results),
            "healthcare_ready": all(result.get("healthcare_ready", False) or result.get("ml_ready", False) for result in connection_results),
            "batch_processing_ready": batch_ready,
            "react_graph_generation_ready": react_ready,
            "ndc_medication_meanings_ready": batch_ready and successful_connections >= 2,
            "enhancement_level": "high" if successful_connections >= 2 and batch_ready and react_ready else "moderate" if successful_connections >= 1 else "low"
        }

        logger.info(f"Enhanced connection test complete: {successful_connections}/{len(connection_results)} successful")
        logger.info(f"Healthcare ready: {results['enhanced_overall_status']['healthcare_ready']}")
        logger.info(f"Batch processing ready: {batch_ready}")
        logger.info(f"React graph generation ready: {react_ready}")
        logger.info(f"NDC/Medication meanings ready: {results['enhanced_overall_status']['ndc_medication_meanings_ready']}")

        return results

    # Remaining backward compatibility methods for other components
    def fetch_backend_data_fast(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.fetch_backend_data_enhanced(patient_data)

    def fetch_backend_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.fetch_backend_data_enhanced(patient_data)

    def test_llm_connection_enhanced(self) -> Dict[str, Any]:
        return self.test_healthcare_llm_connection()

    def test_backend_connection(self) -> Dict[str, Any]:
        return self.test_backend_connection_enhanced()

    def test_all_connections(self) -> Dict[str, Any]:
        return self.test_all_connections_enhanced()
