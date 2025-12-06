import os
import uuid
import requests
import re
from typing import Dict, Any, List, Optional
from config import Config
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

class LangExtractService:
    """Service for extracting financial and legal clauses using Groq API."""
    
    def __init__(self):
        self.initialized = True
        self.api_key = Config.GROQ_API_KEY
        self.groq_client = None
        
        if HAS_GROQ and self.api_key:
            try:
                self.groq_client = Groq(api_key=self.api_key)
                print("LangExtract: Groq client initialized.")
            except Exception as e:
                print(f"LangExtract: Failed to initialize Groq client: {e}")
        
        Config.create_directories()
        
        # Define clause patterns and extraction prompts
        self.clause_types = {
            "payment_clause": {
                "keywords": ["payment", "pay", "invoice", "fee", "charge", "amount due"],
                "description": "Clauses related to payment terms, amounts, and schedules"
            },
            "interest_clause": {
                "keywords": ["interest", "interest rate", "apr", "percentage"],
                "description": "Clauses specifying interest rates and calculations"
            },
            "termination_clause": {
                "keywords": ["terminate", "termination", "cancel", "cancellation", "end agreement"],
                "description": "Clauses defining termination conditions and procedures"
            },
            "confidentiality_clause": {
                "keywords": ["confidential", "non-disclosure", "proprietary", "secret"],
                "description": "Clauses protecting confidential information"
            },
            "liability_clause": {
                "keywords": ["liability", "liable", "responsible", "damages", "indemnify"],
                "description": "Clauses defining liability and responsibilities"
            },
            "governing_law_clause": {
                "keywords": ["governing law", "jurisdiction", "arbitration", "dispute"],
                "description": "Clauses specifying legal jurisdiction and dispute resolution"
            },
            "force_majeure_clause": {
                "keywords": ["force majeure", "act of god", "unforeseen circumstances"],
                "description": "Clauses addressing unforeseen circumstances"
            },
            "renewal_clause": {
                "keywords": ["renewal", "renew", "extend", "extension", "automatic renewal"],
                "description": "Clauses related to contract renewal"
            }
        }

    def _call_groq_api(self, prompt: str) -> Optional[str]:
        """Call Groq API with the given prompt."""
        if not self.groq_client:
            print("Groq client not initialized.")
            return None
            
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            return None

    def _extract_clause_with_ai(self, text: str, clause_type: str, description: str) -> List[Dict[str, Any]]:
        """Use Gemini to extract specific clause type from text."""
        prompt = f"""Analyze the following text and extract all {clause_type.replace('_', ' ')} clauses.
        
{description}

Text to analyze:
{text[:3000]}  

Instructions:
1. Identify all relevant clauses of this type.
2. If the exact type isn't found, find the closest matching clauses (e.g., general obligations, termination rights).
3. For each clause found, extract:
   - The exact text of the clause
   - Key attributes (e.g., amounts, dates, conditions)
   - Confidence level (high/medium/low)

Format your response as a list of blocks:
CLAUSE: [exact clause text]
ATTRIBUTES: [key attributes in JSON format]
CONFIDENCE: [high/medium/low]
---

If absolutely no relevant clauses are found, respond with "NONE"."""

        response = self._call_gemini_api(prompt)
        if not response or response.strip() == "NONE":
            return []
        
        # Parse the response
        clauses = []
        clause_blocks = response.split("---")
        
        for block in clause_blocks:
            if not block.strip():
                continue
                
            clause_data = {
                "class": clause_type,
                "text": "",
                "attributes": {}
            }
            
            # Extract clause text
            clause_match = re.search(r'CLAUSE:\s*(.+?)(?=ATTRIBUTES:|CONFIDENCE:|$)', block, re.DOTALL)
            if clause_match:
                clause_data["text"] = clause_match.group(1).strip()
            
            # Extract attributes
            attr_match = re.search(r'ATTRIBUTES:\s*(.+?)(?=CONFIDENCE:|$)', block, re.DOTALL)
            if attr_match:
                try:
                    import json
                    clause_data["attributes"] = json.loads(attr_match.group(1).strip())
                except:
                    clause_data["attributes"] = {"raw": attr_match.group(1).strip()}
            
            # Extract confidence
            conf_match = re.search(r'CONFIDENCE:\s*(\w+)', block)
            if conf_match:
                clause_data["attributes"]["confidence"] = conf_match.group(1).strip().lower()
            else:
                clause_data["attributes"]["confidence"] = "medium"
            
            if clause_data["text"]:
                clauses.append(clause_data)
        
        return clauses

    def _calculate_char_intervals(self, full_text: str, extraction_text: str) -> Dict[str, int]:
        """Find start and end positions of extracted text in full text."""
        try:
            # Try exact match first
            start = full_text.find(extraction_text)
            if start != -1:
                return {"start_pos": start, "end_pos": start + len(extraction_text)}
            
            # Try fuzzy match (simplified: strip whitespace)
            clean_full = "".join(full_text.split())
            clean_ext = "".join(extraction_text.split())
            start_clean = clean_full.find(clean_ext)
            
            if start_clean != -1:
                # Map back to original indices (approximation)
                # This is complex to do perfectly without a proper alignment library
                # For now, we'll return -1 or try a regex search for the first few words
                words = extraction_text.split()
                if len(words) > 3:
                    snippet = " ".join(words[:3])
                    start = full_text.find(snippet)
                    if start != -1:
                        return {"start_pos": start, "end_pos": start + len(extraction_text)} # Approx end
            
            return {"start_pos": -1, "end_pos": -1}
        except:
            return {"start_pos": -1, "end_pos": -1}

    def _generate_highlighted_html(self, text: str, extractions: List[Dict[str, Any]], document_id: str) -> str:
        """Generate HTML with highlighted clauses."""
        # Define colors for different clause types
        colors = {
            "Payment Clause": "#e0f2fe", # light blue
            "Interest Clause": "#fef9c3", # light yellow
            "Termination Clause": "#fee2e2", # light red
            "Confidentiality Clause": "#f3e8ff", # light purple
            "Liability Clause": "#ffedd5", # light orange
            "Governing Law Clause": "#dcfce7", # light green
            "Force Majeure Clause": "#f1f5f9", # light gray
            "Renewal Clause": "#cffafe", # cyan
            "General Clause": "#e2e8f0" # slate
        }

        # Sort extractions by start position to handle processing in order
        # Filter out those without valid positions
        valid_extractions = [
            e for e in extractions 
            if e.get("char_interval", {}).get("start_pos", -1) != -1
        ]
        valid_extractions.sort(key=lambda x: x["char_interval"]["start_pos"])

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clause Extraction Analysis</title>
            <style>
                body {{ font-family: 'Inter', sans-serif; line-height: 1.6; color: #333; padding: 20px; max-width: 800px; margin: 0 auto; }}
                .highlight {{ padding: 2px 4px; border-radius: 4px; cursor: help; position: relative; border-bottom: 2px solid rgba(0,0,0,0.1); }}
                .highlight:hover::after {{
                    content: attr(data-label);
                    position: absolute;
                    bottom: 100%;
                    left: 50%;
                    transform: translateX(-50%);
                    background: #333;
                    color: white;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    white-space: nowrap;
                    z-index: 10;
                    margin-bottom: 4px;
                }}
                .legend {{ margin-bottom: 20px; padding: 15px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; }}
                .legend-item {{ display: inline-flex; items-center; margin-right: 15px; margin-bottom: 8px; font-size: 12px; }}
                .color-box {{ width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; border: 1px solid rgba(0,0,0,0.1); }}
                h1 {{ font-size: 24px; margin-bottom: 20px; color: #1e293b; }}
            </style>
        </head>
        <body>
            <h1>Clause Extraction Analysis</h1>
            <div class="legend">
                <strong>Clause Types:</strong><br>
                {''.join(f'<span class="legend-item"><span class="color-box" style="background:{c}"></span>{k}</span>' for k, c in colors.items())}
            </div>
            <div class="content">
        """

        current_pos = 0
        for ext in valid_extractions:
            start = ext["char_interval"]["start_pos"]
            end = ext["char_interval"]["end_pos"]
            
            # Add non-highlighted text before this clause
            if start > current_pos:
                html_content += text[current_pos:start].replace("\n", "<br>")
            
            # Add highlighted clause
            clause_type = ext.get("extraction_class", "General Clause").replace("_", " ").title()
            # Normalize type name for color lookup
            color_key = next((k for k in colors.keys() if k.lower() in clause_type.lower()), "General Clause")
            color = colors.get(color_key, "#e2e8f0")
            
            clause_text = text[start:end].replace("\n", "<br>")
            html_content += f'<span class="highlight" style="background-color: {color}" data-label="{clause_type}">{clause_text}</span>'
            
            current_pos = end

        # Add remaining text
        if current_pos < len(text):
            html_content += text[current_pos:].replace("\n", "<br>")

        html_content += """
            </div>
        </body>
        </html>
        """

        # Save to file
        filename = f"clause_highlight_{document_id}.html"
        output_dir = os.path.join(Config.BASE_DIR, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return f"/outputs/{filename}"

    def extract_clauses(self, text: str, models: Optional[List[str]] = None,
                    api_key: Optional[str] = None, document_id: Optional[str] = None,
                    user_instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract financial and legal clauses from text using Groq API.
        Returns JSON in the specific format requested.
        """
        if api_key:
            self.api_key = api_key
            
        if not self.api_key:
            return {
                "success": False,
                "error": "Groq API key not configured",
                "results": {},
                "metadata": {}
            }
        
        # Generate document ID if not provided
        if not document_id:
            document_id = f"doc_{uuid.uuid4().hex[:8]}"

        # Construct prompt for structured JSON output
        system_instruction = f"""Analyze the following financial/legal document and extract key clauses.
        
        Output the result strictly as a valid JSON object with the following structure:
        {{
            "extractions": [
                {{
                    "extraction_class": "class_name (e.g., payment_clause, interest_clause)",
                    "extraction_text": "exact text from document",
                    "alignment_status": "match_exact" or "match_fuzzy",
                    "extraction_index": 1,
                    "group_index": 0,
                    "description": null,
                    "attributes": {{
                        "key": "value"
                    }}
                }}
            ],
            "text": "original text (truncated if too long)",
            "document_id": "{document_id}"
        }}
        """

        if user_instruction and user_instruction.strip():
            # Use user provided instructions
            logging_info = f"Using custom user instructions: {user_instruction[:50]}..."
            specific_instruction = f"""
            User Instructions & Custom Clauses:
            {user_instruction}
            
            Strictly follow the JSON structure defined above. Map the user's requested extractions to the 'extraction_class' field.
            """
        else:
            # Default instructions
            logging_info = "Using default clause types."
            specific_instruction = """
            Clause Types to look for:
            - Payment Clause (payment_due, late_fee, condition)
            - Interest Clause (interest_rate, calculation_period, condition)
            - Termination Clause (notice_period, triggering_party, condition)
            - Confidentiality Clause
            - Liability Clause
            - Governing Law Clause
            """

        prompt = f"""{system_instruction}

        {specific_instruction}

        Text to analyze:
        {text[:10000]}
        """
        
        print(logging_info)

        try:
            response_text = self._call_groq_api(prompt)
            if not response_text:
                raise Exception("No response from Groq API")

            # Clean markdown code blocks if present (Groq might still wrap JSON)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            import json
            data = json.loads(response_text)
            
            # Post-process to add char intervals and ensure structure
            if "extractions" in data:
                for i, extraction in enumerate(data["extractions"]):
                    # Calculate char intervals
                    if "extraction_text" in extraction:
                        interval = self._calculate_char_intervals(text, extraction["extraction_text"])
                        extraction["char_interval"] = interval
                        
                        # Update alignment status based on interval finding
                        if interval["start_pos"] != -1:
                            extraction["alignment_status"] = "match_exact"
                        else:
                            extraction["alignment_status"] = "match_fuzzy"
                    
                    # Ensure indices
                    extraction["extraction_index"] = i + 1
                    if "group_index" not in extraction:
                        extraction["group_index"] = 0

            # Ensure text and document_id are present
            data["text"] = text
            data["document_id"] = document_id
            
            # Generate highlighted HTML
            highlighted_html_url = self._generate_highlighted_html(text, data.get("extractions", []), document_id)
            
            return {
                "success": True,
                "results": {
                    "llama-3.1-8b-instant": {
                        "success": True,
                        "extractions": data.get("extractions", []),
                        "json_output": data
                    }
                },
                "clauses": data.get("extractions", []), # For backward compatibility with frontend
                "text": text,
                "highlighted_html_url": highlighted_html_url,
                "metadata": {
                    "text_length": len(text),
                    "models_used": ["llama-3.1-8b-instant"],
                    "total_clauses": len(data.get("extractions", []))
                }
            }

        except Exception as e:
            print(f"Error in extraction: {e}")
            # Fallback to empty structure
            return {
                "success": False,
                "error": str(e),
                "results": {},
                "clauses": []
            }

    def is_available(self) -> bool:
        return bool(self.api_key)
