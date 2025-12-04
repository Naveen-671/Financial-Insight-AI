# routers/export_router.py
"""Export router for generating downloads and reports."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import json
import io
import csv
from datetime import datetime

router = APIRouter(prefix="/api/v1/export", tags=["Export"])

class ExportRequest(BaseModel):
    """Request model for export."""
    data: Dict[str, Any]
    format: str = "json"  # json, csv, txt

@router.post("/results")
async def export_results(request: ExportRequest):
    """
    Export analysis results in various formats.
    
    Supported formats:
    - json: Complete results as JSON
    - csv: Entity/clause data as CSV
    - txt: Plain text summary
    """
    try:
        data = request.data
        export_format = request.format.lower()
        
        if export_format == "json":
            # Export as JSON
            json_str = json.dumps(data, indent=2)
            
            return StreamingResponse(
                io.BytesIO(json_str.encode()),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                }
            )
        
        elif export_format == "csv":
            # Export entities/clauses as CSV
            output = io.StringIO()
            
            # Check if we have NER data
            if "ner" in data and "entities" in data["ner"]:
                writer = csv.writer(output)
                writer.writerow(["Type", "Text", "Start", "End"])
                
                for entity in data["ner"]["entities"]:
                    writer.writerow([
                        entity.get("label", ""),
                        entity.get("text", ""),
                        entity.get("start", ""),
                        entity.get("end", "")
                    ])
            
            # Add clauses if available
            if "langextract" in data and "results" in data["langextract"]:
                output.write("\n\nExtracted Clauses\n")
                writer = csv.writer(output)
                writer.writerow(["Class", "Text", "Confidence"])
                
                for model_result in data["langextract"]["results"].values():
                    if "extractions" in model_result:
                        for clause in model_result["extractions"]:
                            writer.writerow([
                                clause.get("class", ""),
                                clause.get("text", ""),
                                clause.get("attributes", {}).get("confidence", "")
                            ])
            
            csv_content = output.getvalue()
            
            return StreamingResponse(
                io.BytesIO(csv_content.encode()),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                }
            )
        
        elif export_format == "txt":
            # Export as formatted text
            text_output = []
            text_output.append("=" * 60)
            text_output.append("FINANCIAL INSIGHT ANALYSIS REPORT")
            text_output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            text_output.append("=" * 60)
            text_output.append("")
            
            # NER Section
            if "ner" in data and "entities" in data["ner"]:
                text_output.append("NAMED ENTITY RECOGNITION")
                text_output.append("-" * 60)
                text_output.append(f"Total Entities Found: {len(data['ner']['entities'])}")
                text_output.append("")
                
                # Group by type
                entity_types = {}
                for entity in data["ner"]["entities"]:
                    etype = entity.get("label", "UNKNOWN")
                    if etype not in entity_types:
                        entity_types[etype] = []
                    entity_types[etype].append(entity.get("text", ""))
                
                for etype, texts in entity_types.items():
                    text_output.append(f"{etype}: {len(texts)} instances")
                    for text in texts[:5]:  # Show first 5
                        text_output.append(f"  - {text}")
                    if len(texts) > 5:
                        text_output.append(f"  ... and {len(texts) - 5} more")
                    text_output.append("")
            
            # Sentiment Section
            if "finbert" in data and "statistics" in data["finbert"]:
                stats = data["finbert"]["statistics"]
                text_output.append("SENTIMENT ANALYSIS")
                text_output.append("-" * 60)
                text_output.append(f"Overall Sentiment: {stats.get('overall_sentiment', 'N/A').upper()}")
                text_output.append("")
                
                if "sentiment_distribution" in stats:
                    text_output.append("Sentiment Distribution:")
                    for sentiment, count in stats["sentiment_distribution"].items():
                        text_output.append(f"  {sentiment.capitalize()}: {count}")
                text_output.append("")
            
            # Clause Extraction Section
            if "langextract" in data and "results" in data["langextract"]:
                text_output.append("CLAUSE EXTRACTION")
                text_output.append("-" * 60)
                
                total_clauses = 0
                for model_result in data["langextract"]["results"].values():
                    if "extractions" in model_result:
                        total_clauses += len(model_result["extractions"])
                        
                        for clause in model_result["extractions"]:
                            text_output.append(f"\n[{clause.get('class', 'Unknown').replace('_', ' ').upper()}]")
                            text_output.append(clause.get("text", ""))
                            text_output.append(f"Confidence: {clause.get('attributes', {}).get('confidence', 'N/A')}")
                            text_output.append("")
                
                text_output.append(f"Total Clauses Extracted: {total_clauses}")
                text_output.append("")
            
            text_output.append("=" * 60)
            text_output.append("END OF REPORT")
            text_output.append("=" * 60)
            
            text_content = "\n".join(text_output)
            
            return StreamingResponse(
                io.BytesIO(text_content.encode()),
                media_type="text/plain",
                headers={
                    "Content-Disposition": f"attachment; filename=analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                }
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {export_format}. Use 'json', 'csv', or 'txt'"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@router.post("/generate-report")
async def generate_comprehensive_report(data: Dict[str, Any]):
    """
    Generate a comprehensive HTML report with all analysis results.
    """
    try:
        # Create HTML report
        html_content = []
        html_content.append("<!DOCTYPE html>")
        html_content.append("<html lang='en'>")
        html_content.append("<head>")
        html_content.append("<meta charset='UTF-8'>")
        html_content.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html_content.append("<title>Financial Insight Analysis Report</title>")
        html_content.append("<style>")
        html_content.append("""
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }
            .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
            .stat-card { background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }
            .stat-value { font-size: 2em; font-weight: bold; color: #3498db; }
            .stat-label { color: #7f8c8d; margin-top: 5px; }
            .entity-tag { display: inline-block; padding: 5px 12px; margin: 5px; border-radius: 20px; background: #3498db; color: white; font-size: 0.9em; }
            .clause { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px; }
            .clause-type { font-weight: bold; color: #856404; margin-bottom: 10px; text-transform: uppercase; }
            .sentiment-positive { color: #27ae60; font-weight: bold; }
            .sentiment-negative { color: #e74c3c; font-weight: bold; }
            .sentiment-neutral { color: #95a5a6; font-weight: bold; }
            .metadata { background: #f8f9fa; padding: 15px; border-radius: 4px; margin-top: 30px; font-size: 0.9em; color: #6c757d; }
        """)
        html_content.append("</style>")
        html_content.append("</head>")
        html_content.append("<body>")
        html_content.append("<div class='container'>")
        
        # Header
        html_content.append(f"<h1>Financial Insight Analysis Report</h1>")
        html_content.append(f"<p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>")
        
        # Summary Statistics
        html_content.append("<div class='stat-grid'>")
        
        if "ner" in data and "entities" in data["ner"]:
            html_content.append("<div class='stat-card'>")
            html_content.append(f"<div class='stat-value'>{len(data['ner']['entities'])}</div>")
            html_content.append("<div class='stat-label'>Entities Found</div>")
            html_content.append("</div>")
        
        if "finbert" in data and "statistics" in data["finbert"]:
            sentiment = data["finbert"]["statistics"].get("overall_sentiment", "neutral")
            html_content.append("<div class='stat-card'>")
            html_content.append(f"<div class='stat-value sentiment-{sentiment}'>{sentiment.upper()}</div>")
            html_content.append("<div class='stat-label'>Overall Sentiment</div>")
            html_content.append("</div>")
        
        if "langextract" in data:
            clause_count = 0
            for model_result in data["langextract"].get("results", {}).values():
                clause_count += len(model_result.get("extractions", []))
            html_content.append("<div class='stat-card'>")
            html_content.append(f"<div class='stat-value'>{clause_count}</div>")
            html_content.append("<div class='stat-label'>Clauses Extracted</div>")
            html_content.append("</div>")
        
        html_content.append("</div>")
        
        # NER Results
        if "ner" in data and "entities" in data["ner"]:
            html_content.append("<h2>Named Entity Recognition</h2>")
            entity_types = {}
            for entity in data["ner"]["entities"]:
                etype = entity.get("label", "UNKNOWN")
                if etype not in entity_types:
                    entity_types[etype] = []
                entity_types[etype].append(entity.get("text", ""))
            
            for etype, texts in entity_types.items():
                html_content.append(f"<h3>{etype.replace('_', ' ').title()} ({len(texts)})</h3>")
                html_content.append("<div>")
                for text in set(texts):  # Unique values
                    html_content.append(f"<span class='entity-tag'>{text}</span>")
                html_content.append("</div>")
        
        # Sentiment Results
        if "finbert" in data and "sentence_results" in data["finbert"]:
            html_content.append("<h2>Sentiment Analysis</h2>")
            for sentence in data["finbert"]["sentence_results"][:10]:  # First 10
                sentiment_class = f"sentiment-{sentence.get('label', 'neutral').lower()}"
                html_content.append(f"<p><span class='{sentiment_class}'>[{sentence.get('label', 'NEUTRAL').upper()}]</span> {sentence.get('text', '')}</p>")
        
        # Clause Extraction
        if "langextract" in data and "results" in data["langextract"]:
            html_content.append("<h2>Extracted Clauses</h2>")
            for model_result in data["langextract"]["results"].values():
                for clause in model_result.get("extractions", []):
                    html_content.append("<div class='clause'>")
                    html_content.append(f"<div class='clause-type'>{clause.get('class', 'Unknown').replace('_', ' ')}</div>")
                    html_content.append(f"<p>{clause.get('text', '')}</p>")
                    html_content.append(f"<small>Confidence: {clause.get('attributes', {}).get('confidence', 'N/A')}</small>")
                    html_content.append("</div>")
        
        # Footer
        html_content.append("<div class='metadata'>")
        html_content.append("<p><strong>Financial Insight</strong> - AI-Powered Document Analysis</p>")
        html_content.append(f"<p>Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}</p>")
        html_content.append("</div>")
        
        html_content.append("</div>")
        html_content.append("</body>")
        html_content.append("</html>")
        
        html_str = "\n".join(html_content)
        
        return StreamingResponse(
            io.BytesIO(html_str.encode()),
            media_type="text/html",
            headers={
                "Content-Disposition": f"attachment; filename=financial_insight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")

@router.get("/formats")
async def get_supported_formats():
    """Get list of supported export formats."""
    return {
        "formats": [
            {
                "id": "json",
                "name": "JSON",
                "description": "Complete analysis results in JSON format",
                "extension": ".json",
                "mime_type": "application/json"
            },
            {
                "id": "csv",
                "name": "CSV",
                "description": "Entities and clauses in CSV format",
                "extension": ".csv",
                "mime_type": "text/csv"
            },
            {
                "id": "txt",
                "name": "Text Report",
                "description": "Formatted text summary",
                "extension": ".txt",
                "mime_type": "text/plain"
            },
            {
                "id": "html",
                "name": "HTML Report",
                "description": "Comprehensive HTML report with styling",
                "extension": ".html",
                "mime_type": "text/html"
            }
        ]
    }
