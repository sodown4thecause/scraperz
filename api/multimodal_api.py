from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import logging

from database import get_db
from models import ScrapingJob, ImageExtraction, TableExtraction, StructuredDataExtraction
from multimodal_extractor import MultiModalExtractor, MultiModalExtractionResult
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/multimodal", tags=["multimodal"])

# Pydantic models for API responses
class ImageExtractionResponse(BaseModel):
    id: int
    image_url: str
    alt_text: str
    caption: str
    image_format: str
    width: int
    height: int
    file_size: int
    context: str
    extracted_text: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class TableExtractionResponse(BaseModel):
    id: int
    table_data: Dict[str, Any]
    headers: List[str]
    row_count: int
    column_count: int
    table_caption: str
    table_context: str
    data_types: Dict[str, Any]
    created_at: datetime
    
    class Config:
        from_attributes = True

class StructuredDataResponse(BaseModel):
    id: int
    schema_type: str
    structured_data: Dict[str, Any]
    data_category: str
    confidence_score: float
    validation_status: str
    validation_errors: List[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class MultiModalExtractionRequest(BaseModel):
    url: str
    html_content: str
    job_id: Optional[int] = None

class MultiModalExtractionSummary(BaseModel):
    job_id: int
    url: str
    images_count: int
    tables_count: int
    structured_data_count: int
    processing_time: float
    total_confidence: float
    extraction_metadata: Dict[str, Any]
    created_at: datetime

# Initialize the extractor
extractor = MultiModalExtractor()

@router.post("/extract", response_model=MultiModalExtractionSummary)
async def extract_multimodal_data(
    request: MultiModalExtractionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Extract multi-modal data from HTML content
    """
    try:
        # Get or create scraping job
        if request.job_id:
            job = db.query(ScrapingJob).filter(ScrapingJob.id == request.job_id).first()
            if not job:
                raise HTTPException(status_code=404, detail="Scraping job not found")
        else:
            # Create a new job for this extraction
            job = ScrapingJob(
                url=request.url,
                status="processing",
                created_at=datetime.now()
            )
            db.add(job)
            db.commit()
            db.refresh(job)
        
        # Perform extraction
        result = await extractor.extract_multimodal_data(request.url, request.html_content)
        
        # Store results in background
        background_tasks.add_task(
            store_multimodal_results,
            job.id,
            result,
            db
        )
        
        return MultiModalExtractionSummary(
            job_id=job.id,
            url=result.url,
            images_count=len(result.images),
            tables_count=len(result.tables),
            structured_data_count=len(result.structured_data),
            processing_time=result.processing_time,
            total_confidence=result.total_confidence,
            extraction_metadata=result.extraction_metadata,
            created_at=job.created_at
        )
        
    except Exception as e:
        logger.error(f"Error in multi-modal extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@router.get("/jobs/{job_id}/images", response_model=List[ImageExtractionResponse])
def get_job_images(job_id: int, db: Session = Depends(get_db)):
    """
    Get all extracted images for a job
    """
    job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    images = db.query(ImageExtraction).filter(ImageExtraction.scraping_job_id == job_id).all()
    return images

@router.get("/jobs/{job_id}/tables", response_model=List[TableExtractionResponse])
def get_job_tables(job_id: int, db: Session = Depends(get_db)):
    """
    Get all extracted tables for a job
    """
    job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    tables = db.query(TableExtraction).filter(TableExtraction.scraping_job_id == job_id).all()
    return tables

@router.get("/jobs/{job_id}/structured-data", response_model=List[StructuredDataResponse])
def get_job_structured_data(job_id: int, db: Session = Depends(get_db)):
    """
    Get all extracted structured data for a job
    """
    job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    structured_data = db.query(StructuredDataExtraction).filter(
        StructuredDataExtraction.scraping_job_id == job_id
    ).all()
    return structured_data

@router.get("/jobs/{job_id}/summary", response_model=MultiModalExtractionSummary)
def get_extraction_summary(job_id: int, db: Session = Depends(get_db)):
    """
    Get summary of multi-modal extraction for a job
    """
    job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Count extracted data
    images_count = db.query(ImageExtraction).filter(ImageExtraction.scraping_job_id == job_id).count()
    tables_count = db.query(TableExtraction).filter(TableExtraction.scraping_job_id == job_id).count()
    structured_data_count = db.query(StructuredDataExtraction).filter(
        StructuredDataExtraction.scraping_job_id == job_id
    ).count()
    
    # Calculate average confidence
    structured_data_items = db.query(StructuredDataExtraction).filter(
        StructuredDataExtraction.scraping_job_id == job_id
    ).all()
    
    avg_confidence = 0.0
    if structured_data_items:
        avg_confidence = sum(item.confidence_score for item in structured_data_items) / len(structured_data_items)
    
    return MultiModalExtractionSummary(
        job_id=job.id,
        url=job.url,
        images_count=images_count,
        tables_count=tables_count,
        structured_data_count=structured_data_count,
        processing_time=0.0,  # Not stored in job model
        total_confidence=avg_confidence,
        extraction_metadata={},  # Not stored in job model
        created_at=job.created_at
    )

@router.get("/images/{image_id}/data")
def get_image_data(image_id: int, db: Session = Depends(get_db)):
    """
    Get raw image data
    """
    image = db.query(ImageExtraction).filter(ImageExtraction.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not image.image_data:
        raise HTTPException(status_code=404, detail="Image data not available")
    
    from fastapi.responses import Response
    
    # Determine content type based on format
    content_type_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'bmp': 'image/bmp',
        'webp': 'image/webp'
    }
    
    content_type = content_type_map.get(image.image_format.lower(), 'application/octet-stream')
    
    return Response(
        content=image.image_data,
        media_type=content_type,
        headers={"Content-Disposition": f"inline; filename=image_{image_id}.{image.image_format}"}
    )

@router.get("/tables/{table_id}/export")
def export_table_data(table_id: int, format: str = "json", db: Session = Depends(get_db)):
    """
    Export table data in various formats (json, csv)
    """
    table = db.query(TableExtraction).filter(TableExtraction.id == table_id).first()
    if not table:
        raise HTTPException(status_code=404, detail="Table not found")
    
    if format.lower() == "csv":
        import csv
        import io
        from fastapi.responses import StreamingResponse
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        if table.headers:
            writer.writerow(table.headers)
        
        # Write data rows
        if table.table_data and 'rows' in table.table_data:
            for row in table.table_data['rows']:
                writer.writerow(row)
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=table_{table_id}.csv"}
        )
    
    else:  # Default to JSON
        from fastapi.responses import JSONResponse
        return JSONResponse(content=table.table_data)

@router.get("/analytics/summary")
def get_multimodal_analytics(db: Session = Depends(get_db)):
    """
    Get analytics summary for multi-modal extractions
    """
    # Count totals
    total_images = db.query(ImageExtraction).count()
    total_tables = db.query(TableExtraction).count()
    total_structured_data = db.query(StructuredDataExtraction).count()
    
    # Get format distribution for images
    image_formats = db.query(ImageExtraction.image_format).all()
    format_counts = {}
    for (format_name,) in image_formats:
        format_counts[format_name] = format_counts.get(format_name, 0) + 1
    
    # Get schema type distribution for structured data
    schema_types = db.query(StructuredDataExtraction.schema_type).all()
    schema_counts = {}
    for (schema_type,) in schema_types:
        schema_counts[schema_type] = schema_counts.get(schema_type, 0) + 1
    
    # Calculate average confidence scores
    confidence_scores = db.query(StructuredDataExtraction.confidence_score).all()
    avg_confidence = 0.0
    if confidence_scores:
        avg_confidence = sum(score[0] for score in confidence_scores if score[0]) / len(confidence_scores)
    
    return {
        "totals": {
            "images": total_images,
            "tables": total_tables,
            "structured_data": total_structured_data
        },
        "image_formats": format_counts,
        "schema_types": schema_counts,
        "average_confidence": avg_confidence,
        "generated_at": datetime.now().isoformat()
    }

async def store_multimodal_results(
    job_id: int,
    result: MultiModalExtractionResult,
    db: Session
):
    """
    Store multi-modal extraction results in the database
    """
    try:
        # Store images
        for image_data in result.images:
            # Download and store image data
            image_bytes = None
            try:
                import requests
                response = requests.get(image_data.url, timeout=10)
                if response.status_code == 200:
                    image_bytes = response.content
            except Exception as e:
                logger.warning(f"Failed to download image {image_data.url}: {str(e)}")
            
            db_image = ImageExtraction(
                scraping_job_id=job_id,
                image_url=image_data.url,
                alt_text=image_data.alt_text,
                caption=image_data.caption,
                image_data=image_bytes,
                image_format=image_data.format,
                width=image_data.dimensions[0],
                height=image_data.dimensions[1],
                file_size=image_data.file_size,
                context=image_data.context,
                extracted_text=image_data.text_content
            )
            db.add(db_image)
        
        # Store tables
        for table_data in result.tables:
            db_table = TableExtraction(
                scraping_job_id=job_id,
                table_data={
                    'headers': table_data.headers,
                    'rows': table_data.rows,
                    'summary': table_data.summary,
                    'relationships': table_data.relationships
                },
                headers=table_data.headers,
                row_count=len(table_data.rows),
                column_count=len(table_data.headers),
                table_caption=table_data.caption,
                table_context=table_data.context,
                data_types=table_data.data_types
            )
            db.add(db_table)
        
        # Store structured data
        for structured_data in result.structured_data:
            db_structured = StructuredDataExtraction(
                scraping_job_id=job_id,
                schema_type=structured_data.schema_type,
                structured_data=structured_data.data,
                data_category=structured_data.data.get('@type', 'unknown'),
                confidence_score=structured_data.confidence_score,
                validation_status='valid' if not structured_data.validation_errors else 'invalid',
                validation_errors=structured_data.validation_errors
            )
            db.add(db_structured)
        
        db.commit()
        logger.info(f"Stored multi-modal results for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error storing multi-modal results for job {job_id}: {str(e)}")
        db.rollback()
        raise