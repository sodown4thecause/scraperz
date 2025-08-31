import os
import sys
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Add the project directory to the sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models
import database

# Initialize logger
logger = logging.getLogger(__name__)
from intelligent_analyzer import IntelligentAnalyzer, ContentAnalysis
from adaptive_strategy import AdaptiveStrategyEngine, StrategyResult
from incremental_scraper import IncrementalScrapingEngine
from monitoring import router as monitoring_router, monitoring_background_task
from multimodal_api import router as multimodal_router
from incremental_api import router as incremental_router

# Create database tables
models.Base.metadata.create_all(bind=database.engine)

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables.")

# --- Initialize Intelligent Systems ---
anti_detection_engine = AdvancedAntiDetectionEngine()
intelligent_analyzer = IntelligentAnalyzer()
adaptive_strategy_engine = AdaptiveStrategyEngine(anti_detection_engine=anti_detection_engine)
incremental_scraper = IncrementalScrapingEngine()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Scraping Engine API",
    description="An API powered by ScrapeGraphAI and Gemini to perform intelligent web scraping."
)

# Include monitoring router
app.include_router(monitoring_router)
app.include_router(multimodal_router)
app.include_router(incremental_router)

# --- CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ScrapeRequest(BaseModel):
    url: str
    prompt: str
    user_id: str
    force_refresh: bool = False
    max_age_hours: int = 24

class CrawlRequest(BaseModel):
    url: str
    prompt: str
    user_id: str

class DynamicScrapeRequest(BaseModel):
    url: str
    prompt: str
    user_id: str

# --- Graph Configuration ---
graph_config = {
    "llm": {
        "api_key": gemini_api_key,
        "model": "gemini-pro",
    },
    "verbose": False,
    "headless": True,
}

# --- Helper function to run Scrapy in a separate process ---
def run_spider(start_url, prompt, llm_config, queue):
    try:
        def crawl(q):
            try:
                results = []
                process = CrawlerProcess(get_project_settings())
                spider_kwargs = {
                    'start_url': start_url,
                    'prompt': prompt,
                    'llm_config': llm_config,
                    'results_list': results
                }
                process.crawl(IntelligentSpider, **spider_kwargs)
                process.start()
                q.put(results)
            except Exception as e:
                q.put(e)
        
        p = multiprocessing.Process(target=crawl, args=(queue,))
        p.start()
        p.join()

    except Exception as e:
        queue.put(e)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Scraping Engine API is running."}

@app.post("/scrape")
async def scrape_url(request: ScrapeRequest, db: Session = Depends(database.get_db)):
    if not all([request.url, request.prompt, request.user_id]):
        raise HTTPException(status_code=400, detail="URL, prompt, and user_id are required.")
    
    from monitoring import notify_job_update
    import time
    
    # Create job record
    db_job = models.ScrapingJob(
        user_id=request.user_id, 
        url=request.url, 
        prompt=request.prompt, 
        job_type="scrape",
        status="running"
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    # Notify monitoring system
    await notify_job_update(str(db_job.id), "running", 10)
    
    try:
        start_time = time.time()
        
        # Check if incremental scraping should be performed
        should_scrape, change_result = await incremental_scraper.should_scrape(
            request.url, 
            force_refresh=request.force_refresh,
            max_age_hours=request.max_age_hours
        )
        
        if not should_scrape and change_result:
            # Return cached result with change detection info
            db_job.status = "completed"
            db_job.processing_time = time.time() - start_time
            db_job.quality_score = change_result.confidence
            db_job.strategy = "incremental_cache"
            db.commit()
            
            # Get cached result
            cached_job = db.query(models.ScrapingJob).filter(
                models.ScrapingJob.id == int(change_result.previous_version_id)
            ).first()
            
            if cached_job and cached_job.results:
                cached_data = cached_job.results[0].data
                
                # Add change detection metadata
                if isinstance(cached_data, dict):
                    cached_data['change_detection'] = {
                        'from_cache': True,
                        'similarity_score': change_result.similarity_score,
                        'change_summary': change_result.change_summary,
                        'last_scraped': str(cached_job.created_at)
                    }
                
                await notify_job_update(str(db_job.id), "completed", 100, change_result.confidence)
                return {
                    "data": cached_data, 
                    "job_id": db_job.id, 
                    "quality_score": change_result.confidence,
                    "from_cache": True,
                    "change_detection": change_result.__dict__
                }
        
        # Proceed with full scraping
        await notify_job_update(str(db_job.id), "running", 30)
        
        # Execute adaptive scraping with multi-modal extraction
        strategy_result = await adaptive_strategy_engine.execute_adaptive_scraping(
            request.url, request.prompt, request.user_id
        )
        
        # Update progress
        await notify_job_update(str(db_job.id), "running", 50)
        
        result = strategy_result.data
        processing_time = time.time() - start_time
        
        # Update job with completion details
        db_job.status = "completed"
        db_job.processing_time = processing_time
        db_job.quality_score = strategy_result.confidence_score
        db_job.strategy = strategy_result.strategy
        db.commit()

        # Save result
        db_result = models.ScrapingResult(job_id=db_job.id, data=result)
        db.add(db_result)
        db.commit()
        
        # Save multi-modal extraction results if available
        if isinstance(result, dict) and 'multimodal' in result:
            multimodal_data = result['multimodal']
            
            # Save image extractions
            for img_data in multimodal_data.get('images', []):
                db_image = models.ImageExtraction(
                    job_id=db_job.id,
                    url=img_data.get('url', ''),
                    alt_text=img_data.get('alt_text', ''),
                    width=img_data.get('width'),
                    height=img_data.get('height'),
                    file_size=img_data.get('file_size'),
                    format=img_data.get('format', ''),
                    description=img_data.get('description', '')
                )
                db.add(db_image)
            
            # Save table extractions
            for table_data in multimodal_data.get('tables', []):
                db_table = models.TableExtraction(
                    job_id=db_job.id,
                    table_data=table_data.get('data', {}),
                    headers=table_data.get('headers', []),
                    row_count=table_data.get('row_count', 0),
                    column_count=table_data.get('column_count', 0),
                    caption=table_data.get('caption', ''),
                    summary=table_data.get('summary', '')
                )
                db.add(db_table)
            
            # Save structured data extractions
            for struct_data in multimodal_data.get('structured_data', []):
                db_struct = models.StructuredDataExtraction(
                    job_id=db_job.id,
                    schema_type=struct_data.get('schema_type', ''),
                    data=struct_data.get('data', {}),
                    confidence_score=struct_data.get('confidence_score', 0.0),
                    validation_status=struct_data.get('validation_status', False),
                    extracted_fields=struct_data.get('extracted_fields', [])
                )
                db.add(db_struct)
            
            db.commit()
         
         # Update incremental scraping cache
         try:
             content_for_cache = ""
             if isinstance(result, dict):
                 content_for_cache = result.get('content', str(result))
             else:
                 content_for_cache = str(result)
             
             await incremental_scraper.update_cache(
                 request.url,
                 content_for_cache,
                 str(db_job.id),
                 metadata={
                     'strategy': strategy_result.strategy,
                     'confidence_score': strategy_result.confidence_score,
                     'processing_time': processing_time
                 }
             )
         except Exception as cache_error:
             logger.warning(f"Failed to update incremental cache: {str(cache_error)}")
         
         # Notify completion
         await notify_job_update(str(db_job.id), "completed", 100, strategy_result.confidence_score)

         return {"data": result, "job_id": db_job.id, "quality_score": strategy_result.confidence_score}
    except Exception as e:
        # Update job status to failed
        db_job.status = "failed"
        db_job.error_message = str(e)
        db.commit()
        
        await notify_job_update(str(db_job.id), "failed", 0)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawl")
async def crawl_url(request: CrawlRequest, db: Session = Depends(database.get_db)):
    if not all([request.url, request.prompt, request.user_id]):
        raise HTTPException(status_code=400, detail="URL, prompt, and user_id are required.")

    q = multiprocessing.Queue()
    run_spider(request.url, request.prompt, graph_config, q)
    result = q.get()

    if isinstance(result, Exception):
        raise HTTPException(status_code=500, detail=f"An error occurred during crawling: {result}")
    
    # Save to database
    db_job = models.ScrapingJob(user_id=request.user_id, url=request.url, prompt=request.prompt, job_type="crawl")
    db.add(db_job)
    db.commit()
    db.refresh(db_job)

    db_result = models.ScrapingResult(job_id=db_job.id, data=result)
    db.add(db_result)
    db.commit()

    return {"data": result}

@app.post("/dynamic_scrape")
async def dynamic_scrape_url(request: DynamicScrapeRequest, db: Session = Depends(database.get_db)):
    if not all([request.url, request.prompt, request.user_id]):
        raise HTTPException(status_code=400, detail="URL, prompt, and user_id are required.")
    try:
        # Use Crawl4AI to get the rendered HTML
        crawler = Crawl4AI(url=request.url)
        rendered_html = await crawler.fetch_html()

        if not rendered_html:
            raise HTTPException(status_code=500, detail="Crawl4AI failed to fetch or render the page.")

        smart_scraper_graph = SmartScraperGraph(
            prompt=request.prompt,
            source=rendered_html,
            config=graph_config
        )
        result = smart_scraper_graph.run()

        # Save to database
        db_job = models.ScrapingJob(user_id=request.user_id, url=request.url, prompt=request.prompt, job_type="dynamic_scrape")
        db.add(db_job)
        db.commit()
        db.refresh(db_job)

        db_result = models.ScrapingResult(job_id=db_job.id, data=result)
        db.add(db_result)
        db.commit()

        return {"data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}")
def get_history(user_id: str, db: Session = Depends(database.get_db)):
    jobs = db.query(models.ScrapingJob).filter(models.ScrapingJob.user_id == user_id).order_by(models.ScrapingJob.created_at.desc()).all()
    return jobs

@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: int, db: Session = Depends(database.get_db)):
    job = db.query(models.ScrapingJob).filter(models.ScrapingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # Ensure results are loaded
    results = job.results
    return {"job_id": job.id, "status": job.status, "quality_score": job.quality_score, "results": results if job.status == 'completed' else []}

@app.post("/process_data")
async def process_data(request: ProcessDataRequest):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = ""

        if request.action == AIAction.summarize:
            prompt = f"Summarize the following text concisely:\n\n---\n{request.content}\n---"
        
        elif request.action == AIAction.ask:
            question = request.params.get('query')
            if not question:
                raise HTTPException(status_code=400, detail="A 'query' parameter is required for the 'ask' action.")
            prompt = f"Based on the text below, answer the following question.\n\nText:\n---\n{request.content}\n---\n\nQuestion: {question}"

        elif request.action == AIAction.extract_entities:
            prompt = f"""Analyze the following text and extract key entities. 
Return the result as a single, valid JSON object. Do not include any text before or after the JSON object. 
The JSON object should have keys for each entity type (e.g., 'people', 'organizations', 'locations', 'dates', 'products'). 
Each key should have a list of the string values of the entities found. If no entities of a certain type are found, return an empty list [] for that key."""

Text:
---
{request.content}
---

        response = await model.generate_content_async(prompt)
        
        # Clean up the response if it's JSON
        if request.action == AIAction.extract_entities:
            cleaned_response = response.text.strip().lstrip('```json').rstrip('```')
            return json.loads(cleaned_response)
            
        return {"result": response.text}

    except Exception as e:
        logger.error(f"Error processing data with AI: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during AI processing: {e}")


# --- Startup Events ---
@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(monitoring_background_task())
    await anti_detection_engine.start_background_tasks()
