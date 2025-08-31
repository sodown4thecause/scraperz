from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float, Text, Boolean, ARRAY, Index, LargeBinary
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid
from database import Base

class ContentFingerprint(Base):
    __tablename__ = "content_fingerprints"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, index=True)  # Original URL
    url_hash = Column(String, index=True)  # Hash of the URL
    content_hash = Column(String, unique=True, index=True)  # Hash of the content
    embedding_vector = Column(ARRAY(Float))  # Semantic embedding for similarity
    content_structure = Column(JSON)  # Structure fingerprint (DOM elements, etc.)
    last_modified = Column(DateTime(timezone=True))  # Last modified timestamp from headers
    etag = Column(String)  # ETag from HTTP headers
    content_length = Column(Integer)  # Content length
    cache_control = Column(String)  # Cache control headers
    expires_at = Column(DateTime(timezone=True))  # When this fingerprint expires
    scraping_job_id = Column(Integer, ForeignKey("scraping_jobs.id"))  # Link to scraping job
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Add indexes for better query performance
    __table_args__ = (
        Index('idx_content_fingerprint_url_hash', 'url_hash'),
        Index('idx_content_fingerprint_content_hash', 'content_hash'),
        Index('idx_content_fingerprint_expires', 'expires_at'),
        Index('idx_content_fingerprint_url', 'url'),
    )
    
    # Relationship
    scraping_job = relationship("ScrapingJob", back_populates="content_fingerprints")

class ScrapingJob(Base):
    __tablename__ = "scraping_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    url = Column(String, nullable=False)
    prompt = Column(String, nullable=False)
    job_type = Column(String, default="scrape") # 'scrape', 'crawl', 'dynamic_scrape'
    status = Column(String, default="pending") # 'pending', 'running', 'completed', 'failed'
    strategy_used = Column(String) # 'crawl4ai', 'scrapy', 'scrapegraphai'
    quality_score = Column(Float) # Overall quality score 0-1
    processing_time = Column(Float) # Time taken in seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    
    # Add indexes for better query performance
    __table_args__ = (
        Index('idx_scraping_job_status', 'status'),
        Index('idx_scraping_job_created', 'created_at'),
        Index('idx_scraping_job_url', 'url'),
    )
    
    # Relationships
    results = relationship("ScrapingResult", back_populates="job", cascade="all, delete-orphan")
    semantic_fingerprints = relationship("SemanticFingerprint", back_populates="job", cascade="all, delete-orphan")
    content_classifications = relationship("ContentClassification", back_populates="job", cascade="all, delete-orphan")
    quality_metrics = relationship("QualityMetric", back_populates="job", cascade="all, delete-orphan")
    images = relationship("ImageExtraction", back_populates="scraping_job", cascade="all, delete-orphan")
    tables = relationship("TableExtraction", back_populates="scraping_job", cascade="all, delete-orphan")
    structured_data = relationship("StructuredDataExtraction", back_populates="scraping_job", cascade="all, delete-orphan")
    content_fingerprints = relationship("ContentFingerprint", back_populates="scraping_job", cascade="all, delete-orphan")

class ScrapingResult(Base):
    __tablename__ = "scraping_results"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("scraping_jobs.id"))
    data = Column(JSON, nullable=False)
    raw_html = Column(Text) # Store raw HTML for analysis
    extracted_text = Column(Text) # Clean extracted text
    images_extracted = Column(JSON) # List of extracted images with metadata
    tables_extracted = Column(JSON) # Structured table data
    links_extracted = Column(JSON) # Extracted links with context
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())
    data_size_bytes = Column(Integer) # Size of extracted data
    
    # Relationships
    job = relationship("ScrapingJob", back_populates="results")
    entities = relationship("ExtractedEntity", back_populates="result", cascade="all, delete-orphan")

class SemanticFingerprint(Base):
    __tablename__ = "semantic_fingerprints"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("scraping_jobs.id"))
    content_section = Column(String) # 'header', 'main', 'sidebar', 'footer', etc.
    embedding_vector = Column(ARRAY(Float)) # Gemini embedding vector
    content_hash = Column(String, index=True) # Hash of the content for quick comparison
    similarity_threshold = Column(Float, default=0.85) # Threshold for similarity matching
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    job = relationship("ScrapingJob", back_populates="semantic_fingerprints")

class ContentClassification(Base):
    __tablename__ = "content_classifications"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("scraping_jobs.id"))
    content_type = Column(String) # 'article', 'product', 'review', 'news', 'blog', etc.
    confidence_score = Column(Float) # 0-1 confidence in classification
    classification_method = Column(String) # 'embedding_similarity', 'pattern_matching', 'llm_analysis'
    content_features = Column(JSON) # Features used for classification
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    job = relationship("ScrapingJob", back_populates="content_classifications")

class QualityMetric(Base):
    __tablename__ = "quality_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("scraping_jobs.id"))
    completeness_score = Column(Float) # How complete is the extracted data
    accuracy_score = Column(Float) # Estimated accuracy of extraction
    relevance_score = Column(Float) # How relevant is the data to the prompt
    structure_score = Column(Float) # How well-structured is the extracted data
    freshness_score = Column(Float) # How fresh/recent is the data
    validation_errors = Column(JSON) # List of validation errors found
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    job = relationship("ScrapingJob", back_populates="quality_metrics")

class ExtractedEntity(Base):
    __tablename__ = "extracted_entities"
    
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("scraping_results.id"))
    entity_type = Column(String) # 'person', 'organization', 'location', 'product', etc.
    entity_value = Column(String) # The actual entity text
    confidence_score = Column(Float) # Confidence in entity recognition
    context = Column(Text) # Surrounding context
    position_start = Column(Integer) # Start position in text
    position_end = Column(Integer) # End position in text
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    result = relationship("ScrapingResult", back_populates="entities")
    relationships = relationship("EntityRelationship", foreign_keys="EntityRelationship.source_entity_id", back_populates="source_entity")

class EntityRelationship(Base):
    __tablename__ = "entity_relationships"
    
    id = Column(Integer, primary_key=True, index=True)
    source_entity_id = Column(Integer, ForeignKey("extracted_entities.id"))
    target_entity_id = Column(Integer, ForeignKey("extracted_entities.id"))
    relationship_type = Column(String) # 'works_at', 'located_in', 'manufactured_by', etc.
    confidence_score = Column(Float) # Confidence in relationship
    evidence_text = Column(Text) # Text that supports this relationship
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    source_entity = relationship("ExtractedEntity", foreign_keys=[source_entity_id], back_populates="relationships")
    target_entity = relationship("ExtractedEntity", foreign_keys=[target_entity_id])

class ScrapingCache(Base):
    __tablename__ = "scraping_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    url_hash = Column(String, unique=True, index=True) # Hash of URL + relevant parameters
    content_hash = Column(String, index=True) # Hash of the scraped content
    embedding_vector = Column(ARRAY(Float)) # Semantic embedding for similarity
    cached_data = Column(JSON) # The cached scraping result
    cache_metadata = Column(JSON) # Metadata about caching (strategy used, etc.)
    hit_count = Column(Integer, default=0) # How many times this cache was used
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True)) # When this cache expires
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AntiDetectionLog(Base):
    __tablename__ = "anti_detection_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, index=True)
    detection_type = Column(String) # 'captcha', 'rate_limit', 'ip_block', 'js_challenge'
    detection_method = Column(String) # How we detected the anti-scraping measure
    response_strategy = Column(String) # What strategy we used to respond
    success = Column(Boolean) # Whether our response was successful
    user_agent_used = Column(String)
    proxy_used = Column(String)
    delay_applied = Column(Float) # Delay in seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ImageExtraction(Base):
    __tablename__ = "image_extractions"
    
    id = Column(Integer, primary_key=True, index=True)
    scraping_job_id = Column(Integer, ForeignKey("scraping_jobs.id"))
    image_url = Column(String)
    alt_text = Column(Text)
    caption = Column(Text)
    image_data = Column(LargeBinary) # Store actual image bytes
    image_format = Column(String) # 'jpg', 'png', 'gif', etc.
    width = Column(Integer)
    height = Column(Integer)
    file_size = Column(Integer) # Size in bytes
    context = Column(Text) # Surrounding text context
    position_in_page = Column(JSON) # x, y coordinates and other positioning info
    extracted_text = Column(Text) # OCR extracted text if applicable
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    scraping_job = relationship("ScrapingJob", back_populates="images")

class TableExtraction(Base):
    __tablename__ = "table_extractions"
    
    id = Column(Integer, primary_key=True, index=True)
    scraping_job_id = Column(Integer, ForeignKey("scraping_jobs.id"))
    table_data = Column(JSON) # Structured table data as JSON
    headers = Column(ARRAY(String)) # Table headers
    row_count = Column(Integer)
    column_count = Column(Integer)
    table_caption = Column(Text)
    table_context = Column(Text) # Surrounding context
    data_types = Column(JSON) # Inferred data types for each column
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    scraping_job = relationship("ScrapingJob", back_populates="tables")

class StructuredDataExtraction(Base):
    __tablename__ = "structured_data_extractions"
    
    id = Column(Integer, primary_key=True, index=True)
    scraping_job_id = Column(Integer, ForeignKey("scraping_jobs.id"))
    schema_type = Column(String) # 'json-ld', 'microdata', 'rdfa', 'opengraph', etc.
    structured_data = Column(JSON) # The actual structured data
    data_category = Column(String) # 'product', 'article', 'organization', etc.
    confidence_score = Column(Float) # Confidence in extraction accuracy
    validation_status = Column(String) # 'valid', 'invalid', 'partial'
    validation_errors = Column(JSON) # Any validation errors found
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    scraping_job = relationship("ScrapingJob", back_populates="structured_data")
