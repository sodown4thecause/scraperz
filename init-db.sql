-- Scraperz Database Initialization Script
-- This script sets up the PostgreSQL database with required extensions and configurations

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
CREATE EXTENSION IF NOT EXISTS "vector" WITH SCHEMA public;

-- Create custom types
DO $$ BEGIN
    CREATE TYPE scraping_status AS ENUM (
        'pending',
        'running',
        'completed',
        'failed',
        'cancelled'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE content_type AS ENUM (
        'text',
        'html',
        'json',
        'xml',
        'csv',
        'pdf',
        'image',
        'video',
        'audio',
        'other'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE extraction_strategy AS ENUM (
        'crawl4ai',
        'scrapy',
        'selenium',
        'playwright',
        'requests',
        'hybrid'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE change_type AS ENUM (
        'no_change',
        'minor_change',
        'moderate_change',
        'major_change',
        'complete_change'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create indexes for better performance
-- Note: These will be created automatically by SQLAlchemy, but we can add custom ones here

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Function to calculate content similarity using cosine similarity
CREATE OR REPLACE FUNCTION cosine_similarity(vec1 vector, vec2 vector)
RETURNS float AS $$
BEGIN
    RETURN (vec1 <#> vec2) * -1 + 1;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- Function to generate URL hash
CREATE OR REPLACE FUNCTION generate_url_hash(url text)
RETURNS text AS $$
BEGIN
    RETURN encode(digest(url, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- Function to generate content hash
CREATE OR REPLACE FUNCTION generate_content_hash(content text)
RETURNS text AS $$
BEGIN
    RETURN encode(digest(content, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- Function to clean old cache entries
CREATE OR REPLACE FUNCTION clean_old_cache_entries(retention_days integer DEFAULT 30)
RETURNS integer AS $$
DECLARE
    deleted_count integer;
BEGIN
    DELETE FROM content_fingerprints 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * retention_days;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get scraping statistics
CREATE OR REPLACE FUNCTION get_scraping_stats()
RETURNS TABLE(
    total_jobs bigint,
    completed_jobs bigint,
    failed_jobs bigint,
    pending_jobs bigint,
    running_jobs bigint,
    success_rate numeric,
    avg_duration_seconds numeric
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_jobs,
        COUNT(*) FILTER (WHERE status = 'completed') as completed_jobs,
        COUNT(*) FILTER (WHERE status = 'failed') as failed_jobs,
        COUNT(*) FILTER (WHERE status = 'pending') as pending_jobs,
        COUNT(*) FILTER (WHERE status = 'running') as running_jobs,
        CASE 
            WHEN COUNT(*) > 0 THEN 
                ROUND((COUNT(*) FILTER (WHERE status = 'completed')::numeric / COUNT(*)::numeric) * 100, 2)
            ELSE 0
        END as success_rate,
        ROUND(AVG(EXTRACT(EPOCH FROM (completed_at - started_at))), 2) as avg_duration_seconds
    FROM scraping_jobs
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- Function to find similar URLs based on content fingerprints
CREATE OR REPLACE FUNCTION find_similar_urls(
    target_url text,
    similarity_threshold float DEFAULT 0.8,
    limit_results integer DEFAULT 10
)
RETURNS TABLE(
    url text,
    similarity_score float,
    last_scraped timestamp with time zone
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cf.url,
        cosine_similarity(
            (SELECT embedding_vector FROM content_fingerprints WHERE url = target_url LIMIT 1),
            cf.embedding_vector
        ) as similarity_score,
        cf.last_scraped
    FROM content_fingerprints cf
    WHERE cf.url != target_url
        AND cf.embedding_vector IS NOT NULL
        AND cosine_similarity(
            (SELECT embedding_vector FROM content_fingerprints WHERE url = target_url LIMIT 1),
            cf.embedding_vector
        ) >= similarity_threshold
    ORDER BY similarity_score DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;

-- Create materialized view for scraping analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS scraping_analytics AS
SELECT 
    DATE_TRUNC('day', created_at) as date,
    COUNT(*) as total_jobs,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_jobs,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_jobs,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds,
    AVG(confidence_score) as avg_confidence_score,
    COUNT(DISTINCT url) as unique_urls
FROM scraping_jobs
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '90 days'
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY date DESC;

-- Create index on the materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_scraping_analytics_date ON scraping_analytics (date);

-- Function to refresh analytics
CREATE OR REPLACE FUNCTION refresh_scraping_analytics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY scraping_analytics;
END;
$$ LANGUAGE plpgsql;

-- Create a scheduled job to refresh analytics (requires pg_cron extension)
-- SELECT cron.schedule('refresh-analytics', '0 1 * * *', 'SELECT refresh_scraping_analytics();');

-- Grant permissions to the scraperz user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO scraperz;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO scraperz;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO scraperz;
GRANT USAGE ON SCHEMA public TO scraperz;

-- Insert initial configuration data
INSERT INTO scraping_configs (key, value, description) VALUES
    ('default_timeout', '30', 'Default timeout for scraping requests in seconds'),
    ('max_retries', '3', 'Maximum number of retry attempts for failed requests'),
    ('delay_between_requests', '1', 'Default delay between requests in seconds'),
    ('max_concurrent_requests', '10', 'Maximum number of concurrent scraping requests'),
    ('cache_ttl', '3600', 'Default cache TTL in seconds'),
    ('similarity_threshold', '0.85', 'Similarity threshold for change detection'),
    ('cleanup_interval', '86400', 'Cache cleanup interval in seconds')
ON CONFLICT (key) DO NOTHING;

-- Create notification function for real-time updates
CREATE OR REPLACE FUNCTION notify_scraping_job_change()
RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM pg_notify('scraping_job_created', row_to_json(NEW)::text);
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        PERFORM pg_notify('scraping_job_updated', row_to_json(NEW)::text);
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        PERFORM pg_notify('scraping_job_deleted', row_to_json(OLD)::text);
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Note: Triggers will be created by SQLAlchemy when tables are created

-- Create indexes for full-text search
-- These will be created after tables are set up by SQLAlchemy

-- Log successful initialization
INSERT INTO system_logs (level, message, details) VALUES
    ('INFO', 'Database initialized successfully', 
     json_build_object(
         'timestamp', CURRENT_TIMESTAMP,
         'extensions', ARRAY['uuid-ossp', 'pg_trgm', 'btree_gin', 'btree_gist', 'vector'],
         'functions_created', 8,
         'types_created', 4
     ))
ON CONFLICT DO NOTHING;

COMMIT;