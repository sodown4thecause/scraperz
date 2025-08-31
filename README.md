# Scraperz - Advanced AI-Powered Web Scraping Platform

🚀 **Scraperz** is a cutting-edge web scraping platform that combines the power of AI with intelligent automation to deliver high-quality data extraction at scale.

## ✨ Key Features

### 🧠 AI-Powered Intelligence
- **Semantic Content Analysis**: Uses Google Gemini embeddings for intelligent content understanding
- **Multi-Modal Extraction**: Processes text, images, tables, and structured data
- **Automatic Content Classification**: Smart categorization of scraped content
- **Quality Scoring**: AI-driven assessment of content relevance and quality

### 🔄 Adaptive Scraping Engine
- **Dynamic Strategy Selection**: Automatically switches between Crawl4AI and Scrapy
- **Anti-Detection Systems**: Advanced browser fingerprint randomization
- **Incremental Scraping**: Change detection with embedding-based similarity
- **Intelligent Caching**: Reduces redundant requests and improves performance

### 📊 Real-Time Monitoring
- **Live Dashboard**: Real-time progress tracking and system health
- **Data Quality Metrics**: Comprehensive analytics and reporting
- **Performance Monitoring**: Prometheus and Grafana integration
- **Alert System**: Proactive notifications for issues and completions

### 🏗️ Production-Ready Architecture
- **Microservices Design**: Scalable FastAPI backend with Next.js frontend
- **Docker Orchestration**: Multi-stage builds with development and production configs
- **Database Optimization**: PostgreSQL with advanced indexing and caching
- **Load Balancing**: Nginx reverse proxy with rate limiting

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)
- PostgreSQL 15+ (if running locally)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd scraperz
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start development environment**
   ```bash
   make dev-up
   # or
   docker-compose -f docker-compose.dev.yml up -d
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - pgAdmin: http://localhost:5050
   - Redis Commander: http://localhost:8081
   - Mailhog: http://localhost:8025

### Production Deployment

1. **Configure production environment**
   ```bash
   cp .env.example .env.prod
   # Configure production settings
   ```

2. **Deploy with Docker Compose**
   ```bash
   make prod-deploy
   # or
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Access monitoring**
   - Application: http://localhost
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3001

## 📖 API Documentation

### Core Endpoints

#### Scraping Operations
```bash
# Start a scraping job
POST /api/scrape
{
  "url": "https://example.com",
  "extraction_strategy": "ai_powered",
  "max_pages": 100,
  "force_refresh": false
}

# Get job status
GET /api/jobs/{job_id}

# List all jobs
GET /api/jobs
```

#### Incremental Scraping
```bash
# Check for changes
POST /api/incremental/check-changes
{
  "url": "https://example.com",
  "threshold": 0.8
}

# Get cache statistics
GET /api/incremental/cache/stats

# View change history
GET /api/incremental/changes/{url_hash}
```

#### Content Analysis
```bash
# Analyze content quality
POST /api/analyze/quality
{
  "content": "...",
  "url": "https://example.com"
}

# Extract entities
POST /api/analyze/entities
{
  "text": "..."
}
```

## 🛠️ Development

### Project Structure
```
scraperz/
├── api/                    # FastAPI backend
│   ├── main.py            # Main application
│   ├── models.py          # Database models
│   ├── scraping_engine.py # Core scraping logic
│   ├── incremental_scraper.py # Change detection
│   └── ai_content_analyzer.py # AI analysis
├── app/                   # Next.js frontend
│   ├── components/        # React components
│   ├── pages/            # Page components
│   └── lib/              # Utilities
├── docker-compose.dev.yml # Development environment
├── docker-compose.prod.yml # Production environment
├── Dockerfile.api         # API container
├── Dockerfile            # Frontend container
└── Makefile              # Development commands
```

### Available Commands

```bash
# Development
make dev-up          # Start development environment
make dev-down        # Stop development environment
make dev-logs        # View development logs

# Testing
make test-api        # Run API tests
make test-frontend   # Run frontend tests
make test-e2e        # Run end-to-end tests

# Database
make db-migrate      # Run database migrations
make db-seed         # Seed database with sample data
make db-backup       # Backup database
make db-restore      # Restore database

# Production
make prod-deploy     # Deploy to production
make prod-logs       # View production logs
make prod-status     # Check service status

# Monitoring
make monitor         # Open monitoring dashboard
make metrics         # View system metrics

# Code Quality
make lint            # Run linting
make format          # Format code
make security-check  # Security audit
```

## 🔧 Configuration

### Environment Variables

Key configuration options:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/scraperz
REDIS_URL=redis://localhost:6379

# AI Services
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# Scraping
MAX_CONCURRENT_JOBS=10
DEFAULT_TIMEOUT=30
USER_AGENT_ROTATION=true

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
LOG_LEVEL=INFO
```

### Advanced Configuration

See `.env.example` for complete configuration options including:
- Performance tuning
- Security settings
- Backup configuration
- Monitoring setup
- Development tools

## 📊 Monitoring & Analytics

### Metrics Dashboard
- **System Health**: CPU, memory, disk usage
- **Scraping Performance**: Success rates, response times
- **Data Quality**: Content scores, extraction accuracy
- **Cache Efficiency**: Hit rates, storage optimization

### Alerting
- Failed scraping jobs
- System resource limits
- Data quality degradation
- Security incidents

## 🔒 Security

- **Rate Limiting**: Configurable limits per endpoint
- **Authentication**: JWT-based API authentication
- **Data Encryption**: At-rest and in-transit encryption
- **Input Validation**: Comprehensive request validation
- **Security Headers**: CORS, CSP, and security headers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` endpoint for API documentation
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join community discussions for help and ideas

---

**Built with ❤️ using FastAPI, Next.js, and Google Gemini AI**
