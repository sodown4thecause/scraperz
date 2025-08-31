# Scraperz Project Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help install dev prod test clean build deploy logs status stop restart

# Default target
help:
	@echo "Scraperz Project Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make install     - Install dependencies for development"
	@echo "  make dev         - Start development environment"
	@echo "  make dev-build   - Build and start development environment"
	@echo "  make dev-logs    - Show development logs"
	@echo "  make dev-stop    - Stop development environment"
	@echo ""
	@echo "Production:"
	@echo "  make prod        - Start production environment"
	@echo "  make prod-build  - Build and start production environment"
	@echo "  make prod-logs   - Show production logs"
	@echo "  make prod-stop   - Stop production environment"
	@echo ""
	@echo "Testing:"
	@echo "  make test        - Run all tests"
	@echo "  make test-api    - Run API tests"
	@echo "  make test-frontend - Run frontend tests"
	@echo "  make test-e2e    - Run end-to-end tests"
	@echo "  make coverage    - Generate test coverage report"
	@echo ""
	@echo "Database:"
	@echo "  make db-migrate  - Run database migrations"
	@echo "  make db-seed     - Seed database with sample data"
	@echo "  make db-reset    - Reset database (WARNING: destroys data)"
	@echo "  make db-backup   - Create database backup"
	@echo "  make db-restore  - Restore database from backup"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean       - Clean up containers and volumes"
	@echo "  make clean-all   - Clean everything including images"
	@echo "  make logs        - Show all service logs"
	@echo "  make status      - Show service status"
	@echo "  make restart     - Restart all services"
	@echo "  make update      - Update dependencies and rebuild"
	@echo ""
	@echo "Monitoring:"
	@echo "  make monitor     - Open monitoring dashboard"
	@echo "  make metrics     - Show system metrics"
	@echo "  make health      - Check service health"

# =============================================================================
# DEVELOPMENT COMMANDS
# =============================================================================

install:
	@echo "Installing development dependencies..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file from template"; fi
	npm install
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "Dependencies installed successfully!"

dev:
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "Development environment started!"
	@echo "Frontend: http://localhost:3001"
	@echo "API: http://localhost:8001"
	@echo "pgAdmin: http://localhost:5050"
	@echo "Redis Commander: http://localhost:8081"
	@echo "Jupyter: http://localhost:8888"

dev-build:
	@echo "Building and starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d --build
	@echo "Development environment built and started!"

dev-logs:
	@echo "Showing development logs..."
	docker-compose -f docker-compose.dev.yml logs -f

dev-stop:
	@echo "Stopping development environment..."
	docker-compose -f docker-compose.dev.yml down
	@echo "Development environment stopped!"

# =============================================================================
# PRODUCTION COMMANDS
# =============================================================================

prod:
	@echo "Starting production environment..."
	@if [ ! -f .env.prod ]; then echo "ERROR: .env.prod file not found!"; exit 1; fi
	docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d
	@echo "Production environment started!"
	@echo "Application: http://localhost"
	@echo "Monitoring: http://localhost:3001"

prod-build:
	@echo "Building and starting production environment..."
	@if [ ! -f .env.prod ]; then echo "ERROR: .env.prod file not found!"; exit 1; fi
	docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d --build
	@echo "Production environment built and started!"

prod-logs:
	@echo "Showing production logs..."
	docker-compose -f docker-compose.prod.yml logs -f

prod-stop:
	@echo "Stopping production environment..."
	docker-compose -f docker-compose.prod.yml down
	@echo "Production environment stopped!"

# =============================================================================
# TESTING COMMANDS
# =============================================================================

test:
	@echo "Running all tests..."
	@make test-api
	@make test-frontend
	@echo "All tests completed!"

test-api:
	@echo "Running API tests..."
	docker-compose -f docker-compose.dev.yml exec api pytest tests/ -v --cov=api --cov-report=html

test-frontend:
	@echo "Running frontend tests..."
	docker-compose -f docker-compose.dev.yml exec frontend npm test

test-e2e:
	@echo "Running end-to-end tests..."
	docker-compose -f docker-compose.dev.yml exec frontend npm run test:e2e

coverage:
	@echo "Generating test coverage report..."
	docker-compose -f docker-compose.dev.yml exec api pytest tests/ --cov=api --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

# =============================================================================
# DATABASE COMMANDS
# =============================================================================

db-migrate:
	@echo "Running database migrations..."
	docker-compose -f docker-compose.dev.yml exec api alembic upgrade head

db-seed:
	@echo "Seeding database with sample data..."
	docker-compose -f docker-compose.dev.yml exec api python scripts/seed_database.py

db-reset:
	@echo "WARNING: This will destroy all data in the database!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	docker-compose -f docker-compose.dev.yml exec api alembic downgrade base
	docker-compose -f docker-compose.dev.yml exec api alembic upgrade head
	@echo "Database reset completed!"

db-backup:
	@echo "Creating database backup..."
	@mkdir -p backups
	docker-compose -f docker-compose.dev.yml exec postgres pg_dump -U scraperz scraperz_dev > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Database backup created in backups/"

db-restore:
	@echo "Restoring database from backup..."
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Usage: make db-restore BACKUP_FILE=path/to/backup.sql"; exit 1; fi
	docker-compose -f docker-compose.dev.yml exec -T postgres psql -U scraperz -d scraperz_dev < $(BACKUP_FILE)
	@echo "Database restored from $(BACKUP_FILE)"

# =============================================================================
# MAINTENANCE COMMANDS
# =============================================================================

clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose -f docker-compose.dev.yml down -v
	docker-compose -f docker-compose.prod.yml down -v
	docker system prune -f
	@echo "Cleanup completed!"

clean-all:
	@echo "WARNING: This will remove all containers, volumes, and images!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	docker-compose -f docker-compose.dev.yml down -v --rmi all
	docker-compose -f docker-compose.prod.yml down -v --rmi all
	docker system prune -af
	@echo "Complete cleanup finished!"

logs:
	@echo "Showing all service logs..."
	@if [ -f docker-compose.dev.yml ] && docker-compose -f docker-compose.dev.yml ps -q > /dev/null 2>&1; then \
		docker-compose -f docker-compose.dev.yml logs -f; \
	else \
		docker-compose -f docker-compose.prod.yml logs -f; \
	fi

status:
	@echo "Service Status:"
	@echo "Development:"
	@docker-compose -f docker-compose.dev.yml ps 2>/dev/null || echo "  Not running"
	@echo "Production:"
	@docker-compose -f docker-compose.prod.yml ps 2>/dev/null || echo "  Not running"

restart:
	@echo "Restarting all services..."
	@if docker-compose -f docker-compose.dev.yml ps -q > /dev/null 2>&1; then \
		docker-compose -f docker-compose.dev.yml restart; \
	else \
		docker-compose -f docker-compose.prod.yml restart; \
	fi
	@echo "Services restarted!"

update:
	@echo "Updating dependencies and rebuilding..."
	npm update
	pip install -r requirements.txt --upgrade
	@if docker-compose -f docker-compose.dev.yml ps -q > /dev/null 2>&1; then \
		make dev-build; \
	else \
		make prod-build; \
	fi
	@echo "Update completed!"

# =============================================================================
# MONITORING COMMANDS
# =============================================================================

monitor:
	@echo "Opening monitoring dashboard..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open http://localhost:3001; \
	elif command -v open > /dev/null; then \
		open http://localhost:3001; \
	else \
		echo "Please open http://localhost:3001 in your browser"; \
	fi

metrics:
	@echo "System Metrics:"
	@docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

health:
	@echo "Checking service health..."
	@echo "API Health:"
	@curl -s http://localhost:8001/health 2>/dev/null || curl -s http://localhost:8000/health 2>/dev/null || echo "  API not responding"
	@echo "Frontend Health:"
	@curl -s http://localhost:3001/health 2>/dev/null || curl -s http://localhost:3000/health 2>/dev/null || echo "  Frontend not responding"

# =============================================================================
# UTILITY COMMANDS
# =============================================================================

shell-api:
	@echo "Opening API shell..."
	docker-compose -f docker-compose.dev.yml exec api bash

shell-frontend:
	@echo "Opening frontend shell..."
	docker-compose -f docker-compose.dev.yml exec frontend bash

shell-db:
	@echo "Opening database shell..."
	docker-compose -f docker-compose.dev.yml exec postgres psql -U scraperz -d scraperz_dev

format:
	@echo "Formatting code..."
	black api/
	isort api/
	npm run format
	@echo "Code formatting completed!"

lint:
	@echo "Running linters..."
	flake8 api/
	mypy api/
	npm run lint
	@echo "Linting completed!"

security:
	@echo "Running security checks..."
	bandit -r api/
	npm audit
	@echo "Security checks completed!"

# =============================================================================
# DEPLOYMENT COMMANDS
# =============================================================================

deploy-staging:
	@echo "Deploying to staging..."
	@echo "This would deploy to staging environment"
	# Add your staging deployment commands here

deploy-prod:
	@echo "Deploying to production..."
	@echo "WARNING: This will deploy to production!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo "This would deploy to production environment"
	# Add your production deployment commands here

# =============================================================================
# DOCUMENTATION COMMANDS
# =============================================================================

docs:
	@echo "Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html/
	@echo "Documentation generated in docs/_build/html/"

docs-serve:
	@echo "Serving documentation..."
	cd docs/_build/html && python -m http.server 8080