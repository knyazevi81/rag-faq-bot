DC = docker compose
EXEC = docker exec -it 
ENV = --env-file .env
COMPOSE_FILE = docker/vector_storage.yml
APP_NAME = faqbot
LOGS = docker logs


.PHONY: database
database:
	docker compose -f compose/vector_storage.yml --env-file .env up --build

.PHONY: database-stop
database-stop:
	docker compose -f compose/vector_storage.yml down


# docker compose -f compose/vector_storage.yml --env-file .env up --build
