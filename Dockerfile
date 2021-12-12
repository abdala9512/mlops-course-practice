FROM python:3.9-slim-buster

WORKDIR /app
COPY api/poetry.lock

RUN poetry install

COPY api/ ./api
COPY model/model.pkl ./model/model.pkl
COPY initializer.sh .

RUN chmod +x initializer.sh

EXPOSE 8000

ENTRYPOINT ["./initializer.sh"]