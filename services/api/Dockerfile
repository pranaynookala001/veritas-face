FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN groupadd --gid 10001 veritas \
    && useradd --uid 10001 --gid veritas --create-home --shell /usr/sbin/nologin veritas \
    && install --directory --owner veritas --group veritas --mode 0700 /var/lib/veritas-face/artifacts

COPY services/api/pyproject.toml ./pyproject.toml
COPY services/api/app ./app
RUN python -m pip install --no-cache-dir .

USER veritas

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
