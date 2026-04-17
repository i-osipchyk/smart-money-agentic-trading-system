FROM public.ecr.aws/lambda/python:3.13

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /var/task

COPY pyproject.toml uv.lock ./

RUN uv pip install --system --no-cache \
    "ccxt>=4.5.48" \
    "langchain-anthropic>=1.4.0" \
    "langchain-openai>=0.3.0" \
    "pandas>=3.0.2" \
    "pydantic>=2.12.5" \
    "python-dotenv>=1.2.2"

COPY src/trading/ ./trading/

CMD ["trading.lambda_handler.handler"]
