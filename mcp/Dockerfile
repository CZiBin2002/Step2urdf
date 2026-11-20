# Step2URDF MCP 服务镜像：轻量，只依赖 httpx + mcp SDK，通过 REST 调用 workbench 容器
FROM python:3.13-slim
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --index-url $PIP_INDEX_URL -r requirements.txt
COPY server.py smoke_test.py ./
EXPOSE 18766
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:18766/healthz', timeout=4).status == 200 else 1)"
CMD ["python", "server.py"]
