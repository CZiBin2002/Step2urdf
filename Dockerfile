# Step2URDF 工作台镜像：前端构建 + FastAPI 运行时
# 构建时可覆盖镜像源（默认国内源，海外构建请传官方源）：
#   docker build --build-arg PIP_INDEX_URL=https://pypi.org/simple \
#                --build-arg NPM_REGISTRY=https://registry.npmjs.org .

FROM node:22-slim AS frontend
ARG NPM_REGISTRY=https://registry.npmmirror.com
WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci --include=dev --no-audit --no-fund --registry=$NPM_REGISTRY
COPY index.html vite.config.js ./
COPY frontend ./frontend
RUN npm run build

FROM python:3.13-slim AS runtime
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STEP2URDF_PROJECTS=/data/projects
WORKDIR /app
COPY requirements.txt ./
RUN pip install --index-url $PIP_INDEX_URL -r requirements.txt
# OpenCASCADE (OCP) 运行时需要的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglu1-mesa libxext6 libx11-6 \
    && rm -rf /var/lib/apt/lists/*
COPY workbench ./workbench
COPY --from=frontend /src/dist ./dist
EXPOSE 18765
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:18765/api/projects', timeout=8).status == 200 else 1)"
CMD ["python", "-m", "uvicorn", "workbench.server:app", "--host", "0.0.0.0", "--port", "18765"]
