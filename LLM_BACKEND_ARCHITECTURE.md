# LLM Backend Architecture - 自建推理服務

**核心決策：不使用 Ollama，自建完整的 LLM 服務後端**

---

## 🏗️ 架構概覽

```
┌─────────────────────────────────────────────────────────────┐
│                    Animation AI Studio                       │
│                    (Application Layer)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/WebSocket
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  LLM Service Gateway                         │
│              (Load Balancing & Routing)                      │
│                    FastAPI + Redis                           │
└──────────────────────┬──────────────────────────────────────┘
            │          │          │
            ▼          ▼          ▼
      ┌─────────┐ ┌─────────┐ ┌─────────┐
      │ vLLM    │ │ vLLM    │ │ vLLM    │
      │ Service │ │ Service │ │ Service │
      │ (Qwen)  │ │(DeepSeek│ │ (Coder) │
      └─────────┘ └─────────┘ └─────────┘
           │          │           │
           └──────────┴───────────┘
                     │
                     ▼
         ┌──────────────────────┐
         │   Model Storage      │
         │ (Shared Volume/NFS)  │
         └──────────────────────┘
```

---

## 🎯 核心組件

### 1. vLLM 推理引擎 (推薦)

**為什麼選 vLLM:**
- ⚡ PagedAttention - 記憶體效率高 2-4x
- 🚀 Continuous batching - 吞吐量高 24x
- 🔌 OpenAI API 相容 - 易於整合
- 🎛️ 動態批次大小
- 📊 多GPU支持
- 🔧 量化支持 (FP8, INT8, INT4)

```python
# vLLM 服務啟動範例
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-72B-Instruct \
  --served-model-name qwen-vl-72b \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --tensor-parallel-size 2 \
  --dtype auto \
  --quantization fp8
```

**替代方案:**
- **TGI (Text Generation Inference)** - HuggingFace 官方
- **TensorRT-LLM** - NVIDIA 優化
- **llama.cpp** - CPU 推理(備用)

---

## 📁 目錄結構

```
animation-ai-studio/
├── llm_backend/
│   ├── gateway/                # API Gateway
│   │   ├── main.py            # FastAPI 主程式
│   │   ├── router.py          # 路由管理
│   │   ├── load_balancer.py   # 負載均衡
│   │   └── cache.py           # Redis 快取
│   ├── services/              # LLM 服務
│   │   ├── qwen_vl/           # Qwen2.5-VL 服務
│   │   │   ├── start.sh
│   │   │   └── config.yaml
│   │   ├── deepseek/          # DeepSeek-V3 服務
│   │   │   ├── start.sh
│   │   │   └── config.yaml
│   │   └── qwen_coder/        # Qwen2.5-Coder 服務
│   │       ├── start.sh
│   │       └── config.yaml
│   ├── models/                # 模型存儲
│   │   └── download.sh        # 下載腳本
│   ├── monitoring/            # 監控
│   │   ├── prometheus.yml
│   │   └── grafana/
│   ├── docker/                # Docker 配置
│   │   ├── vllm.Dockerfile
│   │   ├── gateway.Dockerfile
│   │   └── docker-compose.yml
│   └── scripts/               # 管理腳本
│       ├── start_all.sh
│       ├── stop_all.sh
│       └── health_check.sh
└── scripts/core/llm_client/   # 應用層客戶端
    ├── llm_client.py          # 統一客戶端介面
    └── models.py              # 請求/響應模型
```

---

## 🔧 實現細節

### 1. API Gateway (FastAPI)

```python
# llm_backend/gateway/main.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import redis
import json
from typing import List, Dict, Optional
from pydantic import BaseModel

app = FastAPI(title="LLM Service Gateway")

# Redis 快取
cache = redis.Redis(host='localhost', port=6379, db=0)

# 服務註冊表
SERVICES = {
    "qwen-vl-72b": {
        "url": "http://localhost:8000/v1",
        "type": "multimodal",
        "priority": 1
    },
    "deepseek-v3-671b": {
        "url": "http://localhost:8001/v1",
        "type": "reasoning",
        "priority": 1
    },
    "qwen-coder-32b": {
        "url": "http://localhost:8002/v1",
        "type": "code",
        "priority": 2
    }
}

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    """
    統一的聊天完成API
    相容 OpenAI API 格式
    """
    # 檢查快取
    cache_key = f"chat:{request.model}:{hash(str(request.messages))}"
    cached = cache.get(cache_key)
    if cached and not request.stream:
        return json.loads(cached)

    # 路由到對應服務
    service = SERVICES.get(request.model)
    if not service:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    # 轉發請求
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{service['url']}/chat/completions",
            json=request.dict(),
            timeout=300.0
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()

        # 快取結果 (1小時)
        if not request.stream:
            cache.setex(cache_key, 3600, json.dumps(result))

        return result

@app.post("/v1/embeddings")
async def create_embeddings(model: str, input: List[str]):
    """向量嵌入API"""
    # 路由到多模態模型
    service = SERVICES.get("qwen-vl-72b")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{service['url']}/embeddings",
            json={"model": model, "input": input}
        )
        return response.json()

@app.get("/health")
async def health_check():
    """健康檢查"""
    health = {}
    async with httpx.AsyncClient() as client:
        for name, service in SERVICES.items():
            try:
                response = await client.get(f"{service['url']}/health", timeout=5.0)
                health[name] = "healthy" if response.status_code == 200 else "unhealthy"
            except:
                health[name] = "down"
    return health

@app.get("/models")
async def list_models():
    """列出所有可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "type": info["type"]
            }
            for name, info in SERVICES.items()
        ]
    }
```

### 2. vLLM 服務配置

```yaml
# llm_backend/services/qwen_vl/config.yaml

model: Qwen/Qwen2.5-VL-72B-Instruct
served_model_name: qwen-vl-72b
port: 8000

# GPU 配置
gpu_memory_utilization: 0.9
tensor_parallel_size: 2  # 使用2張GPU
dtype: auto
quantization: fp8  # FP8 量化

# 性能配置
max_model_len: 32768
max_num_batched_tokens: 8192
max_num_seqs: 256

# 快取配置
enable_prefix_caching: true
disable_log_stats: false
```

```bash
# llm_backend/services/qwen_vl/start.sh

#!/bin/bash
set -e

MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
PORT=8000

echo "Starting Qwen2.5-VL-72B service on port $PORT..."

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --served-model-name qwen-vl-72b \
  --port $PORT \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --tensor-parallel-size 2 \
  --dtype auto \
  --quantization fp8 \
  --enable-prefix-caching \
  --trust-remote-code

echo "✅ Qwen2.5-VL-72B service started"
```

### 3. DeepSeek-V3 服務配置

```bash
# llm_backend/services/deepseek/start.sh

#!/bin/bash
set -e

MODEL="deepseek-ai/DeepSeek-V3"
PORT=8001

echo "Starting DeepSeek-V3 service on port $PORT..."

# DeepSeek-V3 使用 FP8 量化在單卡 A100 80GB 運行
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --served-model-name deepseek-v3-671b \
  --port $PORT \
  --gpu-memory-utilization 0.95 \
  --max-model-len 65536 \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --quantization fp8 \
  --enable-prefix-caching \
  --trust-remote-code \
  --max-num-seqs 128

echo "✅ DeepSeek-V3 service started"
```

### 4. Docker Compose 編排

```yaml
# llm_backend/docker/docker-compose.yml

version: '3.8'

services:
  # Redis 快取
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  # API Gateway
  gateway:
    build:
      context: ..
      dockerfile: docker/gateway.Dockerfile
    ports:
      - "7000:7000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis
    restart: unless-stopped

  # Qwen2.5-VL 服務
  qwen-vl:
    build:
      context: ..
      dockerfile: docker/vllm.Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MODEL=Qwen/Qwen2.5-VL-72B-Instruct
      - PORT=8000
      - TENSOR_PARALLEL_SIZE=2
    volumes:
      - ../models:/models
      - /dev/shm:/dev/shm  # 共享記憶體
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']  # GPU 0, 1
              capabilities: [gpu]
    restart: unless-stopped

  # DeepSeek-V3 服務
  deepseek:
    build:
      context: ..
      dockerfile: docker/vllm.Dockerfile
    ports:
      - "8001:8001"
    environment:
      - MODEL=deepseek-ai/DeepSeek-V3
      - PORT=8001
      - TENSOR_PARALLEL_SIZE=1
      - QUANTIZATION=fp8
    volumes:
      - ../models:/models
      - /dev/shm:/dev/shm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2']  # GPU 2 (A100 80GB)
              capabilities: [gpu]
    restart: unless-stopped

  # Qwen2.5-Coder 服務
  qwen-coder:
    build:
      context: ..
      dockerfile: docker/vllm.Dockerfile
    ports:
      - "8002:8002"
    environment:
      - MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
      - PORT=8002
      - TENSOR_PARALLEL_SIZE=1
    volumes:
      - ../models:/models
      - /dev/shm:/dev/shm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['3']  # GPU 3
              capabilities: [gpu]
    restart: unless-stopped

  # Prometheus 監控
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ../monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    restart: unless-stopped

  # Grafana 可視化
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ../monitoring/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

---

## 🖥️ 應用層客戶端

```python
# scripts/core/llm_client/llm_client.py

import httpx
import json
from typing import List, Dict, Optional, AsyncIterator
from pydantic import BaseModel

class LLMClient:
    """
    統一的 LLM 客戶端
    連接到自建 LLM 服務後端
    """

    def __init__(self, base_url: str = "http://localhost:7000/v1"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict:
        """
        發送聊天請求

        Args:
            model: 模型名稱 (qwen-vl-72b, deepseek-v3-671b, qwen-coder-32b)
            messages: 對話歷史
            temperature: 溫度參數
            max_tokens: 最大生成長度
            stream: 是否串流輸出
        """
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
        )
        return response.json()

    async def understand_creative_intent(self, user_request: str) -> Dict:
        """
        使用 DeepSeek-V3 理解創意意圖
        """
        messages = [{
            "role": "system",
            "content": "You are a creative director AI specializing in animation content creation."
        }, {
            "role": "user",
            "content": f"""Analyze this creative request in detail:

{user_request}

Provide:
1. Core creative goal
2. Desired style and mood
3. Target audience
4. Success criteria
5. Technical challenges

Return as JSON."""
        }]

        response = await self.chat(
            model="deepseek-v3-671b",
            messages=messages,
            temperature=0.3
        )

        # 解析 JSON 回應
        content = response['choices'][0]['message']['content']
        return json.loads(content)

    async def analyze_video_content(
        self,
        video_frames: List[str],  # base64 encoded
        analysis_focus: str
    ) -> Dict:
        """
        使用 Qwen2.5-VL 分析影片內容
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Analyze this video focusing on: {analysis_focus}"},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
                  for frame in video_frames[:10]]  # 最多10幀
            ]
        }]

        response = await self.chat(
            model="qwen-vl-72b",
            messages=messages,
            temperature=0.2
        )

        return response

    async def generate_code(self, task_description: str) -> str:
        """
        使用 Qwen2.5-Coder 生成代碼
        """
        messages = [{
            "role": "system",
            "content": "You are an expert Python programmer specializing in AI tools and automation."
        }, {
            "role": "user",
            "content": task_description
        }]

        response = await self.chat(
            model="qwen-coder-32b",
            messages=messages,
            temperature=0.1
        )

        return response['choices'][0]['message']['content']

    async def health_check(self) -> Dict:
        """檢查服務健康狀態"""
        response = await self.client.get(f"{self.base_url.replace('/v1', '')}/health")
        return response.json()

# 使用範例
async def example_usage():
    client = LLMClient()

    # 理解創意意圖
    intent = await client.understand_creative_intent(
        "Create a funny parody of Luca's ocean scene"
    )
    print("Creative Intent:", intent)

    # 分析影片
    analysis = await client.analyze_video_content(
        video_frames=[...],  # base64 frames
        analysis_focus="comedic moments and character expressions"
    )
    print("Video Analysis:", analysis)

    # 生成代碼
    code = await client.generate_code(
        "Write a function to apply slow-motion effect to video using MoviePy"
    )
    print("Generated Code:", code)
```

---

## 🚀 部署指南

### 本地部署 (開發環境)

```bash
# 1. 下載模型
cd llm_backend/models
bash download.sh

# 2. 啟動服務
cd ../
./scripts/start_all.sh

# 3. 檢查健康狀態
./scripts/health_check.sh
```

### Docker 部署 (生產環境)

```bash
# 1. 構建鏡像
cd llm_backend/docker
docker-compose build

# 2. 啟動所有服務
docker-compose up -d

# 3. 查看日誌
docker-compose logs -f gateway

# 4. 擴展服務
docker-compose up -d --scale qwen-vl=2
```

### Kubernetes 部署 (大規模)

```yaml
# 待實現 - K8s manifests
```

---

## 📊 監控和日誌

### Prometheus 指標

```yaml
# llm_backend/monitoring/prometheus.yml

global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets:
        - 'qwen-vl:8000'
        - 'deepseek:8001'
        - 'qwen-coder:8002'

  - job_name: 'gateway'
    static_configs:
      - targets: ['gateway:7000']
```

### Grafana Dashboard

- 請求吞吐量
- 延遲分佈 (P50, P95, P99)
- GPU 使用率
- 記憶體使用
- 錯誤率

---

## 🔐 安全考慮

### 1. API 認證

```python
# 添加 JWT 認證
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # 驗證 token
    verify_token(credentials.credentials)
    # ...
```

### 2. 速率限制

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/chat/completions")
@limiter.limit("100/minute")
async def chat_completion(...):
    ...
```

---

## 💰 成本估算

### GPU 需求

```yaml
Qwen2.5-VL 72B (FP8):
  - 2x RTX 4090 (24GB each) = ~$3,200
  - 或 1x A6000 (48GB) = ~$4,500
  - 或 雲端 A100 40GB x2 = ~$2-3/hour

DeepSeek-V3 671B (FP8):
  - 1x A100 80GB = ~$6,000
  - 或 雲端 A100 80GB = ~$3-4/hour

Qwen2.5-Coder 32B:
  - 1x RTX 4090 (24GB) = ~$1,600
  - 或 雲端 A10G = ~$1/hour

Total (本地):
  - 約 $10,000-15,000 (一次性)

Total (雲端):
  - 約 $5-8/hour (按需)
```

---

## ✅ 總結

這個自建 LLM 後端架構提供：

1. ✅ **完全自主控制** - 不依賴 Ollama
2. ✅ **高性能推理** - vLLM 優化
3. ✅ **彈性擴展** - Docker/K8s 支持
4. ✅ **OpenAI 相容** - 易於遷移
5. ✅ **完整監控** - Prometheus + Grafana
6. ✅ **負載均衡** - 多服務協調
7. ✅ **快取優化** - Redis 加速

**下一步:** 開始實現 Gateway 和第一個 vLLM 服務！
