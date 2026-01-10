# 参考代码

[medical-rag](https://github.com/ImprintLab/Medical-Graph-RAG)

[open-graph-rag](https://github.com/OpenGraphRAG/MedicalGraphRAG?tab=readme-ov-file)

[medical-graph-rag](https://github.com/OpenGraphRAG/MedicalGraphRAG)

[ai-medical-assistant](https://github.com/zhttyy520/ai-medical-assistant)

# 快速开始

## 创建环境

    conda create -name medicalrag python=3.11
    pip install -e .

## 基础服务

    1.安装docker desktop
        下载地址：  https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
        安装测试：  docker --version (powershell)
    2.安装ollma
        下载地址：  https://ollama.cc/download.html
    3.milvus安装与启动 powershell
        单例配置下载： 
        $ curl -O https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml
        $ docker compose up -d    # 首次会拉镜像，耐心等 1–3 分钟
        $ docker ps     # 查看运行容器
    4.ollama安装模型 powershell
        $ ollama serve  # 安装并启动 Ollama
        $ ollama pull bge-m3:latest      # 嵌入模型
        $ ollama pull qwen3:4b           # 对话模型，本机测试用一个小号大模型就可以了，真正上线可以采用微调后的领域大模型

##  