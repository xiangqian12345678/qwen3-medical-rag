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
    4.neo4j安装与启动 powershell
        $ PS C:\Users\xiang> docker run -d  --name neo4j  -p 7474:7474  -p 7687:7687   -e NEO4J_AUTH=liuxq/123456 neo4j:latest
    5.ollama安装模型 powershell
        $ ollama serve  # 安装并启动 Ollama
        $ ollama pull bge-m3:latest      # 嵌入模型
        $ ollama pull qwen3:4b           # 对话模型，本机测试用一个小号大模型就可以了，真正上线可以采用微调后的领域大模型

## 研发说明

    使用声明： 随便学习下载和商业化
    正式版本： 当前还在研发中，可以稍微等待1~2周
             将会是一个产品化的，可快速配置的RAG框架，足够覆盖RAG主流功能

## 研发步骤

    1.数据处理框架    90%
        完善基础算法
    2.词库生成       90%
        停用词缺失
    3.索引构建       90% 
        配置无用字段清除
    4.单论对话RAG     
        确认检索字段配置起效
        测试排序字段配置是否生效 weighted
        配置无用字段清除
    5.多轮对话RAG
        召回RRF缺失？
    6.单轮对话agent
        rag结果，经过judge判断，没有不能解决问题，再次转到rag节点进行执行是没有意义的
        建议：  如果判断不能解决问题，要想法扩大检索，比如利用第三方搜索，该搜索要可配置
    7.多轮对话agent
        多个子问题的回答，应该通过大模型进行统一生成一个回答 _build_graph(self):
        当前的检索和数据库查询是二选一，应该修改为并行查询
    8.多模态代理模式
    9.修改类的命名
        形成标准化的RAG架构

    优化：
    针对多轮对话agent
    1. 并行检索
        db检索
        搜索
        图谱
        自研搜索
    2. Query增强
        改写
    3. 召回增强
        多路召回
        问题分析
        上位优化
        假设性文档
    4. 排序过滤
        RRF -> 压缩过滤（低相关文档过滤->无关内容过滤->冗余过滤）->重排序ReRanker->重排序(头尾位置)
    5. 上面的每一个功能可以配置是否启用
    

    数据处理
    1. 完善数据处理流程
    2. 摘要生成
    3. 假设性问题生成
    4. 元数据生成

    多模态
    1.基于多轮对话aqent该造


    流程优化：
    1. 要有判断，是否需要触发RAG，常规问题就不需要走RAG
   
