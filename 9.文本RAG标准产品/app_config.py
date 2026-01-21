from recall.kgraph.kg_loader import KGraphConfigLoader
from recall.milvus.embed_loader import EmbedConfigLoader
from rag_loader import RAGConfigLoader


class APPConfig:

    def __init__(self, rag_config_loader: RAGConfigLoader,
                 milvus_config_loader: EmbedConfigLoader,
                 kgraph_config_loader: KGraphConfigLoader):
        self.rag_config_loader = rag_config_loader
        self.milvus_config_loader = milvus_config_loader
        self.kgraph_config_loader = kgraph_config_loader

        # 拆解： RAGConfig
        self.llm = rag_config_loader.config.llm
        self.agent = rag_config_loader.config.agent
        self.multi_dialogue_rag = rag_config_loader.config.multi_dialogue_rag
