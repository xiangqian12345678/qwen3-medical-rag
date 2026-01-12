"""词表管理"""
import logging
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class Vocabulary:
    """词表数据结构"""
    token2id: Dict[str, int] = field(default_factory=dict)
    id2token: Dict[int, str] = field(default_factory=dict)
    doc_freq: Dict[int, int] = field(default_factory=dict)
    idf: Dict[int, float] = field(default_factory=dict)
    N: int = 0
    avgdl: float = 0.0

    def __post_init__(self):
        """初始化后处理"""
        if not self.id2token:
            self.id2token = {v: k for k, v in self.token2id.items()}

    @classmethod
    def load(cls, vocab_path: str) -> Optional['Vocabulary']:
        """
        加载词表文件

        Args:
            vocab_path: 词表文件路径，支持 .pkl.gz 和 .pkl 格式

        Returns:
            Vocabulary 实例，如果加载失败返回 None
        """
        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            base_dir = Path(__file__).parent.parent.parent / "output" / "vocab"
            vocab_path = base_dir / vocab_path.name

        try:
            if str(vocab_path).endswith('.gz'):
                with gzip.open(vocab_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(vocab_path, 'rb') as f:
                    data = pickle.load(f)

            if isinstance(data, dict):
                valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                return cls(**valid_fields)
            else:
                return cls(**data)
        except Exception as e:
            logger.error(f"词表加载失败: {vocab_path}, 错误: {e}")
            return None

    def save(self, save_path: str):
        """
        保存词表文件

        Args:
            save_path: 保存路径，支持 .pkl.gz 和 .pkl 格式
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'token2id': self.token2id,
            'id2token': self.id2token,
            'doc_freq': self.doc_freq,
            'idf': self.idf,
            'N': self.N,
            'avgdl': self.avgdl,
        }

        if str(save_path).endswith('.gz'):
            with gzip.open(save_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

        logger.info(f"词表已保存: {save_path}")

    def add_token(self, token: str) -> int:
        """
        添加token到词表

        Args:
            token: 文本token

        Returns:
            token ID
        """
        if token not in self.token2id:
            token_id = len(self.token2id)
            self.token2id[token] = token_id
            self.id2token[token_id] = token
            self.doc_freq[token_id] = 0
        return self.token2id[token]

    def update_doc_freq(self, token_ids: List[int]):
        """
        更新文档频率

        Args:
            token_ids: 文档中的token ID列表（去重后）
        """
        for token_id in set(token_ids):
            self.doc_freq[token_id] = self.doc_freq.get(token_id, 0) + 1

    def compute_idf(self):
        """计算IDF值"""
        import math
        self.N = max(self.N, 1)
        for token_id, df in self.doc_freq.items():
            self.idf[token_id] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def tokenize(self, text: str, domain_model: str = "medicine") -> List[int]:
        """
        分词并返回token ID列表

        Args:
            text: 输入文本
            domain_model: 领域模型类型

        Returns:
            token ID列表
        """
        if domain_model == "medicine":
            import re
            tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+|\S', text)
        else:
            tokens = text.split()

        token_ids = []
        for token in tokens:
            if token in self.token2id:
                token_ids.append(self.token2id[token])

        return token_ids

    def get_vocabulary_size(self) -> int:
        """获取词表大小"""
        return len(self.token2id)
