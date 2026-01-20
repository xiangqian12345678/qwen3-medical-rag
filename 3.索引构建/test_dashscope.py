"""测试 DashScope 向量模型集成"""
from config import ConfigLoader
from embedding_client import create_embedding_client


def test_dashscope_embedding():
    """测试 DashScope 向量嵌入功能"""
    print("=" * 60)
    print("测试 DashScope 向量模型集成")
    print("=" * 60)

    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.config

    print(f"\n配置加载成功!")
    print(f"Milvus 集合: {config.milvus.collection_name}")

    # 检查稠密向量字段配置
    for field_name, field_config in config.dense_fields.items():
        print(f"\n字段: {field_name}")
        print(f"  - embed: {field_config.embed}")
        print(f"  - type: {field_config.type}")
        print(f"  - provider: {field_config.provider}")
        print(f"  - model: {field_config.model}")
        print(f"  - dimension: {field_config.dimension}")

        if field_config.embed and field_config.provider == "dashscope":
            print(f"\n创建 DashScope 嵌入客户端...")
            try:
                client = create_embedding_client(field_config)

                # 测试单个文本嵌入
                test_text = "阿司匹林是一种常用的解热镇痛药"
                print(f"\n测试文本: {test_text}")

                print("生成嵌入向量...")
                embedding = client.embed_query(test_text)
                print(f"嵌入向量维度: {len(embedding)}")
                print(f"前5个值: {embedding[:5]}")

                # 测试批量嵌入
                print("\n测试批量嵌入...")
                texts = [
                    "阿司匹林治疗头痛",
                    "布洛芬缓解发热",
                    "感冒病毒引起上呼吸道感染"
                ]
                embeddings = client.embed_documents(texts)
                print(f"批量嵌入完成，共 {len(embeddings)} 个向量")
                for i, emb in enumerate(embeddings):
                    print(f"  文本 {i+1} 维度: {len(emb)}")

                # 如果客户端有关闭方法，关闭它
                if hasattr(client, 'close'):
                    client.close()

                print(f"\n✅ DashScope 向量模型测试成功!")
                break  # 只测试一个字段

            except Exception as e:
                print(f"\n❌ DashScope 嵌入测试失败: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    test_dashscope_embedding()
