# -*- coding: utf-8 -*-
import logging

from langchain_community.chat_models.tongyi import ChatTongyi

from MedicalRag.agent.MedicalAgent import MedicalAgent
from MedicalRag.config.loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

if __name__ == "__main__":
    # 初始化
    config_manager = ConfigLoader()
    power_model = ChatTongyi(model="qwen-plus", temperature=0.1)
    cg = MedicalAgent(config_manager.config, power_model=power_model)
    user_input = "我这两天肚子痛，还拉肚子"
    # 运行一步
    while True:
        state = cg.answer(user_input=user_input)
        print(
            f"Agent: \n\n{state['asking_messages'][-1][-1].content if state['ask_obj'].need_ask else state['dialogue_messages'][-1]}\n\nUser:\n")

        try:
            user_input = input()
        except (UnicodeDecodeError, UnicodeError):
            # 尝试使用 GBK 编码读取（Windows 控制台默认编码）
            import msvcrt

            line = ""
            while True:
                ch = msvcrt.getwch()
                if ch == '\r':
                    msvcrt.getwch()  # 消耗换行符
                    print()
                    break
                elif ch == '\003':
                    raise KeyboardInterrupt
                else:
                    print(ch, end='', flush=True)
                    line += ch
            user_input = line
        print("\n")
