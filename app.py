"""Entry point: launch the Gradio app."""

import logging

from rag_grounded_chatbot.config import Config
from rag_grounded_chatbot.pipeline import GroundedRAGPipeline
from rag_grounded_chatbot.ui import GroundedChatbotUI

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    config = Config()
    pipeline = GroundedRAGPipeline(config)
    ui = GroundedChatbotUI(pipeline)
    demo = ui.build()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()