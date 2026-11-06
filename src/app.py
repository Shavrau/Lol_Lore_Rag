"""
Interactive Gradio Web Application for Runeterra Lore RAG.
Supports multi-source lore querying (Books + Web Lore) with hybrid retrieval and reranking.
"""

import logging
from typing import List, Tuple
import gradio as gr

from src.book_extractor import BookExtractor
from src.config import DEFAULT_RETRIEVAL_MODE, get_book_paths
from src.rag_engine import RuneterraRAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("App")

engine = RuneterraRAGEngine()
book_extractor = BookExtractor()

# Load initial data
logger.info("Initializing knowledge base...")
books_pages = book_extractor.extract_all_books()
engine.ingest(pages=books_pages, include_web_lore=True)

EXAMPLES = [
    ["Quem é o Rei Destruído e como a Ruína começou?"],
    ["Como funciona a magia e os petricitas em Demacia?"],
    ["O que são os Darkin e qual a sua origem?"],
    ["Quem são os Guardiões das Ilhas das Sombras antes da corrupção?"],
    ["What is the relationship between Viego, Isolde, and Kalista?"],
]

AVAILABLE_SOURCES = sorted(list(engine.source_ids.keys())) or ["No sources loaded"]
MODES = ["hybrid+rerank", "hybrid", "dense", "bm25"]


def ui_ask(question: str, selected_sources: List[str], top_k: int, mode: str) -> Tuple[str, List[List[str]]]:
    if not selected_sources or "No sources loaded" in selected_sources:
        selected_sources = None
    return engine.answer_question(question, selected_sources, top_k, mode)


def create_demo():
    custom_css = """
    .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .title-banner { text-align: center; margin-bottom: 20px; }
    """

    with gr.Blocks(title="Runeterra Lore RAG", css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes=["title-banner"]):
            gr.Markdown(
                """
                # Runeterra Lore RAG
                ### O Oráculo da Lore de League of Legends & Universo de Runeterra
                Pesquise histórias, biografias de campeões, livros canônicos (*Realms of Runeterra*, *Ruination*) e contos oficiais.
                Respostas rigorosamente aterradas nas fontes com citações precisas.
                """
            )

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Sua Pergunta sobre a Lore",
                    placeholder="Ex: Como Viego causou a Ruína das Ilhas das Bênçãos?",
                    lines=3,
                )

                with gr.Accordion("Opções Avançadas e Filtros de Fonte", open=False):
                    sources_input = gr.CheckboxGroup(
                        choices=AVAILABLE_SOURCES,
                        value=AVAILABLE_SOURCES,
                        label="Fontes de Lore a pesquisar",
                    )
                    with gr.Row():
                        top_k_input = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Candidatos Rerank (Top K)",
                        )
                        mode_input = gr.Dropdown(
                            choices=MODES,
                            value=DEFAULT_RETRIEVAL_MODE,
                            label="Modo de Recuperação",
                        )

                ask_button = gr.Button("Consultar o Oráculo", variant="primary", size="lg")
                gr.Examples(examples=EXAMPLES, inputs=[question_input])

            with gr.Column(scale=4):
                answer_output = gr.Textbox(
                    label="Resposta do Guardião",
                    lines=12,
                    interactive=False,
                )
                sources_output = gr.Dataframe(
                    headers=["Fonte", "Páginas/Local", "Seção / Capítulo"],
                    datatype=["str", "str", "str"],
                    label="Fontes e Citações Utilizadas",
                    interactive=False,
                    wrap=True,
                )

        ask_button.click(
            ui_ask,
            inputs=[question_input, sources_input, top_k_input, mode_input],
            outputs=[answer_output, sources_output],
        )
        question_input.submit(
            ui_ask,
            inputs=[question_input, sources_input, top_k_input, mode_input],
            outputs=[answer_output, sources_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
