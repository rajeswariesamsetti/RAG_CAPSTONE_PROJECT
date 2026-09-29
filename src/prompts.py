"""
Prompt templates for the RAG Knowledge Assistant.
Defines strict system instructions and context-grounded query templates to prevent hallucination.
"""

from llama_index.core.prompts import PromptTemplate

# Strict System Prompt defining the assistant's boundaries and hallucination control
SYSTEM_PROMPT = (
    "You are a helpful knowledge assistant.\n"
    "Answer ONLY using the provided context.\n"
    "Do not use outside knowledge.\n"
    "If the answer is not contained in the provided context, respond exactly:\n"
    "I don't have enough information to answer that.\n"
    "Do not invent facts.\n"
    "When possible, provide a concise answer and identify the relevant source."
)

# Text QA Template used by LlamaIndex for synthesis
STRICT_RAG_PROMPT_TEMPLATE = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question. If the answer cannot be found in the context above, "
    "respond exactly with: I don't have enough information to answer that.\n"
    "Query: {query_str}\n"
    "Answer: "
)

QA_PROMPT = PromptTemplate(STRICT_RAG_PROMPT_TEMPLATE)

# Refine template (for multi-chunk synthesis if needed)
REFINE_PROMPT_TEMPLATE = (
    "The original query is: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better answer the query. "
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)

REFINE_PROMPT = PromptTemplate(REFINE_PROMPT_TEMPLATE)
