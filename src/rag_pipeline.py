"""
RAG Pipeline module orchestrating the end-to-end question answering workflow:
Question -> Embedding -> ChromaDB Similarity Retrieval -> Context Assembly -> LLM Grounded Answer -> Citations
"""

from typing import Dict, Any, List, Optional
from llama_index.core import VectorStoreIndex
from src.config import (
    DEFAULT_SIMILARITY_TOP_K,
    OUT_OF_CONTEXT_RESPONSE
)
from src.retriever import retrieve_chunks
from src.prompts import STRICT_RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
from src.llm import get_llm
from utils.logging_utils import get_logger

logger = get_logger("rag_pipeline")

class RAGPipeline:
    """
    Production RAG Pipeline managing query retrieval, context assembly,
    grounded LLM inference, and source citation extraction.
    """

    def __init__(
        self,
        index: Optional[VectorStoreIndex] = None,
        llm=None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        provider: str = "Groq",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self._llm = llm

    @property
    def llm(self):
        """Lazy-loads or returns configured LLM instance."""
        if self._llm is None:
            self._llm = get_llm(
                provider=self.provider,
                api_key=self.api_key,
                model_name=self.model_name
            )
        return self._llm

    def set_index(self, index: VectorStoreIndex):
        """Updates the active VectorStoreIndex."""
        self.index = index

    def set_provider(self, provider: str, api_key: Optional[str] = None):
        """Switches the LLM provider (e.g. Groq vs Ollama)."""
        self.provider = provider
        self.api_key = api_key
        self._llm = get_llm(provider=self.provider, api_key=self.api_key)

    def answer_question(
        self,
        question: str,
        similarity_top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline for a given user query.

        Pipeline Stages:
          1. Validation: Ensure non-empty query and available index.
          2. Retrieval: Convert question to embedding, retrieve top-k chunks from ChromaDB.
          3. Context Construction: Format retrieved chunks with source markers.
          4. Prompt Assembly: Inject into strict QA prompt template.
          5. LLM Synthesis: Request grounded completion from LLM.
          6. Post-processing: Enforce strict rejection for unsupported questions.
          7. Source Citations: Package source filename, page, similarity score, and preview.

        Returns:
            Dict[str, Any]:
                {
                    "answer": str,
                    "sources": [
                        {
                            "filename": str,
                            "page": Union[int, str, None],
                            "score": float,
                            "text_preview": str
                        }
                    ],
                    "raw_chunks": List[Dict[str, Any]]
                }
        """
        clean_query = question.strip() if question else ""
        if not clean_query:
            return {
                "answer": "Please ask a question.",
                "sources": [],
                "raw_chunks": []
            }

        if self.index is None:
            logger.warning("Query rejected: No vector index is loaded.")
            return {
                "answer": "No knowledge base loaded. Please upload documents and build the index first.",
                "sources": [],
                "raw_chunks": []
            }

        k = similarity_top_k or self.similarity_top_k

        # Stage 2: Retrieval
        retrieved_chunks = retrieve_chunks(
            query=clean_query,
            index=self.index,
            similarity_top_k=k
        )

        if not retrieved_chunks:
            logger.info("No relevant chunks retrieved from vector store.")
            return {
                "answer": OUT_OF_CONTEXT_RESPONSE,
                "sources": [],
                "raw_chunks": []
            }

        # Stage 3: Context Construction
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            page_info = f"Page {chunk['page']}" if chunk.get("page") else "Page N/A"
            source_info = f"Document: {chunk['filename']} | {page_info}"
            context_parts.append(
                f"[Source {i}: {source_info}]\n{chunk['text']}"
            )
        context_str = "\n\n".join(context_parts)

        # Stage 4: Strict Prompt Construction
        formatted_prompt = STRICT_RAG_PROMPT_TEMPLATE.format(
            context_str=context_str,
            query_str=clean_query
        )

        # Stage 5: Grounded LLM Inference
        try:
            llm_instance = self.llm
            response = llm_instance.complete(formatted_prompt)
            raw_answer = str(response).strip()
        except Exception as e:
            err_str = str(e)
            # Detect invalid/expired API key errors
            if "401" in err_str or "403" in err_str or "Forbidden" in err_str or "Unauthorized" in err_str:
                logger.error(f"Groq API authentication failed: {err_str}")
                raise RuntimeError(
                    "Groq API key is invalid or expired. "
                    "Please get a new key from https://console.groq.com/keys "
                    "and update it in the sidebar or .env file."
                )
            elif "model_not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                logger.warning(f"Target model rejected by Groq: {err_str}. Attempting dynamic fallback...")
                from src.llm import fetch_available_groq_models
                from src.config import FALLBACK_GROQ_MODELS
                avail = fetch_available_groq_models(self.api_key)
                current_model = getattr(llm_instance, "model", "")
                # Try each available model until one works
                tried = {current_model}
                for fallback_model in avail:
                    if fallback_model in tried:
                        continue
                    tried.add(fallback_model)
                    try:
                        logger.info(f"Retrying with fallback model: '{fallback_model}'")
                        fallback_llm = get_llm(
                            provider=self.provider,
                            api_key=self.api_key,
                            model_name=fallback_model
                        )
                        response = fallback_llm.complete(formatted_prompt)
                        raw_answer = str(response).strip()
                        break
                    except Exception as fb_err:
                        logger.warning(f"Fallback model '{fallback_model}' also failed: {fb_err}")
                        continue
                else:
                    raise RuntimeError(
                        f"LLM generation failed: {err_str}\n"
                        "None of the available models could process the request. "
                        "Please verify your Groq API key at https://console.groq.com/keys"
                    )
            else:
                logger.error(f"Error during LLM completion: {e}")
                raise RuntimeError(f"LLM generation failed: {str(e)}")

        # Stage 6: Hallucination and Fallback Standardization
        lower_ans = raw_answer.lower()
        if (
            "don't have enough information" in lower_ans
            or "do not have enough information" in lower_ans
            or "not enough information" in lower_ans
            or "cannot be answered" in lower_ans
            or "context does not provide" in lower_ans
            or "not mentioned in the context" in lower_ans
        ):
            final_answer = OUT_OF_CONTEXT_RESPONSE
        else:
            final_answer = raw_answer

        # Stage 7: Source Citation Packaging
        sources = []
        for chunk in retrieved_chunks:
            sources.append({
                "filename": chunk["filename"],
                "page": chunk["page"],
                "score": chunk["score"],
                "text_preview": (
                    chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                )
            })

        return {
            "answer": final_answer,
            "sources": sources,
            "raw_chunks": retrieved_chunks
        }


def answer_question(
    question: str,
    index: VectorStoreIndex,
    provider: str = "Groq",
    api_key: Optional[str] = None,
    similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K
) -> Dict[str, Any]:
    """
    Convenience function to run a single query through the RAG pipeline.
    """
    pipeline = RAGPipeline(
        index=index,
        similarity_top_k=similarity_top_k,
        provider=provider,
        api_key=api_key
    )
    return pipeline.answer_question(question)
