# rag_pipeline.py
import torch
import time
import gc
import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScienceRAG:
    def __init__(self, 
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333,
                 collection_name: str = "nlp2025_chunks",
                 embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 llm_model: str = "Qwen/Qwen2.5-3B-Instruct"):
        
        self.collection_name = collection_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing RAG on device: {self.device.upper()}")

        # 1. Подключение к Qdrant
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Проверка существования коллекции
        available_collections = {c.name for c in self.client.get_collections().collections}
        if collection_name not in available_collections:
            raise ValueError(
                f"Collection '{collection_name}' not found. "
                f"Available: {list(available_collections)}"
            )
        logger.info(f"Connected to Qdrant collection: {collection_name}")
        
        # 2. Загрузка Encoder
        logger.info(f"Loading retriever: {embed_model}...")
        self.encoder = SentenceTransformer(
            embed_model, 
            trust_remote_code=True, 
            device="cpu"
        )
        logger.info("Retriever loaded on CPU")
        
        # Очистка перед загрузкой LLM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Загрузка LLM
        logger.info(f"Loading LLM: {llm_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        
        # Квантизация только для CUDA
        if self.device == "cuda":
            logger.info("Using 4-bit quantization (NF4)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="sdpa",
                low_cpu_mem_usage=True
            )
        else:
            # CPU fallback без квантизации
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
        
        logger.info(f"LLM loaded on {self.device.upper()}")
        
        # Логирование VRAM (только для CUDA)
        if self.device == "cuda":
            vram_allocated = torch.cuda.memory_allocated() / 1e9
            vram_reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"VRAM allocated: {vram_allocated:.2f} GB")
            logger.info(f"VRAM reserved: {vram_reserved:.2f} GB")
        
        logger.info("RAG system is ready.\n")

    def _retrieve(self, query: str, top_k: int) -> List[Dict]:
        """
        Поиск релевантных документов в векторной базе.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            
        Returns:
            Список payload'ов найденных документов
        """
        try:
            # Энкодинг запроса (на CPU)
            query_vector = self.encoder.encode(
                query, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Поиск в Qdrant
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            return [point.payload for point in search_result.points]
        
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Форматирование найденных документов в контекст для LLM.
        
        Args:
            chunks: Список payload'ов документов
            
        Returns:
            Форматированный текст контекста
        """
        if not chunks:
            return "No relevant documents found."
        
        formatted_text = ""
        for i, chunk in enumerate(chunks):
            # Безопасное извлечение полей
            title = chunk.get('title', 'Unknown Title')
            text = chunk.get('text') or chunk.get('abstract') or chunk.get('content', 'No content')
            
            formatted_text += (
                f"Document [{i+1}]\n"
                f"Title: {title}\n"
                f"Content: {text}\n\n"
            )
        
        return formatted_text

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict[str, str]]:
        """
        Извлечение источников для фронтенда.
        
        Args:
            chunks: Список payload'ов
            
        Returns:
            Список словарей с title и text
        """
        sources = []
        for chunk in chunks:
            text = chunk.get("text") or chunk.get("abstract") or chunk.get("content", "")
            
            sources.append({
                "text": text
            })
        
        return sources

    def answer(self, query: str, top_k: int) -> Dict[str, any]:
        """
        Главный метод: поиск + генерация ответа.
        
        Args:
            query: Вопрос пользователя
            top_k: Количество источников
            
        Returns:
            Словарь с ключами 'answer' и 'sources'
        """
        
        # 1. Retrieval
        retrieved_chunks = self._retrieve(query, top_k)
        
        if not retrieved_chunks:
            logger.warning("No documents found")
            return {
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "sources": []
            }
        
        # 2. Подготовка источников для фронтенда
        sources = self._extract_sources(retrieved_chunks)
        
        # 3. Формирование контекста
        context = self._format_context(retrieved_chunks)
        
        # 4. Подготовка промпта
        system_prompt = (
            "You are a helpful scientific assistant with knowledge base of NLP arxiv paper for 2025 year."
            "Use ONLY the provided context to answer the user's question."
            "If the context doesn't contain enough information, say so explicitly"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        # 5. Генерация ответа
        try:
            text_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer(
                [text_input], 
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=2000,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Декодирование только новых токенов
            response_text = self.tokenizer.batch_decode(
                [output_ids[len(input_ids):] 
                 for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)],
                skip_special_tokens=True
            )[0]
            
            # Очистка GPU памяти после генерации
            if self.device == "cuda":
                del model_inputs, generated_ids
                torch.cuda.empty_cache()
            
            return {
                "answer": response_text.strip(),
                "sources": sources
            }
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": sources
            }

    def cleanup(self):
        """Очистка ресурсов (опционально)"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'encoder'):
            del self.encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Resources cleaned up")