# rag_pipeline.py
import torch
import gc
import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Дефолтный системный промпт — используется, если не передан явно (например, из ноутбуков).
# Для API источник правды — config.SYSTEM_PROMPT, он передаётся в конструктор из main.py.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful scientific assistant with knowledge base of NLP arxiv paper for 2025 year. "
    "Use ONLY the provided context to answer the user's question. "
    "If the context doesn't contain enough information, say so explicitly"
)


class ScienceRAG:
    def __init__(self, 
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333,
                 collection_name: str = "nlp2025_chunks",
                 embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 llm_model: str = "Qwen/Qwen2.5-3B-Instruct",
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 max_new_tokens: int = 2000,
                 max_input_tokens: int = 4096,
                 retrieve_k: int = 30,
                 context_k: int = 10,
                 max_papers: int = 8):
        
        self.collection_name = collection_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.retrieve_k = retrieve_k
        self.context_k = context_k
        self.max_papers = max_papers
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
        
        # 2. Загрузка Encoder (на self.device — на 8GB GPU влезает рядом с 4-bit LLM; fp32 = как на CPU)
        logger.info(f"Loading retriever: {embed_model}...")
        self.encoder = SentenceTransformer(
            embed_model,
            trust_remote_code=True,
            device=self.device
        )
        logger.info(f"Retriever loaded on {self.device.upper()}")
        
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

    def _retrieve(self, query: str) -> List[Dict]:
        """
        Поиск релевантных документов в векторной базе.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Список payload'ов найденных документов
        """
        try:
            # Энкодинг запроса (на CPU)
            query_vector = self.encoder.encode(
                query, 
                prompt_name="query",  # query-инструкция Qwen3 (документы кодировались без неё)
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Поиск в Qdrant
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=self.retrieve_k,
                with_payload=True,
                with_vectors=False
            )
            
            return [
                {**(point.payload or {}), "score": point.score}
                for point in search_result.points
            ]
        
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

    def _group_sources(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """
        Группировка чанков по статье (doc_id) → список PaperSource.
        Статьи упорядочены по лучшему (max) score; внутри статьи чанки —
        по убыванию score (из Qdrant чанки приходят уже отсортированными).
        Возвращается не более self.max_papers статей.
        """
        papers: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "")
            if doc_id not in papers:
                papers[doc_id] = {
                    "doc_id": doc_id,
                    "title": chunk.get("title", ""),
                    "authors": chunk.get("authors", ""),
                    "published": chunk.get("published", ""),
                    "url": f"https://arxiv.org/abs/{doc_id}" if doc_id else "",
                    "score": chunk.get("score", 0.0),  # первый встреченный чанк — лучший
                    "chunks": [],
                }
                order.append(doc_id)
            papers[doc_id]["chunks"].append({
                "text": chunk.get("text", ""),
                "score": chunk.get("score", 0.0),
                "chunk_index": chunk.get("chunk_index", 0),
                "header_1": chunk.get("header_1", ""),
                "header_2": chunk.get("header_2", ""),
                "header_3": chunk.get("header_3", ""),
            })

        return [papers[doc_id] for doc_id in order][: self.max_papers]

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Только ретрив: статьи с релевантными пассажами, без генерации LLM."""
        chunks = self._retrieve(query)
        return self._group_sources(chunks)

    def answer(self, query: str) -> Dict[str, Any]:
        """
        Главный метод: поиск + генерация ответа.
        
        Args:
            query: Вопрос пользователя
            
        Returns:
            Словарь с ключами 'answer', 'sources' и 'context_chunks'
            (context_chunks — топ-context_k чанков, реально поданных в LLM)
        """
        
        # 1. Retrieval
        retrieved_chunks = self._retrieve(query)
        
        if not retrieved_chunks:
            logger.warning("No documents found")
            return {
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "sources": [],
                "context_chunks": []
            }
        
        # 2. Группировка источников по статьям (для фронтенда)
        sources = self._group_sources(retrieved_chunks)

        # 3. Формирование контекста (топ-CONTEXT_K чанков, чтобы не раздувать промпт)
        context_chunks = retrieved_chunks[:self.context_k]   # ровно то, что увидит LLM
        context = self._format_context(context_chunks)
        
        # 4. Подготовка промпта
        messages = [
            {"role": "system", "content": self.system_prompt},
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
                max_length=self.max_input_tokens
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
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
                "sources": sources,
                "context_chunks": context_chunks
            }
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": sources,
                "context_chunks": context_chunks
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