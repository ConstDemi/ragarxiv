# src/eval/_compat.py
# Шим совместимости. ragas 0.4.3 (latest) жёстко импортит
# `langchain_community.chat_models.vertexai`, которого больше нет в
# langchain-community 0.4.x (стек langchain 1.x, нужный судье langchain-anthropic).
# Vertexai мы не используем — подсовываем заглушку ДО импорта ragas, чтобы import не падал.
# Импортировать этот модуль нужно ПЕРЕД любым import ragas.
import importlib.util
import sys
import types

_MOD = "langchain_community.chat_models.vertexai"

if importlib.util.find_spec(_MOD) is None:
    _stub = types.ModuleType(_MOD)

    class ChatVertexAI:  # noqa: N801 — заглушка, никогда не используется
        pass

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules[_MOD] = _stub
    try:
        import langchain_community.chat_models as _cm
        setattr(_cm, "vertexai", _stub)
    except Exception:
        pass
