# tools/map_reduce.py
import ollama

def get_model_context_limit(model: str, default: int = 8192) -> int:
    """Query Ollama for the model's actual context window."""
    try:
        info = ollama.show(model)
        model_info = info.get("model_info", {})
        # key is family-specific, e.g. "qwen2.context_length"
        for key, value in model_info.items():
            if key.endswith("context_length"):
                return int(value)
    except Exception:
        pass
    return default # fail safe rather than crash