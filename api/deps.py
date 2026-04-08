from functools import lru_cache


@lru_cache(maxsize=1)
def get_model():
    from app.utils.inference import load_model
    return load_model()
