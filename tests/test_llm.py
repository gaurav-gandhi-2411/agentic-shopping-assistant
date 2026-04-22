"""
LLM client tests — all hit real Ollama (no mocking).
Requires: ollama serve + llama3.1:8b pulled.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalogue.loader import load_config
from src.llm.client import get_llm_client


@pytest.fixture(scope="module")
def llm():
    return get_llm_client(load_config())


@pytest.mark.requires_ollama
def test_ollama_generate_basic(llm):
    response = llm.generate("Respond with only the word 'OK' and nothing else.")
    assert "ok" in response.lower(), f"Expected 'OK' in response, got: {response!r}"


@pytest.mark.requires_ollama
def test_ollama_chat_with_system(llm):
    response = llm.chat(
        messages=[
            {"role": "system", "content": "You always say 'penguin' in every response."},
            {"role": "user", "content": "hi"},
        ]
    )
    assert "penguin" in response.lower(), f"Expected 'penguin' in response, got: {response!r}"


@pytest.mark.requires_ollama
def test_ollama_stream_yields_chunks(llm):
    chunks = list(llm.generate_stream("Count from 1 to 5, one number per line."))
    assert len(chunks) > 1, f"Expected multiple chunks from streaming, got {len(chunks)}"
    full = "".join(chunks)
    assert len(full) > 0
