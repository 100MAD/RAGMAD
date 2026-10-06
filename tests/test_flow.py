import socket
import time
import uuid

import pytest
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen
from llama_index.core.llms import CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


pytestmark = pytest.mark.skipif(
    not all(_port_open(port) for port in (5433, 6333, 9000)),
    reason="Postgres, Qdrant, and MinIO must be running (docker compose up -d)",
)


class StubLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=256,
            is_chat_model=True,
            model_name="stub",
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        return CompletionResponse(text="Answer from the uploaded document.")

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponseGen:
        text = "Answer from the uploaded document."

        def generate() -> CompletionResponseGen:
            yield CompletionResponse(text=text, delta=text)

        return generate()


@pytest.fixture()
def client():
    from alembic import command
    from alembic.config import Config
    from fastapi.testclient import TestClient

    import app.rag.components as components

    command.upgrade(Config("alembic.ini"), "head")
    components._llm = StubLLM()
    with TestClient(app_import()) as test_client:
        yield test_client
    components._llm = None
    components.reset_components()


def app_import():
    from app.main import app

    return app


def _upload(client, chat_id: str, filename: str, text: str) -> dict:
    response = client.post(
        f"/chats/{chat_id}/documents",
        files=[("files", (filename, text.encode(), "text/markdown"))],
    )
    assert response.status_code == 202, response.text
    document = None
    for _ in range(120):
        documents = client.get(f"/chats/{chat_id}/documents").json()
        document = next(item for item in documents if item["filename"] == filename)
        if document["status"] != "processing":
            break
        time.sleep(0.5)
    assert document is not None
    assert document["status"] == "ready", document
    assert document["chunk_count"] >= 1
    return document


def test_chats_keep_separate_documents_and_restore_messages(client):
    first = client.post("/chats", json={}).json()
    second = client.post("/chats", json={}).json()
    river = f"river-{uuid.uuid4().hex}"
    hill = f"hill-{uuid.uuid4().hex}"

    _upload(
        client,
        first["id"],
        "river.md",
        f"# River club\n\nThe mascot of the river club is a silver heron. Code {river}.\n",
    )
    _upload(
        client,
        second["id"],
        "hill.md",
        f"# Hill club\n\nThe mascot of the hill club is a copper fox. Code {hill}.\n",
    )

    blocked = client.post("/chats", json={})
    empty_id = blocked.json()["id"]
    rejected = client.post(f"/chats/{empty_id}/messages", json={"content": "Hello"})
    assert rejected.status_code == 409

    answer = client.post(
        f"/chats/{first['id']}/messages",
        json={"content": "What is the river club mascot?"},
    )
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["role"] == "assistant"
    assert body["content"]
    assert any("silver heron" in source["text"].lower() for source in body["sources"])
    assert all(source["filename"] == "river.md" for source in body["sources"])
    assert all(hill not in source["text"] for source in body["sources"])

    listing = client.get("/chats")
    restored = next(chat for chat in listing.json() if chat["id"] == first["id"])
    assert restored["title"] == "What is the river club mascot?"

    messages = client.get(f"/chats/{first['id']}/messages")
    assert [item["role"] for item in messages.json()] == ["user", "assistant"]

    other = client.get(f"/chats/{second['id']}/messages")
    assert other.json() == []

    second_answer = client.post(
        f"/chats/{second['id']}/messages",
        json={"content": "What is the hill club mascot?"},
    )
    assert second_answer.status_code == 200, second_answer.text
    assert any("copper fox" in source["text"].lower() for source in second_answer.json()["sources"])
    assert all(river not in source["text"] for source in second_answer.json()["sources"])

    document_id = client.get(f"/chats/{first['id']}/documents").json()[0]["id"]
    downloaded = client.get(f"/documents/{document_id}/download")
    assert downloaded.status_code == 200
    assert b"silver heron" in downloaded.content

    assert client.delete(f"/chats/{first['id']}").status_code == 204
    assert client.delete(f"/chats/{second['id']}").status_code == 204
    assert client.delete(f"/chats/{empty_id}").status_code == 204
    assert client.get(f"/chats/{first['id']}").status_code == 404
