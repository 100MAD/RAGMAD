from app.main import app


def test_operation_ids_match_the_generated_client_names():
    operation_ids = set()
    for path_item in app.openapi()["paths"].values():
        for method, operation in path_item.items():
            if method.startswith("x-") or not isinstance(operation, dict):
                continue
            operation_ids.add(operation["operationId"])

    assert operation_ids == {
        "chats-create",
        "chats-list",
        "chats-get",
        "chats-update",
        "chats-delete",
        "documents-upload",
        "documents-list",
        "documents-download",
        "documents-delete",
        "messages-list",
        "messages-create",
        "evaluation-latest",
        "evaluation-list",
        "evaluation-get",
        "health-check",
    }
