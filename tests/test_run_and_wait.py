import os
import asyncio
import pytest
from openai import APIConnectionError

# Prevent network calls during import
os.environ.setdefault("ASSISTANT_ID", "test")

import monolith


def test_run_and_wait_openai_failure(monkeypatch):
    # avoid waiting during retries
    async def fast_sleep(_):
        pass
    monkeypatch.setattr(monolith.asyncio, "sleep", fast_sleep)

    def fail_create(*args, **kwargs):
        raise APIConnectionError(request=None)

    monkeypatch.setattr(monolith.client.beta.threads.runs, "create", fail_create)

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(monolith.run_and_wait("thread"))

    assert "network error" in str(exc.value)
