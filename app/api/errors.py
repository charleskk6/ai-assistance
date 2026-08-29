"""User-facing error text.

Siri callers get a short Cantonese sentence, because whatever we return here is
what Apple's voice reads out. Everyone else gets the English detail.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse

_SPOKEN = {
    "web_search_failed": "今次暫時搵唔到足夠可靠嘅最新資料，你可以遲啲再試。",
    "search_unavailable": "而家連唔到 web search，你可以遲啲再試。",
    "llm_timeout": "個 local model 今次諗得太耐，唔該再問一次。",
    "llm_unavailable": "而家連唔到 local model，記住喺部 Mac 度開咗 Ollama 先。",
    "model_not_found": "部 Mac 未裝到指定嘅 model，要先 pull 咗佢。",
    "llm_empty": "今次答唔到你，唔該再問一次。",
    "llm_error": "個 local model 出咗問題，你可以遲啲再試。",
    "request_timeout": "今次搞得太耐，唔該再問一次。",
}

_DEFAULT_SPOKEN = "今次處理唔到你嘅問題，你可以遲啲再試。"


def error_response(code: str, message: str, source: str, status: int = 503) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": code,
            "message": _SPOKEN.get(code, _DEFAULT_SPOKEN) if source == "siri" else message,
        },
    )
