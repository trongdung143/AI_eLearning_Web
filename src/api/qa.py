from fastapi import APIRouter, Cookie, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Optional, AsyncGenerator
import json
import os
from src.agents.workflow import graph
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    RemoveMessage,
    SystemMessage,
)

# from src.utils.handler import save_upload_file_into_temp
from langgraph.types import Command
from langgraph.graph.message import REMOVE_ALL_MESSAGES

router = APIRouter()


async def generate_qa_stream(
    question: str,
    user_id: str,
    lesson_id: str,
    type_request: str,
    messages: Optional[list[dict]] = None,
) -> AsyncGenerator[str, None]:
    try:
        input_state = None
        config = {"configurable": {"thread_id": lesson_id, "user_id": user_id, "lesson_id": lesson_id}}
        input_state = {
            "messages": [HumanMessage(content=question)],
            "type_request": type_request,
            "task": question,
            "result": None,
            "lecture": None,
            "quiz": None,
            "document_path": None,
        }

        if messages:
            old_messages = []
            for msg in messages:
                if msg.get("sendertype") == "USER":
                    old_messages.append(HumanMessage(content=msg.get("content")))
                elif msg.get("sendertype") == "AI":
                    old_messages.append(AIMessage(content=msg.get("content")))
            graph.update_state(
                config=config,
                values={
                    "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + old_messages
                },
            )

        async for event in graph.astream(
            input=input_state,
            config=config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            subgraph, data_type, chunk = event
            if data_type == "updates":
                if chunk.get("__interrupt__") and not subgraph:
                    for interrupt in chunk["__interrupt__"]:
                        yield f"data: {json.dumps({'type': 'interrupt',
                                                    'response': interrupt.value.get("AIMessage")
                                                }, ensure_ascii=False)}\n\n"
            if data_type == "messages":
                response, meta = chunk
                agent = meta.get("langgraph_node", "unknown")
                if subgraph:
                    agent = subgraph[0].split(":")[0]
                if agent == "writer":
                    yield f"data: {json.dumps({'type': 'chunk','response': response.content, "agent": agent}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_data = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


@router.post("/qa")
async def qa_stream(
    question: str = Form(""),
    user_id: str = Form(""),
    lesson_id: str = Form(""),
    type_request: str = Form(""),
    messages: Optional[list[dict]] = Form(None),
) -> StreamingResponse:
    return StreamingResponse(
        generate_qa_stream(question, user_id, lesson_id, type_request, messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "X-Accel-Buffering": "no",
        },
    )
