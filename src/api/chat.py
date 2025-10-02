from fastapi import APIRouter, Cookie, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Optional, AsyncGenerator
import json
import os
from src.agents.workflow import graph
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, SystemMessage
from src.utils.handler import save_upload_file_into_temp
from langgraph.types import Command
from langgraph.graph.message import REMOVE_ALL_MESSAGES

router = APIRouter()

async def generate_chat_stream(
    message: str,
    conversation_id: str,
    file_path: Optional[str] = None,
    messages: Optional[list[dict]] = None,
) -> AsyncGenerator[str, None]:
    try:
        input_state = None
        config = {"configurable": {"thread_id": conversation_id}}
        if not graph.get_state(config).values:
            input_state = {
                "messages": [
                    HumanMessage(content=message)],
                "human": None,
                "next_agent": "memory",
                "prev_agent": None,
                "thread_id": conversation_id,
                "tasks": None,
                "results": None,
                "assigned_agents": None,
                "file_path": file_path,
            }
        else:
            input_state = {
                "messages": [
                    HumanMessage(content=message)],
                "human": None,
                "next_agent": "memory",
                "assigned_agents": None,
                "file_path": file_path,
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

        interrupt = graph.get_state(config=config).interrupts
        if interrupt:
            input_state = {
                "messages": [HumanMessage(content=message)],
            }
            input_state = Command(resume=message)
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
                yield f"data: {json.dumps({'type': 'chunk',
                                            'response': response.content,
                                            'agent': agent}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_data = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chatbot_stream(
    message: str = Form(""),
    file: Optional[UploadFile] = File(None),
    conversation_id: str = Cookie(None),
    messages: Optional[list[dict]] = Form(None),
) -> StreamingResponse:
    file_path = None
    if file:
        file_path = save_upload_file_into_temp(file, conversation_id)
    return StreamingResponse(
        generate_chat_stream(message, conversation_id, file_path, messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "X-Accel-Buffering": "no",
        },
    )
