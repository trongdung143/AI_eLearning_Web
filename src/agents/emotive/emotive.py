import os
import shutil
from typing import Sequence

import fitz
from langchain_core.messages import HumanMessage
from langchain_core.tools.base import BaseTool

import src.agents.emotive.prompt
from src.agents.base import BaseAgent
from src.agents.state import State




class emotiveAgent(BaseAgent):
    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        super().__init__(
            agent_name="emotive",
            tools=tools,
            model=None,
        )

        self._prompt = src.agents.emotive.prompt.prompt
        self._chain = self._prompt | self._model
        self._set_subgraph()

    def _extract_text_from_pdf(self, state: State) -> str:
        save_dir = "src/data/temp"
        data_path = os.path.join(save_dir, state.get("thread_id"), state.get("file_path"))
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"PDF not found at: {data_path}")

        with fitz.open(data_path) as doc:
            text = "".join(page.get_text() for page in doc)
        shutil.rmtree(os.path.join(save_dir, state.get("thread_id")))
        print("extracting text from pdf")
        return text

    async def process(self, state: State) -> State:
        tasks = state.get("assigned_agents").get(self._agent_name)
        task = ""
        for t in tasks:
            task = task + f"{t}\n"
        result = None
        try:
            content = self._extract_text_from_pdf(state)
            response = await self._chain.ainvoke(
                {"task": [HumanMessage(content=f"### Nội Dung\n{content}\n\n### Yêu Cầu\n{task}")]}
            )
            result = response.content
            current_tasks, current_results, assigned_agents = self.update_work(state, task, result)
            state.update(
                human=False,
                next_agent=None,
                prev_agent=self._agent_name,
                tasks=current_tasks,
                results=current_results,
                assigned_agents=assigned_agents,
            )
            print("emotive")
        except Exception as e:
            print("ERROR ", self._agent_name)
        return state
