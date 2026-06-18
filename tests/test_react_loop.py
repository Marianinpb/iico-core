import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from iico_core.reasoning.react_loop import ReActLoop
from iico_core.types import (
    TaskTemplate, TaskGoal, TaskStatus, 
    LLMResponse, LLMToolCall, HarnessEventType
)
from iico_core.harness import Harness, HarnessConfig, ProviderConfig

@pytest.fixture
def harness_with_mock_llm():
    cfg = HarnessConfig(
        provider=ProviderConfig(type="openai", endpoint="http://localhost:11434/v1", model="test"),
        memory_path=Path("dummy_memory"),
        skills_path=Path("dummy_skills")
    )
    harness = Harness(cfg)
    harness.llm = AsyncMock()
    harness._skill_registry = MagicMock()
    harness._skill_registry.get_tool_descriptions.return_value = []
    harness._skill_registry.format_for_prompt.return_value = "Mocked skills text"
    
    # Mock execute_skill directly returning a successful result
    from iico_core.bridge.shell import ToolResult
    harness.execute_skill = MagicMock(return_value=ToolResult("test_skill", "success", 0))
    return harness

@pytest.mark.asyncio
async def test_react_loop_execute_simple(harness_with_mock_llm):
    loop = ReActLoop(harness_with_mock_llm)
    
    # First response returns a tool call, second returns final string
    harness_with_mock_llm.llm.chat_with_tools.side_effect = [
        LLMResponse(
            content="", 
            tool_calls=[LLMToolCall(call_id="call_1", name="test_skill", args={})],
            finish_reason="tool_calls"
        ),
        LLMResponse(
            content="Done", 
            tool_calls=[],
            finish_reason="stop"
        )
    ]
    
    events = []
    async for event in loop.execute_simple("Hello"):
        events.append(event)
    
    # Assert events
    event_types = [e.type for e in events]
    assert HarnessEventType.THINKING in event_types
    assert HarnessEventType.SKILL_START in event_types
    assert HarnessEventType.SKILL_DONE in event_types
    assert HarnessEventType.DONE in event_types
    
    assert harness_with_mock_llm.execute_skill.call_count == 1
    assert harness_with_mock_llm.execute_skill.call_args[0][0] == "test_skill"

@pytest.mark.asyncio
async def test_react_loop_execute_task(harness_with_mock_llm):
    loop = ReActLoop(harness_with_mock_llm)
    
    # Mocking task
    task = TaskTemplate(
        id="t1", 
        description="Do something", 
        goals=[TaskGoal(description="Goal 1", verification_skill="verify_test")]
    )
    
    # LLM returns string directly
    harness_with_mock_llm.llm.chat_with_tools.side_effect = [
        LLMResponse(
            content="Task is done", 
            tool_calls=[],
            finish_reason="stop"
        )
    ]
    
    events = []
    async for event in loop.execute_task(task, []):
        events.append(event)
    
    # Verify events
    event_types = [e.type for e in events]
    assert HarnessEventType.TASK_STARTED in event_types
    assert HarnessEventType.GOAL_VERIFIED in event_types
    assert HarnessEventType.TASK_COMPLETED in event_types
    
    assert task.status == TaskStatus.COMPLETED
    assert task.result_summary == "Task is done"
    
    # Verification skill was called
    assert harness_with_mock_llm.execute_skill.call_count == 1
    assert harness_with_mock_llm.execute_skill.call_args[0][0] == "verify_test"

