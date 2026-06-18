import pytest
from pathlib import Path
from iico_core.reasoning.task_manager import TaskManager
from iico_core.types import TaskTemplate, TaskStatus, TaskGoal
from iico_core.harness import Harness, HarnessConfig, ProviderConfig

@pytest.fixture
def empty_harness():
    cfg = HarnessConfig(
        provider=ProviderConfig(type="openai", endpoint="http://localhost:11434/v1", model="test"),
        memory_path=Path("dummy_memory"),
        skills_path=Path("dummy_skills")
    )
    return Harness(cfg)

def test_task_manager_validation(empty_harness):
    tm = TaskManager(empty_harness)
    
    tm._tasks = {
        "t1": TaskTemplate(id="t1", description="1", depends_on=[]),
        "t2": TaskTemplate(id="t2", description="2", depends_on=["t1"]),
        "t3": TaskTemplate(id="t3", description="3", depends_on=["t2", "t4"]) # t4 doesn't exist
    }
    
    errors = tm.validate_dependencies()
    assert len(errors) == 1
    assert "no existe" in errors[0]

def test_task_manager_cycles(empty_harness):
    tm = TaskManager(empty_harness)
    
    tm._tasks = {
        "t1": TaskTemplate(id="t1", description="1", depends_on=["t3"]),
        "t2": TaskTemplate(id="t2", description="2", depends_on=["t1"]),
        "t3": TaskTemplate(id="t3", description="3", depends_on=["t2"])
    }
    
    errors = tm.validate_dependencies()
    assert len(errors) == 1
    assert "ciclos" in errors[0]

def test_task_manager_topological_sort(empty_harness):
    tm = TaskManager(empty_harness)
    
    tm._tasks = {
        "t3": TaskTemplate(id="t3", description="3", depends_on=["t1"]),
        "t1": TaskTemplate(id="t1", description="1", depends_on=[]),
        "t2": TaskTemplate(id="t2", description="2", depends_on=["t1", "t3"])
    }
    
    errors = tm.validate_dependencies()
    assert len(errors) == 0
    
    tm._compute_execution_order()
    assert tm.execution_order == ["t1", "t3", "t2"]

def test_task_manager_progress(empty_harness):
    tm = TaskManager(empty_harness)
    
    tm._tasks = {
        "t1": TaskTemplate(id="t1", description="1", depends_on=[]),
        "t2": TaskTemplate(id="t2", description="2", depends_on=["t1"])
    }
    tm._compute_execution_order()
    
    # Should get t1 first
    t = tm.get_next_task()
    assert t.id == "t1"
    
    # Mark t1 as completed
    t.status = TaskStatus.COMPLETED
    
    # Now t2 should be ready
    t2 = tm.get_next_task()
    assert t2.id == "t2"
    
    progress = tm.get_progress()
    assert progress["total"] == 2
    assert progress["completed"] == 1
    assert progress["pending"] == 1
