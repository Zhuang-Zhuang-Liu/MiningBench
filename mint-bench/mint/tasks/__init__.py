from .base import Task
from .reasoning import ReasoningTask, MultipleChoiceTask, TheoremqaTask
from .reasoning_da import ReasoningDATask, MiningBenchTask
from .codegen import CodeGenTask, HumanEvalTask, MBPPTask
try:
    from .alfworld import AlfWorldTask
except Exception:
    # Allow non-AlfWorld tasks such as GSM8K to run even when optional
    # AlfWorld/TextWorld dependencies are not available in the local env.
    AlfWorldTask = None
