Set my Render workspace to medspace

## Architecture Rules

### Service Layer
- ALL business logic must go through `MedicalSessionService` - it acts as a proxy/facade
- NEVER call `llm_provider`, `stt_provider`, or `tts_provider` directly from UI layer (gradio_app.py)
- The service layer encapsulates all provider interactions and handles coordination between services
- UI layer should ONLY call methods on `MedicalSessionService`

## Code Quality

### Type Hints
- Use built-in types for type hints instead of typing module equivalents (Python 3.9+ style)
  - Use `list` instead of `List` from typing
  - Use `dict` instead of `Dict` from typing
  - Use `set` instead of `Set` from typing
  - Use `tuple` instead of `Tuple` from typing
  - Use `type` instead of `Type` from typing
- Use `|` for union types instead of `Union` (e.g., `str | None` instead of `Optional[str]`)
- Example:
  ```python
  # Good
  def process_items(items: list[str], config: dict[str, int]) -> list[int] | None:
      pass
  
  # Bad
  from typing import List, Dict, Optional
  def process_items(items: List[str], config: Dict[str, int]) -> Optional[List[int]]:
      pass
  ```

### Linting and Formatting
- ALWAYS use `make lint` and `make format` commands instead of direct `poetry run` commands
- NEVER use `poetry run mypy`, `poetry run black`, `poetry run isort` etc. directly - always use make commands
- ALWAYS run `make lint` to check for linting errors before completing any code changes
- Fix all linting errors before considering the task complete
- Use `make format` to auto-format code, but always verify with `make lint` afterwards
- **DO NOT** lint or format non-code files (like `.md`, `.txt`, `.yaml`, `.json`) or configuration files (like those in `config/`). Only apply linting and formatting to Python source code.

### Using make commands with specific files
Both `make lint` and `make format` accept a `FILES` parameter for checking/formatting specific files or directories:
```bash
# Format specific files
make format FILES="app/ui/gradio_app.py app/services/session.py"

# Lint specific directory
make lint FILES="app/services/"

# Default behavior (entire project)
make lint
make format
```

**Note:** When using `make lint FILES="..."` with specific files, mypy may still report errors from dependency files (imports) even though you're only checking specific files. This is expected behavior as mypy performs type checking across the entire import graph.