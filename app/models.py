
from pydantic import BaseModel
from typing import List, Tuple, Optional

class AskRequest(BaseModel):
    """Defines the structure for a request to the /api/ask endpoint."""
    q: str
    # Chat history is now part of the request model.
    # It's a list of (human_message, ai_message) tuples.
    chat_history: Optional[List[Tuple[str, str]]] = None

