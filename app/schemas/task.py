from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class TaskCreate(BaseModel):
    text_content: str

class TaskResponse(BaseModel):
    id: str
    status: str
    created_at: str
    text_content: str
    cleaned_text: Optional[str] = None
    pii_entities: Optional[Dict[str, int]] = None
