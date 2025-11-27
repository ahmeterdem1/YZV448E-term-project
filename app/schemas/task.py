from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TaskCreate(BaseModel):
    text_content: str

class TaskResponse(BaseModel):
    id: str
    status: str
    created_at: str
    text_content: str
