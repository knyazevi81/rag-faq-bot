from pydantic import BaseModel

from enum import Enum

class MediaType(str, Enum):
    pdf: str = "pdf"

