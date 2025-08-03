from fastapi import HTTPException, status


class MediaException(HTTPException):
    def __init__(
        self,
        detail: str = "Base media exception",
        status_code: int = status.HTTP_409_CONFLICT
    ):
        super().__init__(status_code=status_code, detail=detail)

class WrongFormatDocException(MediaException):
    def __init__(self,):
        super().__init__(
            detail="Wrong document format",
            status_code=status.HTTP_409_CONFLICT
        )

class TextNotFoundException(MediaException):
    def __init__(self,):
        super().__init__(
            detail="Сouldn't read text from pdf file",
            status_code=status.HTTP_409_CONFLICT
        )
