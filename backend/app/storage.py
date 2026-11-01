import uuid
from pathlib import Path

from app.config import Settings


class UploadStorage:
    def __init__(self, settings: Settings) -> None:
        self.local_dir = Path(settings.upload_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def save_local(self, filename: str, data: bytes) -> str:
        path = self.local_dir / f"{uuid.uuid4()}_{Path(filename).name}"
        path.write_bytes(data)
        return str(path)
