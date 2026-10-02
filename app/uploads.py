from pathlib import Path
from uuid import uuid4

def allocate_upload_path(directory: Path, filename: str) -> Path:
    basename = Path(filename.replace("\\", "/")).name
    if not basename or basename in {".", ".."}:
        raise ValueError("Некорректное имя файла.")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{uuid4().hex}_{basename}"
