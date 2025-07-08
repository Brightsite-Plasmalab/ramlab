from pathlib import Path
import src.ramlab


dir_root = Path(src.ramlab.__file__).parent.parent
dir_project = dir_root.parent
dir_data = dir_project / "data"
