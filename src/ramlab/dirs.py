from pathlib import Path
import ramlab


dir_root = Path(ramlab.__file__).parent.parent
dir_project = dir_root.parent
dir_data = dir_project / "data"
