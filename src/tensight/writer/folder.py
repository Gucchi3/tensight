# @file folder.py
# @brief The record folder, the run folders, and runs.js (02c EXT-01)
# @author Kohei Yamaguchi, Claude

from collections.abc import Mapping
from datetime        import datetime
from importlib       import resources
from pathlib         import Path

import json
import os
import re
import time

from .line import line_encode



VIEWER_VERSION = "0.5.0"
VIEWER_FILES   = ("tensight.css", "colormaps.js", "tensight_core.js", "tensight.js")
BUNDLE_JS      = ("colormaps.js", "tensight_core.js", "tensight.js")
LOCK_TRIES     = 10
LOCK_WAIT      = 0.1



def folder_prepare(log_dir: Path) -> None:
    '''
    #* Make the record folder and put the viewer files there, when they are missing or older (02a API-01).
    #? Called from RecordWriter. The viewer files come from the package (tensight/viewer).
    #! OSError goes to the caller, which shows MSG-004.
    '''
    
    # Make the folder, and read the version of the viewer that is already there
    log_dir.mkdir(parents=True, exist_ok=True)
    index = log_dir / "index.html"
    found = re.search(r'<meta name="tensight-version" content="([0-9.]+)">', index.read_text(encoding="utf-8")) if index.exists() else None
    if found is not None and tuple(int(n) for n in found.group(1).split(".")) >= tuple(int(n) for n in VIEWER_VERSION.split(".")):
        return None
    
    # Put the page, the scripts and the style of the viewer
    source = resources.files("tensight") / "viewer"
    _folder_write(index, (source / "index.html").read_text(encoding="utf-8"))
    for name in VIEWER_FILES:
        _folder_write(log_dir / "viewer" / name, (source / name).read_text(encoding="utf-8"))
    
    # Put the bundle that the HTML export uses (U-051)
    bundle = {"format": 1, "css": (source / "tensight.css").read_text(encoding="utf-8"), "js": [(source / name).read_text(encoding="utf-8") for name in BUNDLE_JS]}
    _folder_write(log_dir / "viewer" / "bundle.js", line_encode("bundle", bundle))
    return None


def folder_create_run(log_dir: Path, started: datetime, name: str | None) -> Path:
    '''
    #* Make the folder of a run, with a number at the end when the name is taken (02a API-01).
    #! Two processes that make the same name at the same time get different folders, because mkdir fails for one of them.
    '''
    
    # Make the name from the start time and the run name
    base = started.strftime("%Y%m%d-%H%M%S") + ("_" + name if name else "")
    runs = log_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    
    # Try the name, then the name with -2, -3, and so on
    number = 1
    while True:
        folder = runs / (base if number == 1 else base + "-" + str(number))
        try:
            folder.mkdir(exist_ok=False)
        except FileExistsError:
            number += 1
            continue
        (folder / "points").mkdir()
        return folder


def folder_add_run(log_dir: Path, entry: Mapping[str, object]) -> None:
    '''
    #* Add one run to runs.js while holding runs.js.lock (02c).
    #! Raises TimeoutError when the lock stays taken for LOCK_TRIES tries. It removes only a lock that it made.
    '''
    
    # Take the lock, waiting LOCK_WAIT seconds between the tries
    lock   = log_dir / "runs.js.lock"
    handle = None
    for _ in range(LOCK_TRIES):
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(LOCK_WAIT)
    if handle is None:
        raise TimeoutError("runs.js.lock を作れませんでした: " + str(lock))
    
    # Read the runs, add this one, and replace the file
    try:
        path = log_dir / "runs.js"
        runs = []
        if path.exists():
            text = path.read_text(encoding="utf-8")
            try:
                runs = json.loads(text[text.index("(") + 1:text.rindex(");")])["runs"]
            except (ValueError, KeyError, TypeError):
                runs = []
        runs.append(dict(entry))
        _folder_write(path, line_encode("runs", {"format": 1, "runs": runs}))
    finally:
        os.close(handle)
        os.remove(lock)
    return None


def _folder_write(path: Path, text: str) -> None:
    '''
    #* Write a file through a temporary file, so that a reader never sees half of it (02c).
    '''
    
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return None
