# @file run.py
# @brief Start, step, and close a run (02a API-01, API-03, API-04, API-05, API-08)
# @author Kohei Yamaguchi, Claude

from collections.abc import Mapping, Sequence
from datetime        import datetime
from pathlib         import Path

import atexit
import re

from .core      import message_reset, message_warn
from .run_state import RunState
from .stats     import StepCollector, pattern_parse
from .types     import MapSpec, PointKind, RunSettings
from .writer    import RecordWriter



RUN_NAME    = re.compile(r"[A-Za-z0-9_-]{1,64}")
DEFAULT_DIR = "tensight_runs"
KINDS       = ("input", "output", "output_grad", "weight", "bias", "grad", "update")

_current: RunState | None = None



def run_start(log_dir: str | Path | None = None, name: str | None = None, points: Sequence[str] | None = None, kinds: Sequence[PointKind] = ("output",), interval: int = 100, bins: int = 64, sample_size: int = 16, boundary_width: float = 0.05,
              track_elements: Mapping[str, Sequence[int]] | None = None, compare_steps: Sequence[int] | None = None, save_full: Mapping[str, Sequence[int]] | None = None, maps: Mapping[str, str] | Sequence[str] | None = None) -> Path:
    '''
    #* Check the settings, and start a run (02a API-01).
    #! A wrong argument raises TypeError or ValueError before anything starts.
    '''
    
    global _current
    
    # Check the types of the arguments
    _run_check("log_dir", log_dir, log_dir is None or isinstance(log_dir, (str, Path)), "str か Path か None", TypeError)
    _run_check("name", name, name is None or isinstance(name, str), "str か None", TypeError)
    _run_check("points", points, points is None or (isinstance(points, Sequence) and not isinstance(points, str) and all(isinstance(point, str) for point in points)), "str の並びか None", TypeError)
    _run_check("kinds", kinds, isinstance(kinds, Sequence) and not isinstance(kinds, str) and all(isinstance(kind, str) for kind in kinds), "str の並び", TypeError)
    for key, number in (("interval", interval), ("bins", bins), ("sample_size", sample_size)):
        _run_check(key, number, isinstance(number, int) and not isinstance(number, bool), "int", TypeError)
    _run_check("boundary_width", boundary_width, isinstance(boundary_width, (int, float)) and not isinstance(boundary_width, bool), "int か float", TypeError)
    for key, table in (("track_elements", track_elements), ("save_full", save_full)):
        rows = table.items() if isinstance(table, Mapping) else []
        good = all(isinstance(point, str) and isinstance(numbers, Sequence) and not isinstance(numbers, str) and all(type(n) is int for n in numbers) for point, numbers in rows)
        _run_check(key, table, table is None or (isinstance(table, Mapping) and good), "str から int の並びへの Mapping か None", TypeError)
    _run_check("compare_steps", compare_steps, compare_steps is None or (isinstance(compare_steps, Sequence) and not isinstance(compare_steps, str) and all(type(n) is int for n in compare_steps)), "int の並びか None", TypeError)
    by_name = isinstance(maps, Mapping) and all(isinstance(key, str) and isinstance(spec, str) for key, spec in maps.items())
    by_list = isinstance(maps, Sequence) and not isinstance(maps, str) and all(isinstance(key, str) for key in maps)
    _run_check("maps", maps, maps is None or by_name or by_list, "str から str への Mapping か、str の並びか None", TypeError)
    
    # Check the values of the arguments
    unknown = [kind for kind in kinds if kind not in KINDS]
    if unknown:
        raise ValueError("引数 kinds に知らない種類 " + str(unknown[0]) + " があります。使える種類: " + "、".join(KINDS))
    _run_check("name", name, name is None or RUN_NAME.fullmatch(name) is not None, "1 文字以上 64 文字以下の、英字・数字・-・_ だけの文字列", ValueError)
    _run_check("points", points, points is None or all(points), "空でない文字列の並び", ValueError)
    _run_check("kinds", kinds, len(kinds) > 0, "1 つ以上の種類", ValueError)
    _run_check("interval", interval, interval >= 1, "1 以上", ValueError)
    _run_check("bins", bins, 1 <= bins <= 4096, "1 以上 4096 以下", ValueError)
    _run_check("sample_size", sample_size, 0 <= sample_size <= 4096, "0 以上 4096 以下", ValueError)
    _run_check("boundary_width", boundary_width, 0 < boundary_width <= 0.5, "0 より大きく 0.5 以下", ValueError)
    _run_check("track_elements", track_elements, track_elements is None or all(n >= 0 for numbers in track_elements.values() for n in numbers), "0 以上のインデックス", ValueError)
    _run_check("compare_steps", compare_steps, compare_steps is None or all(n >= 1 for n in compare_steps), "1 以上のステップ", ValueError)
    _run_check("save_full", save_full, save_full is None or all(n >= 1 for numbers in save_full.values() for n in numbers), "1 以上のステップ", ValueError)
    _run_check("maps", maps, maps is None or all(key for key in maps), "空でないパターン", ValueError)
    
    # Read the map patterns (IF-11); a sequence of patterns means "auto" for each
    specs: tuple[tuple[str, MapSpec], ...] = ()
    if isinstance(maps, Mapping):
        specs = tuple((key, pattern_parse(spec)) for key, spec in maps.items())
    elif maps is not None:
        specs = tuple((key, "auto") for key in maps)
    
    # Put the settings together
    settings = RunSettings(Path(log_dir) if log_dir is not None else Path.cwd() / DEFAULT_DIR, name, tuple(points) if points is not None else None, tuple(dict.fromkeys(kinds)), interval, bins, sample_size, float(boundary_width),
                           {key: tuple(numbers) for key, numbers in track_elements.items()} if track_elements is not None else None, frozenset(compare_steps or ()), {key: frozenset(numbers) for key, numbers in (save_full or {}).items()}, specs)
    
    # Close the run that is going on, and forget the warnings of the last run
    closed = _current is not None
    if closed:
        run_close()
    message_reset()
    if closed:
        message_warn("MSG-007")
    
    # Start the run: the folders, the collector, and the state
    writer   = RecordWriter(settings.log_dir, name, settings, datetime.now().astimezone())
    _current = RunState(settings, writer, StepCollector(settings, writer))
    return writer.folder if writer.folder is not None else writer.planned


def run_step() -> None:
    '''
    #* Finish one step; a recorded step hands its records to the writer (02a API-03).
    '''
    
    if _current is None:
        message_warn("MSG-006", function="run_step")
        return None
    _current.advance()
    return None


def run_should_record() -> bool:
    '''
    #* Return whether the step that runs next is recorded, without waiting for the GPU (02a API-04).
    '''
    
    return _current.recording if _current is not None else False


def run_record_next() -> None:
    '''
    #* Record the step that runs next, whatever the interval is (02a API-05).
    #! Call it before the forward of that step.
    '''
    
    if _current is None:
        message_warn("MSG-006", function="run_record_next")
        return None
    _current.mark_next()
    return None


def run_close() -> Path | None:
    '''
    #* Write everything that is waiting, take off the hooks, and close the run (02a API-08).
    #? Python also calls this when it ends (atexit).
    '''
    
    global _current
    
    if _current is None:
        return None
    folder   = _current.close()
    _current = None
    return folder


def run_current() -> RunState | None:
    '''
    #* Return the run that is going on, or None.
    #? watch.py, note.py and the tests use this.
    '''
    
    return _current


def _run_check(name: str, value: object, ok: bool, condition: str, error: type[Exception]) -> None:
    '''
    #* Raise the error with the message of 03 6章 when a check fails.
    '''
    
    if not ok:
        raise error("引数 " + name + " は " + condition + " にしてください。渡された値: " + repr(value))
    return None


atexit.register(run_close)
