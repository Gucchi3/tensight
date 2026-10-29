# @file record_writer.py
# @brief RecordWriter: prepares the run folder and writes the records in another thread (C-05)

from base64    import b64encode
from datetime  import datetime
from importlib import metadata
from pathlib   import Path

import platform
import queue
import threading
import torch

from ..core  import message_warn
from ..types import PointInfo, RunSettings, StepRecord
from .folder import folder_add_run, folder_create_run, folder_prepare
from .line   import line_encode, line_record, line_settings



QUEUE_STEPS   = 16
COMPARE_LIMIT = 200 * 2**20

StepItem = tuple[int, str, list[StepRecord], list[tuple[PointInfo, torch.Tensor]], list[tuple[PointInfo, torch.Tensor]]]



class RecordWriter:
    '''
    #* Prepare the run folder, and write the records of each step in another thread (IF-05, IF-08).
    #? RunState makes one for each run. StepCollector calls write_step() at the end of a recorded step.
    #! It never raises into the training thread. A failure shows MSG-004 and stops the writing.
    '''
    
    folder : Path | None
    planned: Path
    
    def __init__(self, log_dir: Path, name: str | None, settings: RunSettings, started: datetime) -> None:
        '''
        #* Make the run folder, write run.js and runs.js, and start the writing thread.
        '''
        
        # Keep the state of the writing thread
        self.folder   = None
        self.planned  = log_dir / "runs" / (started.strftime("%Y%m%d-%H%M%S") + ("_" + name if name else ""))
        self.queue    = queue.Queue[StepItem | None](maxsize=QUEUE_STEPS)
        self.ids      = dict[str, int]()
        self.alerted  = set[int]()
        self.formats  = dict[int, str]()
        self.compared = 0
        self.broken   = False
        self.thread   = None
        
        # Make the folders and write run.js and runs.js; a failure turns the run into one that records nothing
        try:
            folder_prepare(log_dir)
            folder      = folder_create_run(log_dir, started, name)
            started_iso = started.isoformat(timespec="seconds")
            versions    = {"tensight": self._version(), "torch": torch.__version__, "python": platform.python_version()}
            self._append(folder / "run.js", line_encode("run", {"format": 1, "folder": folder.name, "name": name, "started": started_iso, "versions": versions, "settings": line_settings(settings)}), "w")
            folder_add_run(log_dir, {"folder": folder.name, "name": name, "started": started_iso})
        except OSError as error:
            message_warn("MSG-004", path=log_dir, error=error)
            return None
        
        # Start the writing thread
        self.folder  = folder
        self.planned = folder
        self.thread  = threading.Thread(target=self._run, name="tensight-writer", daemon=True)
        self.thread.start()
        return None
    
    def write_step(self, step: int, records: list[StepRecord], compares: list[tuple[PointInfo, torch.Tensor]], fulls: list[tuple[PointInfo, torch.Tensor]]) -> None:
        '''
        #* Hand the records of one step to the writing thread (IF-05).
        #! It waits while QUEUE_STEPS steps are waiting, so that the memory does not grow without end.
        '''
        
        if self.folder is None:
            return None
        self.queue.put((step, datetime.now().astimezone().isoformat(timespec="seconds"), records, compares, fulls))
        return None
    
    def write_now(self, folder: Path, step: int, time: str, records: list[StepRecord], compares: list[tuple[PointInfo, torch.Tensor]], fulls: list[tuple[PointInfo, torch.Tensor]]) -> None:
        '''
        #* Write the catalog lines, the records, the NaN and Inf marks, and the saved tensors of one step.
        #? The writing thread calls this for each step in the queue.
        '''
        
        # Write each record, and add a point line the first time a name comes (02c catalog.js)
        for record in records:
            point_id = self._point(folder, record.point)
            quant    = record.report.quant
            levels   = quant is not None and self.formats.get(point_id) != quant.format
            if quant is not None:
                self.formats[point_id] = quant.format
            self._append(folder / "points" / (str(point_id) + ".js"), line_encode("rec", line_record(folder.name, point_id, step, time, record, levels)), "a")
            
            # Mark the first record that has NaN or Inf (ADR-011)
            if (record.report.nan > 0 or record.report.inf > 0) and point_id not in self.alerted:
                self.alerted.add(point_id)
                self._append(folder / "catalog.js", line_encode("alert", {"format": 1, "run": folder.name, "id": point_id, "step": step, "nan": record.report.nan, "inf": record.report.inf}), "a")
        
        # Write the tensors for the comparison, up to COMPARE_LIMIT bytes in one run (ADR-005)
        for point, tensor in compares:
            if self.compared < 0:
                break
            point_id       = self._point(folder, point)
            data           = tensor.to(torch.float32).contiguous().reshape(-1)
            if self.compared + data.numel() * 4 > COMPARE_LIMIT:
                self.compared = -1
                message_warn("MSG-005")
                break
            self.compared += data.numel() * 4
            text           = b64encode(bytes(data.view(torch.uint8).tolist())).decode("ascii")
            self._append(folder / "compare" / str(step) / (str(point_id) + ".js"), line_encode("cmp", {"format": 1, "run": folder.name, "id": point_id, "step": step, "shape": list(tensor.shape), "dtype": "float32", "data": text}), "w")
        
        # Save the whole tensors (EXT-04)
        for point, tensor in fulls:
            path = folder / "tensors" / str(step) / (str(self._point(folder, point)) + ".pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor, path)
        return None
    
    def close(self) -> None:
        '''
        #* Write everything that is waiting, and stop the writing thread.
        '''
        
        if self.thread is None:
            return None
        self.queue.put(None)
        self.thread.join()
        self.thread = None
        return None
    
    def _run(self) -> None:
        '''
        #* Write the steps from the queue, one by one, until the stop mark comes.
        #! A failure to write a file stops the writing (MSG-004). Any other failure drops that step (MSG-002), so the thread never dies.
        '''
        
        while True:
            item = self.queue.get()
            if item is None:
                return None
            if self.broken or self.folder is None:
                continue
            try:
                self.write_now(self.folder, *item)
            except OSError as error:
                self.broken = True
                message_warn("MSG-004", path=self.folder, error=error)
            except Exception as error:
                message_warn("MSG-002", point="ステップ " + str(item[0]) + " のすべての観測点", error=error)
    
    def _point(self, folder: Path, point: PointInfo) -> int:
        '''
        #* Return the number of a point, and add its line to catalog.js the first time.
        '''
        
        point_id = self.ids.get(point.name)
        if point_id is None:
            point_id             = len(self.ids)
            self.ids[point.name] = point_id
            with open(folder / "catalog.js", "a", encoding="utf-8", newline="\n") as file:
                file.write(line_encode("point", {"format": 1, "run": folder.name, "id": point_id, "name": point.name, "target": point.target, "kind": point.kind, "layer": point.layer}))
        return point_id
    
    def _append(self, path: Path, text: str, mode: str) -> None:
        '''
        #* Write one line in one write call (02c), making the folder when needed.
        '''
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode, encoding="utf-8", newline="\n") as file:
            file.write(text)
        return None
    
    def _version(self) -> str:
        '''
        #* Return the installed version of tensight, or 0.0.0 when it is not installed.
        '''
        
        try:
            return metadata.version("tensight")
        except metadata.PackageNotFoundError:
            return "0.0.0"
