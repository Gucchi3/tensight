# @file run_state.py
# @brief RunState: the state of one run, and how its steps go on (C-01)

from pathlib import Path

from .observe import ModelObserver
from .stats   import StepCollector
from .types   import RunSettings
from .writer  import RecordWriter



class RunState:
    '''
    #* Hold the state of one run: the settings, the step number, whether the step that runs now is recorded, and the parts.
    #? run.py keeps the current one. watch_model() adds observers to it.
    #! step is the number of finished steps; the step that runs now is step + 1.
    '''
    
    settings : RunSettings
    writer   : RecordWriter
    collector: StepCollector
    observers: list[ModelObserver]
    step     : int
    recording: bool
    
    def __init__(self, settings: RunSettings, writer: RecordWriter, collector: StepCollector) -> None:
        '''
        #* Start at step 0; step 1 is recorded only when the interval is 1 (or step 1 is saved).
        '''
        
        self.settings  = settings
        self.writer    = writer
        self.collector = collector
        self.observers = []
        self.step      = 0
        self.recording = self._decide(1)
        if self.recording:
            self.collector.begin(1)
        return None
    
    def watch(self, observer: ModelObserver) -> None:
        '''
        #* Add the observer of a model; an older observer of the same model is taken off first (02a API-02).
        '''
        
        for old in [found for found in self.observers if found.model is observer.model]:
            old.detach()
            self.observers.remove(old)
        observer.set_recording(self.recording, self.step + 1)
        self.observers.append(observer)
        return None
    
    def advance(self) -> None:
        '''
        #* Finish the step that ran, and decide whether the next one is recorded (02a API-03).
        '''
        
        # Hand the records of a recorded step to the writer (IF-05)
        if self.recording:
            self.collector.finish(self.step + 1)
        
        # Go to the next step, and tell the observers whether it is recorded (IF-02)
        self.step      += 1
        self.recording  = self._decide(self.step + 1)
        for observer in self.observers:
            observer.set_recording(self.recording, self.step + 1)
        if self.recording:
            self.collector.begin(self.step + 1)
        return None
    
    def mark_next(self) -> None:
        '''
        #* Record the step that runs now, whatever the interval is (02a API-05).
        '''
        
        self.recording = True
        for observer in self.observers:
            observer.set_recording(True, self.step + 1)
        self.collector.begin(self.step + 1)
        return None
    
    def close(self) -> Path | None:
        '''
        #* Take off the hooks, forget the step that did not finish, and write everything that is waiting (02a API-08).
        '''
        
        for observer in self.observers:
            observer.detach()
        self.observers.clear()
        self.collector.clear()
        self.writer.close()
        return self.writer.folder
    
    def _decide(self, number: int) -> bool:
        '''
        #* Return whether a step is recorded: a multiple of the interval, or a step of compare_steps or save_full.
        '''
        
        settings = self.settings
        return number % settings.interval == 0 or number in settings.compare_steps or any(number in steps for steps in settings.save_full.values())
