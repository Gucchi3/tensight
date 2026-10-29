# @file exceptions.py
# @brief Error of the example: show the message on stderr and stop with a code
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Error codes
1. The numbers of epochs or the record interval are wrong.
2. The float32 weights do not fit the QAT model.
3. The download of CIFAR-10 is broken.
'''
##########################################################

import sys



class Error(SystemExit):
    '''
    #* Show an error of the example on stderr (the message, the detail, and a hint), and stop the program with its code.
    #? main.py and utils/cifar10.py raise it. The codes are listed at the top of this file.
    #! Pass detail, hint, and code as keyword arguments.
    '''
    
    def __init__(self, message: str, *, detail: str | None = None, hint: str | None = None, code: int) -> None:
        '''
        #* Print the error, and keep the code as the exit status.
        '''
        
        lines = [f"[ERROR] {message}"] + ([detail] if detail else []) + ([f"Hint: {hint}"] if hint else [])
        print("\n".join(lines), file=sys.stderr)
        super().__init__(code)
        return None
