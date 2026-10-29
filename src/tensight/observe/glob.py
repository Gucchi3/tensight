# @file glob.py
# @brief Name patterns with "*" and "?" (02a API-01 points and maps)

import functools
import re



@functools.lru_cache(maxsize=256)
def glob_compile(pattern: str) -> re.Pattern[str]:
    '''
    #* Turn a name pattern into a regular expression: "*" is any text, "?" is one character, and the rest stays as it is.
    #! Use fullmatch(). "[" and "]" are plain characters, so names of nested values can be written.
    '''
    
    return re.compile("".join(".*" if char == "*" else "." if char == "?" else re.escape(char) for char in pattern))
