# @file pattern.py
# @brief Patterns of maps in the einops style, and their presets (FR-035, ADR-012)

from dataclasses import replace

import functools
import math
import re

from ..types import MapPattern, MapSpec
from .types  import MapPlan

##########################################################
'''
Add a preset
1. Write the pattern in the einops style (02a 4章).
2. Add "name: pattern" to MAP_PRESET_REGISTRY.
3. Add the name to the table of presets in 02a 4章. If its cells should be square, add it to mapSquare() in tensight_core.js too.
'''
##########################################################



MAP_PRESET_REGISTRY = {
    "conv1d"         : "batch channel length -> channel length",
    "conv2d"         : "batch channel height width -> channel (height width)",
    "conv2d_image"   : "batch channel height width -> height width",
    "conv3d"         : "batch channel depth height width -> channel (depth height width)",
    "tokens"         : "batch token dim -> dim token",
    "attention_qk"   : "batch head query key -> query key",
    "attention_heads": "batch head query key -> head (query key)",
    "weight"         : "out in ... -> out (in ...)",
    "matrix"         : "... row col -> row col",
}
PATTERN_DEFAULTS    = {1: "dim -> dim", 2: "batch dim -> dim", 3: "tokens", 4: "batch ... dim -> dim (...)"}
PATTERN_NAME        = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PATTERN_TOKEN       = re.compile(r"\s*(?:(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<dots>\.\.\.)|(?P<open>\()|(?P<close>\))|(?P<bad>\S))")



@functools.cache
def pattern_parse(text: str) -> MapSpec:
    '''
    #* Read a pattern, a preset name, or "auto" (IF-11, 02a 4章).
    #? run_start(), note_tensor() and inspect_tensor() call this at their entry.
    #! Raises ValueError that says where the mistake is (02a 4章 table of mistakes).
    '''
    
    # A single name is "auto" or a preset
    word = text.strip()
    if "->" not in text and PATTERN_NAME.fullmatch(word):
        if word == "auto":
            return "auto"
        if word in MAP_PRESET_REGISTRY:
            spec = pattern_parse(MAP_PRESET_REGISTRY[word])
            return replace(spec, preset=word) if isinstance(spec, MapPattern) else spec
        raise ValueError("並べ方の式「" + text + "」: プリセット「" + word + "」はありません。使える名前: " + "、".join([*MAP_PRESET_REGISTRY, "auto"]))
    
    # Split the text at the one arrow
    head  = "並べ方の式「" + text + "」"
    if text.count("->") != 1:
        raise ValueError(head + ": 「->」が " + str(text.count("->")) + " 個あります。1 個にしてください")
    arrow = text.index("->")
    
    # Cut both sides into names, "...", and brackets, keeping the positions (counted from 1)
    sides = list[list[tuple[str, str, int]]]()
    for part, start in ((text[:arrow], 0), (text[arrow + 2:], arrow + 2)):
        items = list[tuple[str, str, int]]()
        for found in PATTERN_TOKEN.finditer(part):
            kind     = found.lastgroup or "bad"
            position = start + found.start(kind) + 1
            if kind == "bad":
                raise ValueError(head + "の " + str(position) + " 文字目: 名前に使えない文字「" + found.group(kind) + "」があります")
            items.append((kind, found.group(kind), position))
        sides.append(items)
    
    # Read the left side: names and at most one "...", without brackets
    left = list[str]()
    for kind, word, position in sides[0]:
        if kind in ("open", "close"):
            raise ValueError(head + "の " + str(position) + " 文字目: 左側に括弧があります")
        name = "..." if kind == "dots" else word
        if name in left:
            raise ValueError(head + "の " + str(position) + " 文字目: 左側で「" + name + "」を2回使っています")
        left.append(name)
    
    # Read the right side into groups: one name, "...", or the names in one pair of brackets
    groups  = list[tuple[str, ...]]()
    current = None
    opened  = 0
    for kind, word, position in sides[1]:
        if kind == "open":
            if current is not None:
                raise ValueError(head + "の " + str(position) + " 文字目: 括弧が入れ子になっています")
            current = list[str]()
            opened  = position
            continue
        if kind == "close":
            if current is None:
                raise ValueError(head + "の " + str(position) + " 文字目: 「)」に対応する「(」がありません")
            if not current:
                raise ValueError(head + "の " + str(opened) + " 文字目: 括弧の中が空です")
            groups.append(tuple(current))
            current = None
            continue
        name = "..." if kind == "dots" else word
        if name not in left:
            raise ValueError(head + "の " + str(position) + " 文字目: 右側の「" + name + "」が左側にありません")
        if any(name in group for group in groups) or (current is not None and name in current):
            raise ValueError(head + "の " + str(position) + " 文字目: 右側で「" + name + "」を2回使っています")
        if current is None:
            groups.append((name,))
        else:
            current.append(name)
    
    # Check the brackets and the number of groups
    if current is not None:
        raise ValueError(head + "の " + str(opened) + " 文字目: 括弧が閉じていません")
    if len(groups) not in (1, 2):
        raise ValueError(head + ": 右側の組が " + str(len(groups)) + " 個あります。1 個か 2 個にしてください")
    return MapPattern(text.strip(), None, tuple(left), tuple(groups))


def pattern_default(ndim: int) -> MapPattern | None:
    '''
    #* Return the fixed pattern for a number of dimensions (02a 4章), or None for 0 dimensions.
    '''
    
    if ndim == 0:
        return None
    spec = pattern_parse(PATTERN_DEFAULTS[min(ndim, 4)])
    return spec if isinstance(spec, MapPattern) else None


def pattern_plan(pattern: MapPattern, shape: tuple[int, ...]) -> MapPlan | None:
    '''
    #* Fit a pattern to the shape of a tensor, or return None when the numbers of dimensions do not match (03 U-021).
    '''
    
    # Check the number of dimensions; "..." takes the dimensions that the names leave
    count = len(shape)
    named = len([name for name in pattern.left if name != "..."])
    if ("..." in pattern.left and count < named) or ("..." not in pattern.left and count != named):
        return None
    
    # Give each name its dimensions, from the front
    dims  = dict[str, tuple[int, ...]]()
    start = 0
    for name in pattern.left:
        width      = count - named if name == "..." else 1
        dims[name] = tuple(range(start, start + width))
        start     += width
    
    # The first group is the rows, the second the columns, and the rest is reduced
    rows   = tuple(dim for name in pattern.groups[0] for dim in dims[name])
    cols   = tuple(dim for name in pattern.groups[1] for dim in dims[name]) if len(pattern.groups) == 2 else ()
    reduce = tuple(dim for dim in range(count) if dim not in rows and dim not in cols)
    return MapPlan(reduce, rows, cols, math.prod(shape[dim] for dim in rows), math.prod(shape[dim] for dim in cols), pattern.groups[0], pattern.groups[1] if len(pattern.groups) == 2 else ())
