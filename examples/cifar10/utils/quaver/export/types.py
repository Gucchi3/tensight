# @file types.py
# @brief Types for the export
# @author Kohei Yamaguchi, Claude

from typing import Literal

ExportLayout = Literal["channels_last", "channels_first", "hwc", "chw"]
