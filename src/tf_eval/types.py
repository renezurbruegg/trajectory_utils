from typing import TypedDict

class TfRecord(TypedDict):
    timestamp: int
    parent_frame: str
    child_frame: str
    translation_x: float
    translation_y: float
    translation_z: float
    rotation_x: float
    rotation_y: float
    rotation_z: float
    rotation_w: float