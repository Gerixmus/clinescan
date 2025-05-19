from dataclasses import dataclass
from typing import List

@dataclass
class Function:
    code: List[str]
    vul: int
    flaw_line_no: List[int]
    bigvul_id: int

    def line_vulnerable(self, line_no: int) -> bool:
        return line_no in self.flaw_line_no