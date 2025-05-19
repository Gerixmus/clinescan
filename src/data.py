import json
from function import Function

def get_entries() -> list[Function]:
    with open("data.json", "r") as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        code_lines = item["code"].split('\n')
        func = Function(
            code=code_lines,
            vul=item["vul"],
            flaw_line_no=item["flaw_line_no"],
            bigvul_id=item["bigvul_id"]
        )
        data.append(func)
    return data
