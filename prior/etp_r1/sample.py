"""Sample data from ETP-R1."""

from . import ETP_R1_DIR, ANNOTATION_DIR, CONNECTIVITY_DIR
from json import loads, load, dump

# Filter out only E9uDoFAP3SH
TARGET_SCENE = "E9uDoFAP3SH"

# {"instr_id": "prevalent_1144679_3", "scan": "p5wJjkQkbXX", "path": ["767338ad85174714a86e1e60a866a829", "8d6d6147cd1743bd869844e834a0cf77", "40394bc73ced485b9254ad8c874ebbe3", "1ac9330105c84fe3bf9058aebfd26f6a", "846d923830d14a189ab5133f7f6c2d75", "6854cd178a4641b5b8386ad0121b1b04"], "heading": 1.0471975511965976, "instr_encoding": [0, 15504, 25737, 136, 35691, 8305, 70, 109412, 33, 1911, 7514, 5, 7068, 15504, 7108, 13438, 136, 35691, 8305, 70, 6957, 1911, 7, 5, 15504, 7108, 13438, 136, 35691, 11015, 70, 60228, 468, 468, 91, 4, 19686, 214, 8305, 70, 47589, 1911, 7, 47, 70, 50782, 5, 13695, 122395, 8305, 70, 21334, 9803, 1911, 5, 50782, 98, 188, 70, 141947, 5, 7279, 2301, 6626, 80923, 7, 5, 2], "task_type_encoding": 1}
SAMPLE_PREVALENT = ANNOTATION_DIR / "R2R_Prevalent_enc_xlmr.jsonl"
DEST_PREVALENT = ETP_R1_DIR / "sample" / "R2R_Prevalent_enc_xlmr.jsonl"
SAMPLE_CONNECTIVITY = CONNECTIVITY_DIR / "E9uDoFAP3SH_connectivity.json"

with open(SAMPLE_CONNECTIVITY) as f:
    connectivity = load(f)

with open(SAMPLE_PREVALENT) as sample_prevalent:
    with open(DEST_PREVALENT, "w") as dest_prevalent:
        for line in sample_prevalent:
            entry = loads(line)
            if entry["scan"] != TARGET_SCENE:
                continue
            dump(entry, dest_prevalent, ensure_ascii=False, indent=None)
            dest_prevalent.write("\n")
