"""
__author__ = Yash Patel
__name__   = generate.py
__description__ = Generates the tests to be run given a file in root directory
"""

import os

def generate(in_fn="tests.txt",out_fn="tests.sh"):
    if not os.path.isfile(in_fn):
        print("No test file found! Exiting now...")

    template = "python3 app.py -r y -p {} -q {} --mr {} --cs {}"
    output_lines = []
    with open(in_fn, "r") as f:
        for line in f:
            params = line.split(";")
            if len(params) != 4:
                continue

            p, qs, mr, cs = params
            p  = float(p.strip())
            qs = [float(q.strip()) for q in qs.split(",")]
            mr = int(mr.strip())
            cs = cs.replace(" ", "")

            for q in qs:
                output_lines.append(template.format(p,q,mr,cs))
    
    with open(out_fn, "w") as f:
        f.writelines(output_lines)

if __name__ == "__main__":
    generate("tests.txt")