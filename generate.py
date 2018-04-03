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

            p, q, mr, cs = params
            p  = float(p.strip())
            q  = float(q.strip())
            mr = int(mr.strip())
            cs = cs.replace(" ", "")

            output_lines.append(template.format(p,q,mr,cs))
    
    with open(out_fn, "w") as f:
        f.writelines(output_lines)

if __name__ == "__main__":
    generate("tests.txt")