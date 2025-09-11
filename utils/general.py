import re
def get_atom_num(isotope: str):
    a = re.compile(r"\d+")
    num = a.findall(isotope)
    if len(num) == 0:
        return 0
    elif len(num) == 1:
        return int(num[0])
    else:
        return int(num[0]) + int(num[1]) * 0.1