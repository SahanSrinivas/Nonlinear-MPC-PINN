import re
import zipfile

path = r"G:\Shared drives\Analytics\New Metrics Trackers\Brilliance - Alkermes - Narcolepsy, IH\Alkermes Brilliance Study Monthly Tracker.xlsm"


def col_to_num(col: str) -> int:
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - 64)
    return n


def parse_ref(ref: str):
    m = re.match(r"^([A-Z]+)(\d+)$", ref)
    return col_to_num(m.group(1)), int(m.group(2))


with zipfile.ZipFile(path) as z:
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8")
    rid = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="worksheets/([^"]+)"', rels))
    wb = z.read("xl/workbook.xml").decode("utf-8")
    m = re.search(r'name="Site Screens-NT1"[^>]*r:id="([^"]+)"', wb)
    xml = z.read("xl/worksheets/" + rid[m.group(1)]).decode("utf-8")

occupied = set()
for cm in re.finditer(r'<c r="([^"]+)"', xml):
    c, r = parse_ref(cm.group(1))
    occupied.add((r, c))

# find first 30-row window where cols 1-8 are all empty starting row >=6
for start in range(6, 300):
    ok = True
    for r in range(start, start + 35):
        for c in range(1, 9):
            if (r, c) in occupied:
                ok = False
                break
        if not ok:
            break
    if ok:
        print("Empty block A:H rows", start, "to", start + 34)
        break
else:
    print("No empty 35-row block in A:H up to row 300")

# check columns J:Q rows 6-50
for start in range(6, 100):
    ok = True
    for r in range(start, start + 35):
        for c in range(10, 18):  # J=10 to Q=17
            if (r, c) in occupied:
                ok = False
                break
        if not ok:
            break
    if ok:
        print("Empty block J:Q rows", start, "to", start + 34)
        break

# decode a few shared strings for A6 values
ss = z.read("xl/sharedStrings.xml").decode("utf-8")
strings = re.findall(r"<t[^>]*>(.*?)</t>", ss)
for idx in ["55542", "55852", "55829", "55541", "56085"]:
    i = int(idx)
    if i < len(strings):
        print("shared", idx, "->", strings[i][:60])
