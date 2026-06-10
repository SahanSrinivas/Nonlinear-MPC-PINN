import re
import zipfile

path = r"G:\Shared drives\Analytics\New Metrics Trackers\Brilliance - Alkermes - Narcolepsy, IH\Alkermes Brilliance Study Monthly Tracker.xlsm"


def col_to_num(col: str) -> int:
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - 64)
    return n


def is_col_a_to_h(ref: str) -> bool:
    m = re.match(r"^([A-Z]+)(\d+)$", ref)
    if not m:
        return False
    c = col_to_num(m.group(1))
    return 1 <= c <= 8


with zipfile.ZipFile(path) as z:
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8")
    rid = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="worksheets/([^"]+)"', rels))
    wb = z.read("xl/workbook.xml").decode("utf-8")
    m = re.search(r'name="Site Screens-NT1"[^>]*r:id="([^"]+)"', wb)
    xml = z.read("xl/worksheets/" + rid[m.group(1)]).decode("utf-8")

occ = []
for cm in re.finditer(r'<c r="([^"]+)"[^>]*>(.*?)</c>', xml, re.DOTALL):
    ref = cm.group(1)
    if not is_col_a_to_h(ref):
        continue
    row = int(re.search(r"\d+", ref).group())
    if row < 6 or row > 250:
        continue
    chunk = cm.group(2)
    has_formula = "<f" in chunk
    vm = re.search(r"<v>(.*?)</v>", chunk)
    val = vm.group(1) if vm else None
    occ.append((row, ref, has_formula, val))

print("A-H occupied rows 6-250:", len(occ))
for item in occ[:30]:
    print(item)
print("...")
for item in occ[-15:]:
    print(item)
rows = sorted({r for r, _, _, _ in occ})
print("distinct rows:", len(rows), "range", rows[:3], "...", rows[-3:])
