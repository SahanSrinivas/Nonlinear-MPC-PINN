import re
import zipfile
import xml.etree.ElementTree as ET

path = r"G:\Shared drives\Analytics\New Metrics Trackers\Brilliance - Alkermes - Narcolepsy, IH\Alkermes Brilliance Study Monthly Tracker.xlsm"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

with zipfile.ZipFile(path) as z:
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8")
    rid_to_file = dict(
        re.findall(r'Id="(rId\d+)"[^>]*Target="worksheets/([^"]+)"', rels)
    )
    wbxml = z.read("xl/workbook.xml").decode("utf-8")
    m = re.search(r'name="Site Screens-NT1"[^>]*r:id="([^"]+)"', wbxml)
    sheet_file = rid_to_file[m.group(1)]
    sheet_xml = z.read("xl/worksheets/" + sheet_file).decode("utf-8")
    rels_path = "xl/worksheets/_rels/" + sheet_file + ".rels"
    srels = z.read(rels_path).decode("utf-8") if rels_path in z.namelist() else ""

root = ET.fromstring(sheet_xml)

# merged cells
merge = root.find("m:mergeCells", NS)
if merge is not None:
    print("MERGED CELLS:")
    for mc in merge.findall("m:mergeCell", NS):
        print(" ", mc.attrib.get("ref"))

# sheet protection
prot = root.find("m:sheetProtection", NS)
print("PROTECTED:", prot is not None)
if prot is not None:
    print(" ", prot.attrib)

# table parts
if srels:
    tables = re.findall(r'Target="../tables/([^"]+)"', srels)
    print("TABLES:", tables)

# scan cells A6:H250 for any value/formula
cells = {}
for c in root.findall(".//m:sheetData/m:row/m:c", NS):
    ref = c.attrib.get("r", "")
    col = re.match(r"([A-Z]+)", ref)
    if not col:
        continue
    col = col.group(1)
    if col > "H":
        continue
    row = int(re.sub(r"[A-Z]", "", ref))
    if row < 6 or row > 250:
        continue
    f = c.find("m:f", NS)
    v = c.find("m:v", NS)
    t = c.attrib.get("t")
    has_formula = f is not None
    val = None
    if v is not None:
        val = v.text
    cells[ref] = {"formula": has_formula, "value": val, "type": t}

occupied = sorted(cells.keys(), key=lambda x: (int(re.sub(r"[A-Z]", "", x)), x))
print(f"OCCUPIED A6:H250 count: {len(occupied)}")
for ref in occupied[:40]:
    info = cells[ref]
    print(ref, info)
if len(occupied) > 40:
    print("...")
    for ref in occupied[-10:]:
        print(ref, cells[ref])

# rows with anything in A-H
rows_with_data = set()
for ref in occupied:
    rows_with_data.add(int(re.sub(r"[A-Z]", "", ref)))
print("Rows 6-250 with any A-H content:", len(rows_with_data))
print("Last occupied row:", max(rows_with_data) if rows_with_data else None)
