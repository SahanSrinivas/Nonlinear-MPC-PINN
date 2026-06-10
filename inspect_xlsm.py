import re
import zipfile

path = r"G:\Shared drives\Analytics\New Metrics Trackers\Brilliance - Alkermes - Narcolepsy, IH\Alkermes Brilliance Study Monthly Tracker.xlsm"

with zipfile.ZipFile(path) as z:
    wb = z.read("xl/workbook.xml").decode("utf-8")
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8")
    rid_to_file = dict(
        re.findall(r'Id="(rId\d+)"[^>]*Target="worksheets/([^"]+)"', rels)
    )

    for m in re.finditer(r'name="(Site Screens-NT1)"[^>]*r:id="([^"]+)"', wb):
        fname = "xl/worksheets/" + rid_to_file[m.group(2)]
        xml = z.read(fname).decode("utf-8")
        for cell in ["A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6", "A125", "E7"]:
            mm = re.search(rf'<c r="{cell}"[^>]*>.*?</c>', xml, re.DOTALL)
            if not mm:
                print(f"--- {cell}: not found")
                continue
            chunk = mm.group(0)
            f = re.search(r"<f[^>]*>(.*?)</f>", chunk, re.DOTALL)
            print(f"--- {cell}")
            if f:
                print(f.group(1)[:800])
