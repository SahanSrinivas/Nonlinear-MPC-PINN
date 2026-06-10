import re
import zipfile

path = r"G:\Shared drives\Analytics\New Metrics Trackers\Brilliance - Alkermes - Narcolepsy, IH\Alkermes Brilliance Study Monthly Tracker.xlsm"

with zipfile.ZipFile(path) as z:
    styles = z.read("xl/styles.xml").decode("utf-8")
    # cellXfs with protection
    xfs = re.findall(r"<xf ([^/>]*)/>", styles)
    locked_xf = []
    for i, xf in enumerate(xfs):
        if 'locked="0"' in xf:
            locked_xf.append(i)
        elif "applyProtection" in xf and 'locked="0"' not in xf:
            pass
    print("cellXfs count", len(xfs))
    print("unlocked xf indices sample", locked_xf[:20], "count", len(locked_xf))

    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8")
    rid = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="worksheets/([^"]+)"', rels))
    wb = z.read("xl/workbook.xml").decode("utf-8")
    m = re.search(r'name="Site Screens-NT1"[^>]*r:id="([^"]+)"', wb)
    xml = z.read("xl/worksheets/" + rid[m.group(1)]).decode("utf-8")

    for ref in ["A6", "B6", "D6", "A7", "E10", "A125"]:
        cm = re.search(rf'<c r="{ref}"([^>]*)>', xml)
        if cm:
            print(ref, "attrs:", cm.group(1).strip())
