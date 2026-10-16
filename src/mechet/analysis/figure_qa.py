"""Machine-check vector delivery without pretending to inspect the chemistry."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .tree_records import file_digest


def inspect_figure(files, style):
    from PIL import Image
    from pypdf import PdfReader

    problems = []
    for ext in ("svg", "pdf", "png"):
        info = files[ext]
        if not Path(info["path"]).is_file() or file_digest(info["path"]) != info["sha256"]:
            problems.append("file_hash_mismatch:" + ext)
    root = ET.parse(files["svg"]["path"]).getroot()
    counts = Counter(node.tag.rsplit("}", 1)[-1] for node in root.iter())
    ids = [node.attrib["id"] for node in root.iter() if "id" in node.attrib]
    references = []
    for node in root.iter():
        for key, value in node.attrib.items():
            references.extend(re.findall(r"url\(#([^)]*)\)", value))
            if key.endswith("href") and value.startswith("#"):
                references.append(value[1:])
    if len(ids) != len(set(ids)):
        problems.append("duplicate_svg_ids")
    missing = sorted(set(references) - set(ids))
    if missing:
        problems.append("unresolved_svg_references")
    if counts["image"]:
        problems.append("raster_embedded_in_svg")
    if not counts["path"]:
        problems.append("no_vector_paths")
    reader = PdfReader(files["pdf"]["path"])
    page = reader.pages[0]
    width_mm, height_mm = [float(v) * 25.4 / 72 for v in [page.mediabox.width, page.mediabox.height]]
    if len(reader.pages) != 1 or abs(width_mm - style["format"]["full_width_mm"]) > .1 or height_mm > style["format"]["max_height_mm"] + .1:
        problems.append("incorrect_publication_page_geometry")
    fonts = []
    for ref in page.get("/Resources", {}).get("/Font", {}).values():
        font = ref.get_object()
        descendants = font.get("/DescendantFonts") or [font]
        for child in descendants:
            child = child.get_object()
            descriptor = child.get("/FontDescriptor")
            descriptor = descriptor.get_object() if descriptor else {}
            fonts.append({"name": str(child.get("/BaseFont")), "subtype": str(child.get("/Subtype")),
                "embedded": any(k in descriptor for k in ["/FontFile", "/FontFile2", "/FontFile3"])})
    embedded = bool(fonts) and all(font["embedded"] for font in fonts)
    if not embedded:
        problems.append("missing_embedded_pdf_fonts")
    with Image.open(files["png"]["path"]) as im:
        pixels = im.size
    dpi = pixels[0] / (width_mm / 25.4)
    if dpi < style["format"]["preview_dpi"] - 1:
        problems.append("preview_resolution_too_low")
    return {"machine_checks_passed": not problems, "problems": problems,
        "svg": {"ids": len(ids), "duplicate_ids": len(ids) - len(set(ids)), "unresolved_references": missing,
            "raster_images": counts["image"], "vector_paths": counts["path"], "editable_text_elements": counts["text"]},
        "pdf": {"pages": len(reader.pages), "width_mm": width_mm, "height_mm": height_mm,
            "fonts": fonts, "all_fonts_embedded": embedded},
        "png": {"width_px": pixels[0], "height_px": pixels[1], "effective_dpi": dpi},
        "visual_inspection_status": "not_established_by_machine_checks",
        "chemistry_correctness_status": "requires_source_and_drawing_review"}
