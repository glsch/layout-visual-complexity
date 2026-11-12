from pathlib import Path

import cv2
import numpy as np
import pytest

from layout_visual_complexity.inputs import InputError
from layout_visual_complexity.inputs.resolve import InputSettings, Resolver
from layout_visual_complexity.inputs.selection import (
    FOLIO_FILTER,
    document_id,
    label_filter,
    page_folio,
    page_ids,
    passes,
    select,
)

HOST = "https://iiif.example.org"
LABELS = ["FC", "Ir", "0001r", "0001v", "0002r", "0002v", "0002bis r", "0003r", "Ir", "BC"]


def manifest(labels: list[str], shelfmark: str = "I. G. 28.") -> dict:
    canvases = []
    for number, label in enumerate(labels):
        service = f"{HOST}/images/{number}"
        resource = {
            "@id": f"{service}/full/full/0/default.jpg",
            "@type": "dctypes:Image",
            "width": 3000,
            "height": 4000,
            "service": {"@context": "http://iiif.io/api/image/2/context.json", "@id": service},
        }
        canvases.append(
            {
                "@id": f"{HOST}/canvas/{number}",
                "@type": "sc:Canvas",
                "label": label,
                "width": 3000,
                "height": 4000,
                "images": [{"@type": "oa:Annotation", "motivation": "sc:painting", "resource": resource}],
            }
        )
    return {
        "@context": "http://iiif.io/api/presentation/2/context.json",
        "@id": f"{HOST}/manifest",
        "@type": "sc:Manifest",
        "label": "Missale",
        "metadata": [{"label": "Shelfmark/Identifier", "value": shelfmark}],
        "sequences": [{"canvases": canvases}],
    }


def resolver(settings: InputSettings | None = None, labels: list[str] = LABELS) -> Resolver:
    return Resolver(settings or InputSettings(), lambda url: manifest(labels))


def test_document_and_page_ids_follow_the_dataset() -> None:
    assert document_id("I. G. 28.") == "ig28"
    assert document_id("Teplá MS.E.9") == "teplamse9"
    assert page_folio("0115r") == "115r" and page_folio("0002bis r") == "2bisr"
    assert page_ids("ig28", ["0001r", None, "Ir", "Ir"]) == ["ig28_1r", "ig28_p2", "ig28_ir_p3", "ig28_ir_p4"]


def test_page_ids_dont_depend_on_the_selection() -> None:
    every = resolver().resolve([f"{HOST}/manifest"])[0].pages
    some = resolver(InputSettings(pages="9")).resolve([f"{HOST}/manifest"])[0].pages
    assert [page.page for page in some] == ["ig28_ir_p9"]
    assert {page.page for page in every} >= {page.page for page in some}


def test_selections_by_index_and_by_label() -> None:
    assert select("3-5", LABELS) == [3, 4, 5]
    assert select("1r-2v", LABELS) == [3, 4, 5, 6]
    assert select("fol. 1v, 9-", LABELS) == [4, 9, 10]
    assert select("", LABELS) == list(range(1, 11))
    numbered = ["1", "2", "3", "4"]
    assert select("label:2-label:3", numbered) == [2, 3]
    with pytest.raises(InputError, match="several pages"):
        select("Ir", LABELS)
    with pytest.raises(InputError, match="out of range"):
        select("11", LABELS)
    with pytest.raises(InputError, match="backwards"):
        select("2v-1r", LABELS)


def test_the_folio_filter_keeps_the_leaves() -> None:
    leaves = label_filter(FOLIO_FILTER)
    kept = [label for label in LABELS if passes(label, leaves)]
    assert kept == ["0001r", "0001v", "0002r", "0002v", "0002bis r", "0003r"]
    assert passes("FC", None) and not passes(None, leaves)


def test_a_manifest_with_a_selection_and_a_filter() -> None:
    settings = InputSettings(pages="1-8", label_filter=FOLIO_FILTER)
    (document,) = resolver(settings).resolve([f"{HOST}/manifest"])
    assert document.id == "ig28"
    assert [page.page for page in document.pages] == [
        "ig28_1r",
        "ig28_1v",
        "ig28_2r",
        "ig28_2v",
        "ig28_2bisr",
        "ig28_3r",
    ]
    first = document.pages[0]
    assert (first.position, first.label, first.canvas) == (3, "0001r", f"{HOST}/canvas/2")
    assert (first.image_service, first.width, first.height) == (f"{HOST}/images/2", 3000, 4000)


def image(path: Path, size: tuple[int, int] = (60, 80)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((size[1], size[0], 3), 200, np.uint8))
    return path


def test_a_folder_a_single_image_and_their_ids(tmp_path: Path) -> None:
    folder = tmp_path / "BnF fr. 24428"
    for name in ["010r.jpg", "002v.jpg", "notes.xml", "001r.png"]:
        image(folder / name) if name.endswith(("jpg", "png")) else (folder / name).write_text("<alto/>")
    (document,) = resolver().resolve([str(folder)])
    assert document.id == "bnffr24428"
    assert [page.page for page in document.pages] == ["bnffr24428_1r", "bnffr24428_2v", "bnffr24428_10r"]
    (single,) = resolver().resolve([str(folder / "002v.jpg")])
    assert single.pages[0].page == "bnffr24428_2v" and single.pages[0].path.endswith("002v.jpg")


def test_list_files_plain_and_tabular(tmp_path: Path) -> None:
    folder = tmp_path / "ms"
    for number in range(1, 4):
        image(folder / f"{number}r.jpg")
    plain = tmp_path / "pages.txt"
    plain.write_text(f"# the sample\n{HOST}/manifest\t1r-1v\nms/2r.jpg\nms/3r.jpg\n")
    documents = resolver().resolve([str(plain)])
    assert [(d.id, [p.page for p in d.pages]) for d in documents] == [
        ("ig28", ["ig28_1r", "ig28_1v"]),
        ("ms", ["ms_2r", "ms_3r"]),
    ]
    table = tmp_path / "pages.tsv"
    table.write_text(f"manifest\tcanvas\tlabel\n{HOST}/manifest\t{HOST}/canvas/7\t\n{HOST}/manifest\t\t0002r\n")
    (document,) = resolver().resolve([str(table)])
    assert [page.page for page in document.pages] == ["ig28_2r", "ig28_3r"]


def test_a_page_table_keeps_its_ids(tmp_path: Path) -> None:
    table = tmp_path / "images.tsv"
    rows = [
        "image\tmanuscript\tfolio\tcanvas\timage_url\timage_service\twidth\theight",
        f"msa_1r\tmsa\t0001r\t{HOST}/c/1\t\t{HOST}/i/1\t1000\t1500",
        f"msb_2v\tmsb\t0002v\t{HOST}/c/2\t{HOST}/i/2.jpg\t\t\t",
        f"msa_3r\tmsa\t0003r\t{HOST}/c/3\t\t{HOST}/i/3\t1000\t1500",
    ]
    table.write_text("\n".join(rows) + "\n")
    documents = Resolver(InputSettings()).resolve([str(table)])
    assert [(d.id, [p.page for p in d.pages]) for d in documents] == [
        ("msa", ["msa_1r", "msa_3r"]),
        ("msb", ["msb_2v"]),
    ]
    assert documents[1].pages[0].image_url == f"{HOST}/i/2.jpg" and documents[1].pages[0].width is None


def test_image_requests_and_services_as_sources() -> None:
    (document,) = resolver().resolve([f"{HOST}/iiif/abc/full/max/0/default.jpg"])
    assert document.pages[0].image_service == f"{HOST}/iiif/abc"
    (phaidra,) = resolver().resolve(["https://phaidra.example.org/api/imageserver?IIIF=o:2222610.tif"])
    assert phaidra.id == "o2222610tif" and phaidra.pages[0].image_service.endswith("IIIF=o:2222610.tif")


def test_two_images_under_one_id_are_refused(tmp_path: Path) -> None:
    image(tmp_path / "a" / "ms" / "1r.jpg")
    image(tmp_path / "b" / "ms" / "1r.jpg")
    with pytest.raises(InputError, match="both named"):
        resolver().resolve([str(tmp_path / "a" / "ms"), str(tmp_path / "b" / "ms")])
