from pathlib import Path

import httpx2
import pytest

from conftest import manifest
from layout_visual_complexity.download import Cache, DownloadError, HttpClient, HttpSettings, cached_json, user_agent
from layout_visual_complexity.inputs.canvases import match_label, normalise_label
from layout_visual_complexity.inputs.iiif import Canvas, canvases


def canvas(position: int, label: str) -> Canvas:
    return Canvas(position, f"c{position}", label, f"s{position}", None, 100, 150)


@pytest.mark.parametrize("label", ["8v", "fol. 8v", "f. 8 v", "008v", "Folio 8V", "f8v"])
def test_labels_match_without_zeros_prefixes_spaces_or_case(label: str) -> None:
    found = [canvas(1, "8r"), canvas(2, label), canvas(3, "9r")]
    match = match_label("0008v", found)
    assert match.canvas == found[1]


def test_recto_and_verso_are_kept_apart() -> None:
    assert normalise_label("0008v") != normalise_label("8r")
    assert normalise_label("FC") == "fc" and normalise_label("p. 36") == "36"


def test_no_match_lists_the_nearest_labels() -> None:
    found = [canvas(n, f"{n}r") for n in range(1, 30)]
    match = match_label("0352v", found)
    assert match.canvas is None and not match.matches
    assert 0 < len(match.nearest) <= 5


def test_two_matches_are_reported_not_chosen() -> None:
    found = [canvas(4, "Ir"), canvas(5, "Iv"), canvas(372, "Ir")]
    match = match_label("Ir", found)
    assert match.canvas is None
    assert [c.position for c in match.matches] == [4, 372]


def test_a_canvas_named_by_its_identifier() -> None:
    found = [
        Canvas(1, "https://x/m/canvas/ID0025r/", "23r", "s1", None, 10, 20),
        Canvas(2, "https://x/m/canvas/ID0027r/", "25r", "s2", None, 10, 20),
    ]
    match = match_label("00055-ID0025r", found)
    assert match.canvas == found[0] and match.by_identifier
    assert match_label("00055-ID0099r", found).canvas is None


def test_canvases_of_a_presentation_2_manifest() -> None:
    found = canvases(manifest("msa"), "https://iiif.example.org/msa/manifest")
    assert [c.label for c in found] == ["FC", "1r", "fol. 1v", "f. 2 r", "3r"]
    assert found[1].image_service == "https://iiif.example.org/images/msa/1"
    assert (found[1].width, found[1].height, found[1].position) == (1000, 1500, 2)


def test_canvases_of_a_presentation_3_manifest() -> None:
    data = {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "type": "Manifest",
        "items": [
            {
                "id": "https://x/c1",
                "type": "Canvas",
                "label": {"none": ["12v"]},
                "width": 10,
                "height": 20,
                "items": [
                    {
                        "type": "AnnotationPage",
                        "items": [
                            {
                                "type": "Annotation",
                                "motivation": "painting",
                                "body": {
                                    "id": "https://x/i1/full/max/0/default.jpg",
                                    "type": "Image",
                                    "service": [{"id": "https://x/i1", "type": "ImageService3"}],
                                    "width": 3000,
                                    "height": 4000,
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }
    (only,) = canvases(data)
    assert (only.label, only.image_service, only.width, only.height) == ("12v", "https://x/i1", 3000, 4000)


def test_the_client_retries_slows_down_and_caches(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx2.Response(503, headers={"retry-after": "0"})
        if len(calls) == 2:
            return httpx2.Response(500)
        return httpx2.Response(200, json={"ok": True})

    cache = Cache(tmp_path / "cache", ".json")
    settings = HttpSettings(interval=0.0, backoff=0.0, contact="team@example.org")
    with HttpClient(settings, httpx2.MockTransport(handler)) as client:
        assert cached_json(client, cache, "https://h.example.org/m") == {"ok": True}
        assert cached_json(client, cache, "https://h.example.org/m") == {"ok": True}
    assert len(calls) == 3
    assert "team@example.org" in calls[0].headers["user-agent"]


def test_the_client_gives_up_on_refusals(tmp_path: Path) -> None:
    transport = httpx2.MockTransport(lambda request: httpx2.Response(403))
    with HttpClient(HttpSettings(interval=0.0, backoff=0.0), transport) as client, pytest.raises(DownloadError):
        client.get("https://h.example.org/m")


def test_the_contact_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LVC_CONTACT", "lab@example.org")
    assert user_agent().endswith("; lab@example.org)")
    monkeypatch.delenv("LVC_CONTACT")
    assert user_agent().startswith("layout-visual-complexity/")


def test_a_page_named_by_its_canvas_identifier_gets_the_canvas_label(tmp_path: Path) -> None:
    import pandas as pd

    from layout_visual_complexity.inputs.canvases import resolve_canvases

    def handler(request: httpx2.Request) -> httpx2.Response:
        data = manifest("msa")
        for number, item in enumerate(data["sequences"][0]["canvases"]):
            item["@id"] = f"https://iiif.example.org/msa/canvas/ID{number + 2:04d}r/"
        return httpx2.Response(200, json=data)

    images = pd.DataFrame(
        {
            "image": ["MS_A-00003-ID0003r"],
            "folio_label": ["00003-ID0003r"],
            "iiif_manifest": ["https://iiif.example.org/msa/manifest"],
            "canvas": [None],
            "image_service": [None],
            "width": pd.array([pd.NA], dtype="Int64"),
            "height": pd.array([pd.NA], dtype="Int64"),
        }
    )
    with HttpClient(HttpSettings(interval=0.0), httpx2.MockTransport(handler)) as client:
        filled, unresolved, _ = resolve_canvases(images, client, Cache(tmp_path))
    # The canvas whose id ends in ID0003r is the second, labelled 1r.
    assert filled.loc[0, "canvas"] == "https://iiif.example.org/msa/canvas/ID0003r/"
    assert filled.loc[0, "folio_label"] == "1r" and unresolved.empty
