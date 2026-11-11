from pathlib import Path

import httpx2
import pytest

from layout_visual_complexity.download import Cache, DownloadError, HttpClient, HttpSettings, cached_json, user_agent
from layout_visual_complexity.inputs.iiif import canvases

HOST = "https://iiif.example.org"


def manifest(key: str) -> dict:
    """A IIIF Presentation 2 manifest: a cover, then the folios."""
    items = []
    for number, label in enumerate(["FC", "1r", "fol. 1v", "f. 2 r", "3r"]):
        service = f"{HOST}/images/{key}/{number}"
        resource = {
            "@id": f"{service}/full/full/0/default.jpg",
            "@type": "dctypes:Image",
            "width": 1000,
            "height": 1500,
            "service": {"@context": "http://iiif.io/api/image/2/context.json", "@id": service},
        }
        items.append(
            {
                "@id": f"{HOST}/{key}/canvas/{number}",
                "@type": "sc:Canvas",
                "label": label,
                "width": 1000,
                "height": 1500,
                "images": [{"@type": "oa:Annotation", "motivation": "sc:painting", "resource": resource}],
            }
        )
    return {
        "@context": "http://iiif.io/api/presentation/2/context.json",
        "@id": f"{HOST}/{key}/manifest",
        "@type": "sc:Manifest",
        "sequences": [{"canvases": items}],
    }


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
