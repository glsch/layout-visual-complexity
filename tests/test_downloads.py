import io
import threading
import time
from pathlib import Path

import httpx2
import numpy as np
from PIL import Image

from layout_visual_complexity.download import (
    HttpClient,
    HttpSettings,
    ImageCache,
    ImageFetcher,
    ImageSettings,
    image_request,
    size_request,
)

HOST = "https://iiif.example.org"


def jpeg(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((height, width, 3), 180, np.uint8)).save(buffer, "JPEG")
    return buffer.getvalue()


def test_the_size_asked_never_exceeds_the_images_own() -> None:
    assert size_request(1157, 1645, 2000) == "full"
    assert size_request(1157, 1645, 2000, image_api=3) == "max"
    assert size_request(2000, 1400, 2000) == "full"
    assert size_request(3000, 4000, 2000) == ",2000"
    assert size_request(4000, 3000, 2000) == "2000,"
    assert image_request(f"{HOST}/i/1/", 3000, 4000, 2000) == f"{HOST}/i/1/full/,2000/0/default.jpg"


class Server:
    """A IIIF image server for one image of 3000×4000 and one of 1000×1500; it answers any size at the right shape."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        self.requests.append(url)
        full = (3000, 4000) if "/big/" in url else (1000, 1500)
        if url.endswith("/info.json"):
            return httpx2.Response(200, json={"width": full[0], "height": full[1]})
        size = url.split("/")[-3]
        if size in ("full", "max"):
            return httpx2.Response(200, content=jpeg(*full))
        width, height = size.split(",")
        if width:
            return httpx2.Response(200, content=jpeg(int(width), round(int(width) * full[1] / full[0])))
        return httpx2.Response(200, content=jpeg(round(int(height) * full[0] / full[1]), int(height)))


def test_page_images_come_at_the_long_edge_with_the_full_size_kept(tmp_path: Path) -> None:
    server = Server()
    settings = HttpSettings(interval=0.0, backoff=0.0)
    with HttpClient(settings, httpx2.MockTransport(server)) as client:
        fetcher = ImageFetcher(client, tmp_path, ImageSettings(long_edge=2000))
        big = fetcher.from_service(f"{HOST}/big/1", 3000, 4000, 2)
        small = fetcher.from_service(f"{HOST}/small/1", None, None, 2)
        again = fetcher.from_service(f"{HOST}/big/1", 3000, 4000, 2)
    assert (big.width, big.height, big.full_width, big.full_height) == (1500, 2000, 3000, 4000)
    assert (small.width, small.height, small.full_width, small.full_height) == (1000, 1500, 1000, 1500)
    assert again.path == big.path
    assert server.requests == [
        f"{HOST}/big/1/full/,2000/0/default.jpg",
        f"{HOST}/small/1/info.json",
        f"{HOST}/small/1/full/full/0/default.jpg",
    ]


def test_a_large_local_image_is_made_smaller_and_a_small_one_used_as_it_is(tmp_path: Path) -> None:
    large, small = tmp_path / "large.jpg", tmp_path / "small.jpg"
    large.write_bytes(jpeg(2600, 3900))
    small.write_bytes(jpeg(800, 1200))
    fetcher = ImageFetcher(HttpClient(HttpSettings()), tmp_path / "cache", ImageSettings(long_edge=2000))
    made = fetcher.from_file(large)
    used = fetcher.from_file(small)
    assert (made.width, made.height, made.full_width, made.full_height) == (1333, 2000, 2600, 3900)
    assert made.path.parent == tmp_path / "cache" / "images"
    assert used.path == small and not used.cached


def test_the_cache_evicts_the_oldest_past_its_bound_but_not_what_is_in_use(tmp_path: Path) -> None:
    cache = ImageCache(tmp_path, limit=250)
    first = cache.put("first", b"x" * 100)
    cache.release(first)
    second = cache.put("second", b"x" * 100)  # in use: pinned
    time.sleep(0.01)
    third = cache.put("third", b"x" * 100)
    assert not first.exists() and second.exists() and third.exists()
    cache.release(third)
    fourth = cache.put("fourth", b"x" * 100)
    assert second.exists() and not third.exists() and fourth.exists()
    assert cache.get("second") == second and cache.get("third") is None
    assert ImageCache(tmp_path, limit=250).size == 200


def test_hosts_are_fetched_from_at_once_each_at_its_own_pace() -> None:
    active: dict[str, int] = {}
    most: dict[str, int] = {}
    lock = threading.Lock()

    def slow(request: httpx2.Request) -> httpx2.Response:
        host = request.url.host
        with lock:
            active[host] = active.get(host, 0) + 1
            most[host] = max(most.get(host, 0), active[host])
            together = sum(active.values())
            most["all"] = max(most.get("all", 0), together)
        time.sleep(0.05)
        with lock:
            active[host] -= 1
        return httpx2.Response(200, content=b"ok")

    settings = HttpSettings(interval=0.0, connections=1, backoff=0.0)
    with HttpClient(settings, httpx2.MockTransport(slow)) as client:
        urls = [f"https://{host}.example.org/{n}" for host in ("a", "b", "c") for n in range(4)]
        threads = [threading.Thread(target=client.get, args=(url,)) for url in urls]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert most["a.example.org"] == most["b.example.org"] == most["c.example.org"] == 1
    assert most["all"] > 1
