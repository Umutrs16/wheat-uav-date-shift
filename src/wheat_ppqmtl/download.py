from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from wheat_ppqmtl.io_utils import write_json
from wheat_ppqmtl.paths import configure_stdio, configs, ensure_dir


def _urlopen(request, timeout: int, attempts: int = 5):
    last_error = None
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise last_error


def list_remote_files(dataset_id: str, version: str, api_root: str) -> list[dict]:
    found: dict[str, dict] = {}
    for token in ("UAV", "xls", "py", "txt", "GPR", "tif"):
        payload = json.dumps({"dataSetId": dataset_id, "version": version, "search": token}).encode()
        request = urllib.request.Request(
            api_root + "/gin-sdb-filetree/public/file/searchTreeList",
            data=payload,
            headers={"User-Agent": "wheat-ppqmtl/0.1", "Content-Type": "application/json"},
            method="POST",
        )
        with _urlopen(request, 60) as response:
            body = json.loads(response.read().decode("utf-8"))
        if body.get("code") != 20000:
            raise RuntimeError(f"ScienceDB file search failed: {body.get('messageEn')}")
        _walk(body.get("data") or {}, found)
    return list(found.values())


def _walk(node: dict, found: dict) -> None:
    if node.get("type") == "file" or node.get("dir") is False:
        found[node["path"]] = {
            "file_id": node.get("id"),
            "label": node.get("label"),
            "remote_path": node.get("path"),
            "size": int(node.get("size") or 0),
            "md5": node.get("signature") or "",
            "fcstr": node.get("fcstr") or "",
        }
    for child in node.get("children") or []:
        _walk(child, found)


def _md5_file(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(
    file_id: str,
    dest: Path,
    download_root: str,
    expected_md5: str = "",
    expected_size: int = 0,
    attempts: int = 3,
) -> None:
    """Download a ScienceDB file.

    The public mole API is a preview: GeoTIFFs come back as PNG and fail the
    published MD5. The site's own download button uses china.scidb.cn, which
    returns the file with the published Content-Length. curl is used because
    urllib's TLS handshake fails against that host on this machine.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if not expected_md5 or _md5_file(dest) == expected_md5:
            print(f"skip existing {dest.name}", flush=True)
            return
        print(f"existing {dest.name} failed MD5; downloading again", flush=True)
        dest.unlink()
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if not curl:
        raise RuntimeError("curl is required to download ScienceDB files")
    url = download_root + file_id
    temporary = dest.with_suffix(dest.suffix + ".part")
    log_path = dest.with_suffix(dest.suffix + ".curl.log")
    last_error: Exception | None = None
    for attempt in range(attempts):
        if attempt > 0 and temporary.exists() and expected_size and temporary.stat().st_size >= expected_size:
            temporary.unlink()
        command = [
            curl, "-L", "--http1.1", "--fail",
            "--retry", "8", "--retry-all-errors", "--retry-delay", "5",
            "--connect-timeout", "30",
            "-C", "-",
            "-A", "wheat-ppqmtl/0.1",
            "-o", str(temporary),
            url,
        ]
        print(f"download {dest.name} attempt {attempt + 1}/{attempts}", flush=True)
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log)
                while process.poll() is None:
                    size = temporary.stat().st_size if temporary.exists() else 0
                    target = f" / {expected_size / 1e9:.2f} GB" if expected_size else ""
                    print(f"  {dest.name} {size / 1e9:.2f} GB{target}", flush=True)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        continue
            if process.returncode != 0:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-500:]
                raise RuntimeError(f"curl exit {process.returncode}: {tail}")
            size = temporary.stat().st_size
            if expected_size and size != expected_size:
                if size > expected_size:
                    temporary.unlink(missing_ok=True)
                raise RuntimeError(f"size mismatch for {dest.name}: {size} != {expected_size}")
            digest = _md5_file(temporary)
            if expected_md5 and digest != expected_md5:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"MD5 mismatch for {dest.name}: {digest} != {expected_md5}")
            temporary.replace(dest)
            log_path.unlink(missing_ok=True)
            print(f"saved {dest.name} ({dest.stat().st_size} bytes)", flush=True)
            return
        except Exception as exc:
            last_error = exc
            print(f"  {dest.name} attempt {attempt + 1} failed: {exc}", flush=True)
            time.sleep(3 * (attempt + 1))
    raise last_error if last_error else RuntimeError(f"download failed for {dest.name}")


def local_path(root: Path, remote_path: str) -> Path:
    relative = remote_path.split("/UAV_LCC/", 1)[-1].lstrip("/")
    return root / "data" / "raw" / relative


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Download ScienceDB V4 wheat UAV files.")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Labels to download, for example observed_data.xlsx")
    args = parser.parse_args()
    cfg = configs()["data"]["dataset"]
    root = configs()["root"]
    try:
        files = list_remote_files(cfg["dataset_id"], cfg["version"], cfg["api_root"])
    except Exception as exc:
        existing = list((root / "data" / "raw").rglob("observed_data.xlsx"))
        if args.skip_images and existing:
            print(f"ScienceDB listing failed ({exc}). Continuing with files already in data/raw.")
            return
        raise
    manifest = root / "data" / "interim" / "remote_file_list.json"
    write_json(manifest, {"version": cfg["version"], "files": files})
    print(f"remote files: {len(files)}")
    files = sorted(files, key=lambda item: int(item["size"]))
    download_root = cfg.get("file_download_root", "https://china.scidb.cn/download?fileId=")
    for item in files:
        if args.only and item["label"] not in args.only:
            continue
        if args.skip_images and str(item["label"]).lower().endswith((".tif", ".tiff")):
            print(f"skip image {item['label']} ({item['size'] / 1e9:.2f} GB)")
            continue
        image = str(item["label"]).lower().endswith((".tif", ".tiff"))
        try:
            download_file(
                item["file_id"],
                local_path(root, item["remote_path"]),
                download_root,
                item["md5"],
                int(item["size"]),
            )
        except Exception as exc:
            print(f"download failed for {item['label']}: {exc}", flush=True)
            if image or item["label"] == "observed_data.xlsx":
                raise
    ensure_dir(root / "data" / "raw")


if __name__ == "__main__":
    main()
