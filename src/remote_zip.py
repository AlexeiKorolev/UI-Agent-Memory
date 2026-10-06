"""Extract selected files from the GUIOdyssey v2 *split* zip on Hugging Face via HTTP range reads.

The v2 screenshots ship as screenshots.z01..z08 + screenshots.zip (92.6 GB). We only need a few
thousand PNGs, so we parse the (ZIP64) central directory from the last part and range-read each
needed member from the right part (members may straddle part boundaries).
"""
import io
import json
import os
import struct
import time
import zlib
from pathlib import Path

import requests

REPO = "hflqf88888/GUIOdyssey"
REV = "61632d0f3f4d51d7e9561ce4f84347dd06b2019d"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/screenshots/"
PARTS = [f"screenshots.z{i:02d}" for i in range(1, 9)] + ["screenshots.zip"]


class SplitZip:
    def __init__(self, session=None, base=BASE, parts=PARTS, sizes=None):
        self.s = session or requests.Session()
        self.base, self.parts = base, parts
        # exact part sizes from the HF tree API at REV (avoids per-part resolver HEAD requests)
        self.sizes = sizes or [10737418240] * 8 + [6740164517]
        self._cdn = {}  # part -> (resolved CDN url, time); avoids a hub request per range read

    def _url(self, disk, refresh=False):
        u, t = self._cdn.get(disk, (None, 0))
        if refresh or u is None or time.time() - t > 600:
            r = self.s.head(self.base + self.parts[disk], allow_redirects=False, timeout=60)
            u = r.headers.get("Location") if r.status_code in (301, 302, 303, 307, 308) else self.base + self.parts[disk]
            if u.startswith("/"):
                u = "https://huggingface.co" + u
            self._cdn[disk] = (u, time.time())
        return u

    def _size(self, part):
        r = self.s.head(self.base + part, allow_redirects=True, timeout=60)
        r.raise_for_status()
        return int(r.headers["Content-Length"])

    def _get(self, disk, start, length):
        """Read `length` bytes starting at offset `start` of part `disk`, continuing into later parts."""
        out = b""
        while length > 0:
            avail = self.sizes[disk] - start
            if avail <= 0:
                disk, start = disk + 1, start - self.sizes[disk]
                continue
            n = min(length, avail)
            for attempt in range(12):
                try:
                    r = self.s.get(self._url(disk, refresh=attempt > 1),
                                   headers={"Range": f"bytes={start}-{start + n - 1}"},
                                   allow_redirects=True, timeout=120)
                    r.raise_for_status()
                    assert len(r.content) == n, (len(r.content), n)
                    break
                except Exception as exc:
                    if attempt == 11:
                        raise
                    # 429 rate limiting from the HF hub: exponential backoff
                    time.sleep(min(300, 5 * 2 ** attempt))
            out += r.content
            length -= n
            disk, start = disk + 1, 0
        return out

    def central_directory(self):
        last = len(self.parts) - 1
        tail_len = min(self.sizes[last], 1 << 16)
        tail = self._get(last, self.sizes[last] - tail_len, tail_len)
        i = tail.rfind(b"PK\x05\x06")
        assert i >= 0, "EOCD not found"
        (_, disk_no, cd_disk, n_this, n_total, cd_size, cd_off, _) = struct.unpack("<IHHHHIIH", tail[i:i + 22])
        if cd_off == 0xFFFFFFFF or n_total == 0xFFFF or cd_disk == 0xFFFF:
            j = tail.rfind(b"PK\x06\x07")  # zip64 locator
            _, z64_disk, z64_off, _ = struct.unpack("<IIQI", tail[j:j + 20])
            rec = self._get(z64_disk, z64_off, 56)
            assert rec[:4] == b"PK\x06\x06"
            (_, _, _, _, disk_no, cd_disk, n_this, n_total, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", rec)
        cd = self._get(cd_disk, cd_off, cd_size)
        entries = {}
        p = 0
        while p < len(cd):
            assert cd[p:p + 4] == b"PK\x01\x02", p
            (_, _, _, flag, method, _, _, crc, csize, usize, nlen, xlen, clen, disk_start, _, _, lho) = \
                struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode("utf8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
            # zip64 extra field
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    vals = extra[q + 4:q + 4 + hlen]
                    k = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                    if disk_start == 0xFFFF:
                        disk_start = struct.unpack("<I", vals[k:k + 4])[0]; k += 4
                q += 4 + hlen
            entries[os.path.basename(name)] = dict(name=name, method=method, csize=csize, usize=usize,
                                                  disk=disk_start, off=lho, crc=crc, flag=flag)
            p += 46 + nlen + xlen + clen
        return entries

    def read_member(self, e):
        # one range read covering local header + (guessed) name/extra + data; top up if extra is larger
        guess = 30 + len(e["name"].encode()) + 64
        buf = self._get(e["disk"], e["off"], guess + e["csize"])
        assert buf[:4] == b"PK\x03\x04"
        nlen, xlen = struct.unpack("<HH", buf[26:30])
        need = 30 + nlen + xlen + e["csize"]
        if need > len(buf):
            disk, off = e["disk"], e["off"] + len(buf)
            while off >= self.sizes[disk]:
                off -= self.sizes[disk]; disk += 1
            buf += self._get(disk, off, need - len(buf))
        raw = buf[30 + nlen + xlen:need]
        if e["method"] == 0:
            data = raw
        elif e["method"] == 8:
            data = zlib.decompress(raw, -15)
        else:
            raise ValueError(f"unsupported compression {e['method']}")
        assert zlib.crc32(data) & 0xFFFFFFFF == e["crc"], "CRC mismatch"
        return data


def load_index(cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.load(open(cache_path))
    z = SplitZip()
    idx = z.central_directory()
    json.dump(idx, open(cache_path, "w"))
    return idx


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ThreadPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="text file with screenshot basenames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    idx = load_index(a.index)
    print("central directory entries:", len(idx))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = [l.strip() for l in open(a.files) if l.strip()]
    todo = [n for n in names if not (out / n).exists()]
    missing = [n for n in todo if n not in idx]
    print(f"requested {len(names)}, todo {len(todo)}, missing-from-zip {len(missing)}")
    z = SplitZip()

    def job(n):
        if n not in idx:
            return n, False
        data = z.read_member(idx[n])
        tmp = out / (n + ".tmp")
        tmp.write_bytes(data); tmp.rename(out / n)
        return n, True

    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for n, ok in ex.map(job, todo):
            done += ok
            if done % 500 == 0:
                print("downloaded", done, flush=True)
    print("done", done, "missing", missing[:20])
