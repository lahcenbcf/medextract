"""
MedExtract-IA: Zero-Disk Bunny.net Image Uploader

Streams in-memory image buffers directly to Bunny.net Storage Zone
without writing to disk. Returns CDN URLs for each uploaded image.
"""

import io
import requests
from typing import Optional


def upload_image_to_bunny(
    buffer: io.BytesIO,
    destination_path: str,
    storage_key: str,
    storage_zone: str = "nobles",
    cdn_host: str = "https://ziania-storage.b-cdn.net",
) -> Optional[str]:
    """
    Upload an in-memory image buffer to Bunny.net Storage Zone.

    Args:
        buffer: BytesIO buffer containing the image data
        destination_path: Path within the storage zone (e.g. 'qcm-images/42/img_1.png')
        storage_key: Bunny.net Storage Zone API Key
        storage_zone: Storage zone name
        cdn_host: CDN host for URL generation

    Returns:
        CDN URL of the uploaded image, or None on failure
    """
    url = f"https://storage.bunnycdn.com/{storage_zone}/{destination_path}"

    buffer.seek(0)
    data = buffer.read()

    # Detect content type from first bytes
    content_type = "image/png"  # default
    if data[:3] == b'\xff\xd8\xff':
        content_type = "image/jpeg"
    elif data[:8] == b'\x89PNG\r\n\x1a\n':
        content_type = "image/png"
    elif data[:4] == b'GIF8':
        content_type = "image/gif"
    elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        content_type = "image/webp"

    try:
        response = requests.put(
            url,
            data=data,
            headers={
                "AccessKey": storage_key,
                "Content-Type": content_type,
                "Content-Length": str(len(data)),
            },
            timeout=30,
        )

        if 200 <= response.status_code < 300:
            cdn_url = f"{cdn_host}/{destination_path}"
            return cdn_url
        else:
            print(f"[BunnyUpload] Failed to upload {destination_path}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"[BunnyUpload] Error uploading {destination_path}: {e}")
        return None


def upload_all_images(
    image_buffers: dict[str, io.BytesIO],
    storage_key: str,
    course_name: Optional[str] = None,
    job_id: Optional[int] = None,
    storage_zone: str = "nobles",
    cdn_host: str = "https://ziania-storage.b-cdn.net",
) -> dict[str, str]:
    """
    Upload all extracted images to Bunny.net.

    Args:
        image_buffers: Dict mapping image keys to BytesIO buffers
        storage_key: Bunny.net Storage Zone API Key
        course_name: Optional course name for folder organization
        job_id: Optional ingestion job ID fallback
        storage_zone: Storage zone name
        cdn_host: CDN host for URL generation

    Returns:
        Dict mapping image keys to CDN URLs
    """
    url_map: dict[str, str] = {}

    folder = course_name if course_name else (str(job_id) if job_id else "general")

    for key, buffer in image_buffers.items():
        # Detect extension from buffer content
        buffer.seek(0)
        header = buffer.read(12)
        buffer.seek(0)

        ext = "png"
        if header[:3] == b'\xff\xd8\xff':
            ext = "jpg"
        elif header[:4] == b'GIF8':
            ext = "gif"
        elif header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            ext = "webp"

        destination = f"qcm-images/{folder}/{key}.{ext}"

        cdn_url = upload_image_to_bunny(
            buffer=buffer,
            destination_path=destination,
            storage_key=storage_key,
            storage_zone=storage_zone,
            cdn_host=cdn_host,
        )

        if cdn_url:
            url_map[key] = cdn_url
            print(f"[BunnyUpload] ✓ {key} → {cdn_url}")
        else:
            print(f"[BunnyUpload] ✗ Failed: {key}")

    return url_map

