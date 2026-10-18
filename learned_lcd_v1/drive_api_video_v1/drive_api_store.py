"""Private, namespaced Drive API storage. Importing this module does no I/O.

Credentials live only in memory. No mount, model code, credential export, or
sharing-permission changes are performed. All API failures are sanitized.
"""
from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

CLIENT_ID = "722189933070-a4uva2qavhmps1unt1tb672uhdh71ed5.apps.googleusercontent.com"
EXPECTED_EMAIL_SHA256 = "3fd1b14a8da31a4adb823f9b444f59fb18027597e86eebfa3817a79925a79bbc"
SCOPE = "https://www.googleapis.com/auth/drive.file"
TOKEN_URI = "https://oauth2.googleapis.com/token"
NAMESPACE_KEY = "triumph_namespace"
NAMESPACE = "drive_api_video_v1"
FOLDER_MIME = "application/vnd.google-apps.folder"
CHUNK_BYTES = 8 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
API_RETRIES = 2
MAX_UPLOAD_CALLS_WITHOUT_PROGRESS = 3
FIELDS = "id,name,mimeType,size,sha256Checksum,webViewLink,webContentLink,createdTime,parents,appProperties,trashed"


class DriveStoreError(RuntimeError):
    """Contains only fixed messages; never API bodies, URLs or credentials."""


def _json_loads(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DriveStoreError("Duplicate JSON fields are not allowed.")
            result[key] = value
        return result
    try:
        return json.loads(data, object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except DriveStoreError:
        raise
    except Exception:
        raise DriveStoreError("Invalid JSON data.") from None


def _email_matches(value):
    try:
        return isinstance(value, str) and hashlib.sha256(value.strip().casefold().encode("utf-8")).hexdigest() == EXPECTED_EMAIL_SHA256
    except (UnicodeError, ValueError):
        return False


def _google_libraries():
    # Lazy imports keep static tooling/tests independent of Google packages.
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload, MediaIoBaseDownload
        from google_auth_httplib2 import AuthorizedHttp
        from httplib2 import Http
        return SimpleNamespace(Credentials=Credentials, Request=Request, build=build,
                               MediaFileUpload=MediaFileUpload, MediaIoBaseUpload=MediaIoBaseUpload,
                               MediaIoBaseDownload=MediaIoBaseDownload, AuthorizedHttp=AuthorizedHttp, Http=Http)
    except Exception:
        raise DriveStoreError("Google API dependencies are unavailable.") from None


def _name(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 240 or value in {".", ".."}
            or any(character in value for character in ("\x00", "\r", "\n", "/", "\\"))):
        raise DriveStoreError("Invalid Drive object name.")
    return value


def _file_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", value):
        raise DriveStoreError("Invalid Drive file identifier.")
    return value


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise DriveStoreError("Expected SHA256 must contain 64 hexadecimal characters.")
    return value.lower()


def _size(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriveStoreError("Expected byte count must be a nonnegative integer.")
    return value


def _quote(value):
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _local_hash(path):
    digest, size = hashlib.sha256(), 0
    try:
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                digest.update(chunk)
                size += len(chunk)
    except Exception:
        raise DriveStoreError("Cannot read the local input file.") from None
    return digest.hexdigest(), size


class _HashingSink:
    def __init__(self, destination, limit):
        self.destination, self.limit = destination, limit
        self.digest, self.size = hashlib.sha256(), 0

    def write(self, data):
        if self.size + len(data) > self.limit:
            raise DriveStoreError("Downloaded data exceeds its expected byte count.")
        if self.destination is not None:
            written = self.destination.write(data)
            if written != len(data):
                raise DriveStoreError("Local download write was incomplete.")
        self.digest.update(data)
        self.size += len(data)
        return len(data)


class DriveStore:
    """Use from_info/from_colab_secret; methods return owned Drive metadata.

    download_verified returns metadata plus local_path. Verified transfers add
    verifiedSha256 and verificationMethod; sha256Checksum remains the actual
    server value, or None when unavailable. read_json returns the JSON value.
    """
    def __init__(self, api, libraries):
        self._api, self._libraries = api, libraries

    def __repr__(self):
        return "DriveStore(namespace='drive_api_video_v1')"

    def __getstate__(self):
        raise TypeError("DriveStore contains an authenticated transport and cannot be serialized.")

    @classmethod
    def from_info(cls, info):
        if (not isinstance(info, dict) or info.get("type") != "authorized_user"
                or info.get("client_id") != CLIENT_ID or info.get("token_uri") != TOKEN_URI
                or info.get("scopes") != [SCOPE] or not _email_matches(info.get("verified_email"))
                or any(not isinstance(info.get(key), str) or not info[key] for key in ("refresh_token", "client_secret"))):
            raise DriveStoreError("Credential identity, endpoint, or exact scope does not match this application.")
        libraries = _google_libraries()
        # Do not forward arbitrary credential fields (quota project, universe,
        # pre-existing access token, etc.) from a supplied JSON object.
        authorized = {key: info[key] for key in ("refresh_token", "client_id", "client_secret")}
        authorized["scopes"] = [SCOPE]
        try:
            credentials = libraries.Credentials.from_authorized_user_info(authorized, scopes=[SCOPE])
            transport = libraries.Request()
            def bounded_request(*args, **kwargs):
                kwargs["timeout"] = 30
                return transport(*args, **kwargs)
            credentials.refresh(bounded_request)
            granted = credentials.granted_scopes
            if granted is not None and set(granted) != {SCOPE}:
                raise DriveStoreError("Refreshed credential has unexpected granted scopes.")
            transport_http = libraries.Http(timeout=30)
            # Match googleapiclient.http.build_http(): resumable uploads use
            # 308 Resume Incomplete without a redirect Location header.
            transport_http.redirect_codes = transport_http.redirect_codes - {308}
            http = libraries.AuthorizedHttp(credentials, http=transport_http, max_refresh_attempts=1)
            api = libraries.build("drive", "v3", http=http, cache_discovery=False, static_discovery=True)
        except DriveStoreError:
            raise
        except Exception:
            raise DriveStoreError("OAuth credential refresh or Drive client setup failed; details withheld.") from None
        store = cls(api, libraries)
        about = store._execute(lambda: api.about().get(fields="user(emailAddress,permissionId)"), "Drive account verification")
        user = about.get("user") if isinstance(about, dict) else None
        if not isinstance(user, dict) or not _email_matches(user.get("emailAddress")):
            raise DriveStoreError("Authenticated Drive account does not match the required account.")
        expected_id = info.get("drive_permission_id")
        if expected_id is not None and user.get("permissionId") != expected_id:
            raise DriveStoreError("Authenticated Drive account identifier does not match.")
        return store

    @classmethod
    def from_colab_secret(cls, name="TRIUMPH_DRIVE_OAUTH"):
        try:
            from google.colab import userdata
            secret = userdata.get(name)
        except Exception:
            raise DriveStoreError("The Colab OAuth secret is missing or this notebook cannot access it.") from None
        try:
            valid_secret = isinstance(secret, str) and len(secret.encode("utf-8")) <= 65536
        except UnicodeError:
            valid_secret = False
        if not valid_secret:
            raise DriveStoreError("The Colab OAuth secret has an invalid format.")
        info = _json_loads(secret)
        return cls.from_info(info)

    def _execute(self, factory, operation):
        try:
            return factory().execute(num_retries=API_RETRIES)
        except Exception:
            raise DriveStoreError(operation + " failed; API details withheld.") from None

    def _owned_metadata(self, value):
        if (not isinstance(value, dict) or not isinstance(value.get("appProperties"), dict)
                or value["appProperties"].get(NAMESPACE_KEY) != NAMESPACE or value.get("trashed") is True):
            raise DriveStoreError("Drive object is not an active object owned by this application.")
        result = {key: value[key] for key in FIELDS.split(",") if key in value}
        _file_id(result.get("id"))
        if not isinstance(result.get("name"), str) or not isinstance(result.get("mimeType"), str):
            raise DriveStoreError("Drive returned incomplete object metadata.")
        result.setdefault("sha256Checksum", None)
        result.setdefault("size", None)
        result.setdefault("webViewLink", None)
        return result

    def metadata(self, file_id):
        file_id = _file_id(file_id)
        result = self._execute(lambda: self._api.files().get(fileId=file_id, fields=FIELDS), "Drive metadata read")
        result = self._owned_metadata(result)
        if result["id"] != file_id:
            raise DriveStoreError("Drive returned an unexpected object identifier.")
        return result

    def _parent(self, parent_id):
        if parent_id is None:
            return "root"
        result = self.metadata(parent_id)
        if result["mimeType"] != FOLDER_MIME:
            raise DriveStoreError("The requested parent is not an application-owned folder.")
        return result["id"]

    def _list(self, parent_id, name=None):
        parent = self._parent(parent_id)
        query = ("trashed = false and '" + _quote(parent) + "' in parents and appProperties has { key='" +
                 NAMESPACE_KEY + "' and value='" + NAMESPACE + "' }")
        if name is not None:
            query += " and name = '" + _quote(_name(name)) + "'"
        rows, ids, tokens = [], set(), set()
        token = None
        while True:
            page = self._execute(lambda: self._api.files().list(q=query, pageSize=1000, pageToken=token,
                                 spaces="drive", fields="nextPageToken,files(" + FIELDS + ")"), "Drive object listing")
            if not isinstance(page, dict) or not isinstance(page.get("files", []), list):
                raise DriveStoreError("Drive returned an invalid listing.")
            for row in page.get("files", []):
                item = self._owned_metadata(row)
                if item["id"] in ids:
                    raise DriveStoreError("Drive listing returned duplicate object identifiers.")
                if parent_id is not None and parent_id not in item.get("parents", []):
                    raise DriveStoreError("Drive listing returned an object outside the requested parent.")
                if name is not None and item["name"] != name:
                    raise DriveStoreError("Drive listing returned an unexpected object name.")
                ids.add(item["id"])
                rows.append(item)
            token = page.get("nextPageToken")
            if not token:
                return rows
            if not isinstance(token, str) or token in tokens:
                raise DriveStoreError("Drive listing did not advance to a new page.")
            tokens.add(token)

    def list_files(self, parent_id):
        """List all active namespaced children, including job folders, on all pages."""
        if parent_id is None:
            raise DriveStoreError("An application-owned parent folder is required.")
        return self._list(parent_id)

    def _find(self, name, parent_id):
        matches = self._list(parent_id, name)
        if len(matches) > 1:
            raise DriveStoreError("Multiple application-owned objects have the same name; refusing ambiguous reuse.")
        return matches[0] if matches else None

    def ensure_folder(self, name, parent_id=None):
        name = _name(name)
        existing = self._find(name, parent_id)
        if existing is not None:
            if existing["mimeType"] != FOLDER_MIME:
                raise DriveStoreError("The requested folder name is already an application-owned file.")
            return existing
        parent = self._parent(parent_id)
        body = {"name": name, "mimeType": FOLDER_MIME, "appProperties": {NAMESPACE_KEY: NAMESPACE}, "parents": [parent]}
        created = self._execute(lambda: self._api.files().create(body=body, fields=FIELDS), "Drive folder creation")
        created = self._owned_metadata(created)
        if created["mimeType"] != FOLDER_MIME or created["name"] != name:
            raise DriveStoreError("Created Drive folder metadata does not match the request.")
        match = self._find(name, parent_id)
        if match is None or match["id"] != created["id"]:
            raise DriveStoreError("Created Drive folder cannot be identified unambiguously.")
        return match

    def find_file(self, name, parent_id):
        if parent_id is None:
            raise DriveStoreError("An application-owned parent folder is required.")
        result = self._find(_name(name), parent_id)
        if result is not None and result["mimeType"] == FOLDER_MIME:
            raise DriveStoreError("The requested file name is already an application-owned folder.")
        return result

    def _binary_metadata(self, metadata, expected_size=None, expected_sha=None):
        if metadata["mimeType"].startswith("application/vnd.google-apps."):
            raise DriveStoreError("Only binary Drive files support verified transfers.")
        raw_size = metadata.get("size")
        if isinstance(raw_size, bool) or not isinstance(raw_size, (str, int)) or not str(raw_size).isdigit():
            raise DriveStoreError("Drive did not provide a valid binary byte count.")
        size = int(raw_size)
        if expected_size is not None and size != expected_size:
            raise DriveStoreError("Drive byte count differs from the expected content.")
        remote_hash = metadata.get("sha256Checksum")
        if remote_hash is not None:
            remote_hash = _digest(remote_hash)
            if expected_sha is not None and remote_hash != expected_sha:
                raise DriveStoreError("Drive SHA256 differs from the expected content.")
        return size, remote_hash

    def _stream(self, file_id, destination, expected_size):
        sink = _HashingSink(destination, expected_size)
        try:
            media = self._api.files().get_media(fileId=file_id)
            download = self._libraries.MediaIoBaseDownload(sink, media, chunksize=CHUNK_BYTES)
            done, stalled, calls = False, 0, 0
            maximum_calls = 8 + 4 * ((expected_size + CHUNK_BYTES - 1) // CHUNK_BYTES)
            while not done:
                calls += 1
                if calls > maximum_calls:
                    raise DriveStoreError("Drive download exceeded its bounded transfer attempts.")
                before = sink.size
                _, done = download.next_chunk(num_retries=API_RETRIES)
                stalled = stalled + 1 if sink.size == before and not done else 0
                if stalled >= MAX_UPLOAD_CALLS_WITHOUT_PROGRESS:
                    raise DriveStoreError("Drive download made no progress.")
        except DriveStoreError:
            raise
        except Exception:
            raise DriveStoreError("Drive download failed; API details withheld.") from None
        if sink.size != expected_size:
            raise DriveStoreError("Downloaded byte count differs from the expected content.")
        return sink.digest.hexdigest()

    def _verify_remote(self, metadata, expected_sha, expected_size):
        _, server_hash = self._binary_metadata(metadata, expected_size, expected_sha)
        method = "server_sha256"
        if server_hash is None:
            if self._stream(metadata["id"], None, expected_size) != expected_sha:
                raise DriveStoreError("Downloaded remote content failed SHA256 verification.")
            method = "downloaded_sha256"
        return dict(metadata, verifiedSha256=expected_sha, verificationMethod=method)

    def _upload(self, media, parent_id, name, digest, size, mime_type):
        existing = self.find_file(name, parent_id)
        if existing is not None:
            return self._verify_remote(existing, digest, size)
        body = {"name": name, "parents": [parent_id], "mimeType": mime_type,
                "appProperties": {NAMESPACE_KEY: NAMESPACE}}
        try:
            upload = self._api.files().create(body=body, media_body=media, fields=FIELDS)
            response, previous, stalled, calls = None, -1, 0, 0
            maximum_calls = 8 + 4 * ((size + CHUNK_BYTES - 1) // CHUNK_BYTES)
            while response is None:
                calls += 1
                if calls > maximum_calls:
                    raise DriveStoreError("Drive upload exceeded its bounded transfer attempts.")
                status, response = upload.next_chunk(num_retries=API_RETRIES)
                progress = getattr(status, "resumable_progress", None)
                if response is None:
                    if isinstance(progress, int) and not 0 <= progress <= size:
                        raise DriveStoreError("Drive upload returned invalid byte progress.")
                    if not isinstance(progress, int) or progress <= previous:
                        stalled += 1
                    else:
                        stalled = 0
                        previous = progress
                    if stalled >= MAX_UPLOAD_CALLS_WITHOUT_PROGRESS:
                        raise DriveStoreError("Drive upload made no progress.")
        except DriveStoreError:
            raise
        except Exception:
            raise DriveStoreError("Drive upload failed; API details withheld. No verified result is claimed.") from None
        created = self._owned_metadata(response)
        current = self.metadata(created["id"])
        if current["name"] != name or parent_id not in current.get("parents", []):
            raise DriveStoreError("Uploaded object is outside its requested destination.")
        verified = self._verify_remote(current, digest, size)
        match = self.find_file(name, parent_id)
        if match is None or match["id"] != current["id"]:
            raise DriveStoreError("Uploaded object cannot be identified unambiguously.")
        return verified

    def upload_verified(self, path, parent_id, name=None):
        path = Path(path)
        name = _name(path.name if name is None else name)
        digest, size = _local_hash(path)
        mime_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        try:
            media = self._libraries.MediaFileUpload(str(path), mimetype=mime_type, chunksize=CHUNK_BYTES, resumable=True)
        except Exception:
            raise DriveStoreError("Cannot prepare the local file for upload.") from None
        return self._upload(media, parent_id, name, digest, size, mime_type)

    def upload_json(self, value, parent_id, name):
        name = _name(name)
        try:
            data = (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        except Exception:
            raise DriveStoreError("Cannot encode the JSON object.") from None
        if len(data) > MAX_JSON_BYTES:
            raise DriveStoreError("JSON object exceeds the permitted byte count.")
        stream = io.BytesIO(data)
        try:
            media = self._libraries.MediaIoBaseUpload(stream, mimetype="application/json", chunksize=CHUNK_BYTES, resumable=True)
        except Exception:
            raise DriveStoreError("Cannot prepare JSON for upload.") from None
        return self._upload(media, parent_id, name, hashlib.sha256(data).hexdigest(), len(data), "application/json")

    def download_verified(self, file_id, target, sha256, size):
        digest, size = _digest(sha256), _size(size)
        metadata = self.metadata(file_id)
        self._binary_metadata(metadata, size, digest)
        target = Path(target)
        temporary = target.with_name("." + target.name + ".part-" + uuid.uuid4().hex)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("xb") as destination:
                actual = self._stream(file_id, destination, size)
                destination.flush()
                os.fsync(destination.fileno())
            if actual != digest:
                raise DriveStoreError("Downloaded content failed SHA256 verification.")
            os.replace(temporary, target)
        except DriveStoreError:
            raise
        except Exception:
            raise DriveStoreError("Local verified download publication failed.") from None
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return dict(metadata, verifiedSha256=digest, verificationMethod="downloaded_sha256", local_path=str(target))

    def read_json(self, file_id):
        metadata = self.metadata(file_id)
        size, expected_hash = self._binary_metadata(metadata)
        if size > MAX_JSON_BYTES:
            raise DriveStoreError("Remote JSON exceeds the permitted byte count.")
        stream = io.BytesIO()
        actual = self._stream(file_id, stream, size)
        if expected_hash is not None and expected_hash != actual:
            raise DriveStoreError("Remote JSON content differs from its server checksum.")
        return _json_loads(stream.getvalue())
