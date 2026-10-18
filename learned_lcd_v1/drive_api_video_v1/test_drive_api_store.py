"""Fake API tests: no credentials, Google requests, models, or browser access."""
import copy
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import drive_api_store as storage

FAKE_EMAIL = "owner@example.test"
PRIVATE_SENTINEL = "PRIVATE_TOKEN_RESPONSE_URL_SENTINEL"


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Request:
    def __init__(self, api, callback):
        self.api, self.callback = api, callback

    def execute(self, num_retries=0):
        self.api.retry_values.append(num_retries)
        if self.api.fail_execute:
            raise RuntimeError(PRIVATE_SENTINEL)
        return self.callback()


class FakeDrive:
    def __init__(self):
        self.objects, self.contents, self.retry_values = {}, {}, []
        self.counter = 0
        self.email, self.permission_id = FAKE_EMAIL, "owner_id"
        self.page_size = 2
        self.fail_execute = self.fail_upload = self.fail_download_after_chunk = False
        self.omit_checksum = self.corrupt_upload = self.stall_upload = False
        self.media_reads = 0

    def add(self, name, data=b"", parent="root", *, folder=False, owned=True):
        self.counter += 1
        file_id = "id_" + str(self.counter)
        properties = {storage.NAMESPACE_KEY: storage.NAMESPACE} if owned else {}
        row = {"id": file_id, "name": name, "parents": [parent], "appProperties": properties,
               "mimeType": storage.FOLDER_MIME if folder else "application/octet-stream",
               "createdTime": "2026-10-18T00:00:00Z", "webViewLink": "https://example.test/view/" + file_id}
        if not folder:
            row["size"] = str(len(data))
            if not self.omit_checksum:
                row["sha256Checksum"] = digest(data)
            self.contents[file_id] = data
        self.objects[file_id] = row
        return copy.deepcopy(row)

    def files(self):
        return self

    def about(self):
        api = self
        return SimpleNamespace(get=lambda **kwargs: Request(api, lambda: {
            "user": {"emailAddress": api.email, "permissionId": api.permission_id}}))

    def get(self, *, fileId, fields):
        return Request(self, lambda: copy.deepcopy(self.objects[fileId]))

    def list(self, *, q, pageSize, pageToken, spaces, fields):
        def result():
            parent = re.search(r"'([^']*)' in parents", q).group(1)
            match = re.search(r" and name = '(.*)'$", q)
            name = match.group(1).replace("\\'", "'").replace("\\\\", "\\") if match else None
            rows = [copy.deepcopy(row) for row in self.objects.values()
                    if parent in row["parents"] and row["appProperties"].get(storage.NAMESPACE_KEY) == storage.NAMESPACE
                    and not row.get("trashed") and (name is None or row["name"] == name)]
            start = int(pageToken or 0)
            page = {"files": rows[start:start + self.page_size]}
            if start + self.page_size < len(rows):
                page["nextPageToken"] = str(start + self.page_size)
            return page
        return Request(self, result)

    def create(self, *, body, fields, media_body=None):
        if media_body is None:
            return Request(self, lambda: self.add(body["name"], parent=body["parents"][0], folder=True))
        api = self
        class Upload:
            def next_chunk(self, num_retries=0):
                api.retry_values.append(num_retries)
                if api.fail_upload:
                    raise RuntimeError(PRIVATE_SENTINEL)
                if api.stall_upload:
                    return SimpleNamespace(resumable_progress=0), None
                data = media_body.data
                if api.corrupt_upload:
                    data = bytes([data[0] ^ 1]) + data[1:] if data else b"corrupt"
                created = api.add(body["name"], data, body["parents"][0])
                api.objects[created["id"]]["mimeType"] = body["mimeType"]
                return None, copy.deepcopy(api.objects[created["id"]])
        return Upload()

    def get_media(self, *, fileId):
        self.media_reads += 1
        return SimpleNamespace(api=self, file_id=fileId)


class FakeDownload:
    def __init__(self, sink, media, chunksize):
        self.sink, self.media, self.offset = sink, media, 0

    def next_chunk(self, num_retries=0):
        api = self.media.api
        api.retry_values.append(num_retries)
        if api.fail_download_after_chunk and self.offset:
            raise RuntimeError(PRIVATE_SENTINEL)
        data = api.contents[self.media.file_id]
        chunk = data[self.offset:self.offset + 3]
        self.sink.write(chunk)
        self.offset += len(chunk)
        return None, self.offset >= len(data)


def fake_libraries(api):
    credentials = SimpleNamespace(granted_scopes=[storage.SCOPE], refresh=mock.Mock())
    credentials_factory = mock.Mock(return_value=credentials)
    return SimpleNamespace(
        Credentials=SimpleNamespace(from_authorized_user_info=credentials_factory),
        Request=mock.Mock(return_value=mock.Mock()),
        AuthorizedHttp=mock.Mock(return_value=object()),
        Http=mock.Mock(return_value=SimpleNamespace(redirect_codes={301, 302, 303, 307, 308})),
        build=mock.Mock(return_value=api),
        MediaFileUpload=lambda path, **kwargs: SimpleNamespace(data=Path(path).read_bytes()),
        MediaIoBaseUpload=lambda stream, **kwargs: SimpleNamespace(data=stream.getvalue()),
        MediaIoBaseDownload=FakeDownload, credentials=credentials)


def fake_info():
    return {"type": "authorized_user", "client_id": storage.CLIENT_ID, "client_secret": "DUMMY_SECRET",
            "refresh_token": "DUMMY_REFRESH", "scopes": [storage.SCOPE], "token_uri": storage.TOKEN_URI,
            "verified_email": FAKE_EMAIL, "drive_permission_id": "owner_id"}


class DriveStoreTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeDrive()
        self.libs = fake_libraries(self.api)
        self.store = storage.DriveStore(self.api, self.libs)
        self.folder = self.api.add("jobs", folder=True)["id"]

    def test_resumable_upload_308_is_not_treated_as_redirect(self):
        self.assertIn(308, self.libs.Http.return_value.redirect_codes)
        with mock.patch.object(storage, "EXPECTED_EMAIL_SHA256", digest(FAKE_EMAIL.encode())), \
             mock.patch.object(storage, "_google_libraries", return_value=self.libs):
            storage.DriveStore.from_info(fake_info())
        self.libs.Http.assert_called_once_with(timeout=30)
        http = self.libs.AuthorizedHttp.call_args.kwargs["http"]
        self.assertIs(http, self.libs.Http.return_value)
        self.assertEqual(http.redirect_codes, {301, 302, 303, 307})
        self.assertEqual(self.libs.AuthorizedHttp.call_args.kwargs["max_refresh_attempts"], 1)

    def test_from_info_filters_extra_fields_and_rechecks_account(self):
        info = fake_info()
        info.update(token="DUMMY_OLD_ACCESS", universe_domain="untrusted.example", quota_project_id="untrusted")
        with mock.patch.object(storage, "EXPECTED_EMAIL_SHA256", digest(FAKE_EMAIL.encode())), \
             mock.patch.object(storage, "_google_libraries", return_value=self.libs):
            store = storage.DriveStore.from_info(info)
        passed = self.libs.Credentials.from_authorized_user_info.call_args.args[0]
        self.assertEqual(set(passed), {"refresh_token", "client_id", "client_secret", "scopes"})
        self.libs.credentials.refresh.assert_called_once()
        self.assertNotIn("DUMMY", repr(store))
        with self.assertRaises(TypeError):
            pickle.dumps(store)

    def test_wrong_account_scope_and_endpoint_fail_before_access(self):
        with mock.patch.object(storage, "EXPECTED_EMAIL_SHA256", digest(FAKE_EMAIL.encode())), \
             mock.patch.object(storage, "_google_libraries", return_value=self.libs):
            for key, value in (("client_id", "other"), ("token_uri", "https://untrusted.example"),
                               ("scopes", [storage.SCOPE, "https://www.googleapis.com/auth/drive"]),
                               ("verified_email", "other@example.test")):
                invalid = fake_info()
                invalid[key] = value
                with self.assertRaises(storage.DriveStoreError):
                    storage.DriveStore.from_info(invalid)
            self.libs.Credentials.from_authorized_user_info.assert_not_called()
            self.api.email = "other@example.test"
            with self.assertRaisesRegex(storage.DriveStoreError, "Authenticated Drive account"):
                storage.DriveStore.from_info(fake_info())
            self.api.email = FAKE_EMAIL
            self.libs.credentials.granted_scopes = [storage.SCOPE, "other"]
            with self.assertRaisesRegex(storage.DriveStoreError, "unexpected granted scopes"):
                storage.DriveStore.from_info(fake_info())

    def test_refresh_and_api_failures_are_sanitized(self):
        self.libs.credentials.refresh.side_effect = RuntimeError(PRIVATE_SENTINEL)
        with mock.patch.object(storage, "EXPECTED_EMAIL_SHA256", digest(FAKE_EMAIL.encode())), \
             mock.patch.object(storage, "_google_libraries", return_value=self.libs):
            with self.assertRaises(storage.DriveStoreError) as failure:
                storage.DriveStore.from_info(fake_info())
        self.assertNotIn(PRIVATE_SENTINEL, str(failure.exception))
        self.assertIsNone(failure.exception.__cause__)
        self.api.fail_execute = True
        with self.assertRaises(storage.DriveStoreError) as failure:
            self.store.metadata(self.folder)
        self.assertNotIn(PRIVATE_SENTINEL, str(failure.exception))
        self.assertTrue(failure.exception.__suppress_context__)

    def test_colab_secret_is_private_and_failure_is_sanitized(self):
        google = ModuleType("google")
        google.__path__ = []
        colab = ModuleType("google.colab")
        colab.userdata = SimpleNamespace(get=mock.Mock(return_value=json.dumps(fake_info())))
        google.colab = colab
        with mock.patch.dict(sys.modules, {"google": google, "google.colab": colab}), \
             mock.patch.object(storage.DriveStore, "from_info", return_value=self.store) as factory:
            self.assertIs(storage.DriveStore.from_colab_secret(), self.store)
            colab.userdata.get.assert_called_once_with("TRIUMPH_DRIVE_OAUTH")
            factory.assert_called_once_with(fake_info())
            colab.userdata.get.side_effect = RuntimeError(PRIVATE_SENTINEL)
            with self.assertRaises(storage.DriveStoreError) as failure:
                storage.DriveStore.from_colab_secret()
            self.assertNotIn(PRIVATE_SENTINEL, str(failure.exception))
            self.assertTrue(failure.exception.__suppress_context__)
            colab.userdata.get.side_effect = None
            colab.userdata.get.return_value = "\ud800"
            with self.assertRaisesRegex(storage.DriveStoreError, "invalid format"):
                storage.DriveStore.from_colab_secret()

    def test_owned_folder_reuse_does_not_adopt_arbitrary_same_name(self):
        foreign = self.api.add("assets", folder=True, owned=False)
        owned = self.store.ensure_folder("assets")
        self.assertNotEqual(owned["id"], foreign["id"])
        self.assertEqual(self.store.ensure_folder("assets")["id"], owned["id"])
        with self.assertRaises(storage.DriveStoreError):
            self.store.ensure_folder("child", foreign["id"])
        self.api.add("assets", folder=True)
        with self.assertRaisesRegex(storage.DriveStoreError, "Multiple"):
            self.store.ensure_folder("assets")

    def test_listing_reads_every_page_and_retains_folder_metadata(self):
        for index in range(5):
            self.api.add("job_" + str(index), parent=self.folder, folder=index % 2 == 0)
        self.api.add("foreign", parent=self.folder, owned=False)
        rows = self.store.list_files(self.folder)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all("createdTime" in row and "mimeType" in row for row in rows))
        self.assertEqual(self.store.find_file("job_1", self.folder)["name"], "job_1")
        self.assertIsNone(self.store.find_file("missing", self.folder))
        self.assertTrue(all(retries == storage.API_RETRIES for retries in self.api.retry_values))

    def test_upload_is_immutable_and_reuses_only_matching_content(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "video.mp4"
            source.write_bytes(b"example video")
            first = self.store.upload_verified(source, self.folder)
            self.assertEqual(first["verifiedSha256"], digest(b"example video"))
            self.assertEqual(first["verificationMethod"], "server_sha256")
            self.assertEqual(self.store.upload_verified(source, self.folder)["id"], first["id"])
            source.write_bytes(b"different data")
            with self.assertRaises(storage.DriveStoreError):
                self.store.upload_verified(source, self.folder)
            self.assertEqual(self.api.contents[first["id"]], b"example video")

    def test_missing_server_checksum_requires_remote_stream_not_app_property(self):
        self.api.omit_checksum = True
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "file.bin"
            source.write_bytes(b"123456789")
            result = self.store.upload_verified(source, self.folder)
            self.assertIsNone(result["sha256Checksum"])
            self.assertEqual(result["verificationMethod"], "downloaded_sha256")
            self.assertGreater(self.api.media_reads, 0)
            self.api.objects[result["id"]]["appProperties"]["sha256"] = digest(b"123456789")
            self.api.contents[result["id"]] = b"CORRUPTED"
            with self.assertRaisesRegex(storage.DriveStoreError, "SHA256"):
                self.store.upload_verified(source, self.folder)

    def test_upload_corruption_and_partial_failure_never_claim_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "file.bin"
            source.write_bytes(b"123456")
            self.api.corrupt_upload = True
            with self.assertRaisesRegex(storage.DriveStoreError, "SHA256"):
                self.store.upload_verified(source, self.folder)
            self.api.corrupt_upload = False
            self.api.fail_upload = True
            with self.assertRaises(storage.DriveStoreError) as failure:
                self.store.upload_verified(source, self.folder, name="partial.bin")
            self.assertNotIn(PRIVATE_SENTINEL, str(failure.exception))
            self.api.fail_upload = False
            self.api.stall_upload = True
            with self.assertRaisesRegex(storage.DriveStoreError, "no progress"):
                self.store.upload_verified(source, self.folder, name="stalled.bin")

    def test_atomic_download_preserves_old_target_on_corruption_or_interruption(self):
        data = b"123456789"
        row = self.api.add("download.bin", data, self.folder)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "download.bin"
            target.write_bytes(b"previous valid local file")
            self.api.contents[row["id"]] = b"CORRUPTED"
            with self.assertRaisesRegex(storage.DriveStoreError, "SHA256"):
                self.store.download_verified(row["id"], target, digest(data), len(data))
            self.assertEqual(target.read_bytes(), b"previous valid local file")
            self.api.contents[row["id"]] = data
            self.api.fail_download_after_chunk = True
            with self.assertRaises(storage.DriveStoreError) as failure:
                self.store.download_verified(row["id"], target, digest(data), len(data))
            self.assertNotIn(PRIVATE_SENTINEL, str(failure.exception))
            self.assertEqual(target.read_bytes(), b"previous valid local file")
            self.assertEqual(list(Path(directory).glob("*.part-*")), [])
            self.api.fail_download_after_chunk = False
            result = self.store.download_verified(row["id"], target, digest(data), len(data))
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(result["local_path"], str(target))

    def test_json_roundtrip_is_immutable_and_rejects_invalid_values(self):
        value = {"job_id": "example", "source": {"bytes": 3}}
        # The fake uses tiny chunks; increase its per-download size allowance for
        # this JSON fixture without changing the real transport chunk policy.
        original = storage.CHUNK_BYTES
        with mock.patch.object(storage, "CHUNK_BYTES", 3):
            row = self.store.upload_json(value, self.folder, "owner's-job.json")
            self.assertEqual(self.store.read_json(row["id"]), value)
            self.assertEqual(self.store.upload_json(value, self.folder, "owner's-job.json")["id"], row["id"])
            with self.assertRaises(storage.DriveStoreError):
                self.store.upload_json({"job_id": "changed"}, self.folder, "owner's-job.json")
        self.assertEqual(storage.CHUNK_BYTES, original)
        with self.assertRaises(storage.DriveStoreError):
            self.store.upload_json({"value": float("nan")}, self.folder, "invalid.json")

    def test_oversized_and_duplicate_key_json_reject(self):
        row = self.api.add("bad.json", b'{"x":1,"x":2}', self.folder)
        with self.assertRaises(storage.DriveStoreError):
            self.store.read_json(row["id"])
        self.api.objects[row["id"]]["size"] = str(storage.MAX_JSON_BYTES + 1)
        with self.assertRaisesRegex(storage.DriveStoreError, "permitted byte count"):
            self.store.read_json(row["id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
