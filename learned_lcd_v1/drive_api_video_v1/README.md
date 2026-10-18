# Drive API video test

This is a storage migration around the existing LCD9 + speed11 + RPM9 diagnostic. It preserves the frozen evaluator and batch sizes (64 / 16 / 128); it does not retrain, change thresholds or claim higher accuracy.

1. Add the authorized-user JSON to Colab Secrets as `TRIUMPH_DRIVE_OAUTH` and grant access to this notebook. Credentials are not notebook code or output.
2. Start on CPU. Run the setup check, then disable `SETUP_CHECK_ONLY` to browse and save a video.
3. Switch to T4 and run cell 2. A blank job folder ID chooses the newest completed input upload. Every selected source frame goes to all three models with the existing full diagnostic labels.
4. Results are uploaded and verified before the final receipt is published and the GPU is released. Download the video using the private Drive link, without a runtime.

The API client requests only `drive.file` and checks the intended account, client and scope. It operates in its own namespaced private folder; it does not adopt arbitrary pre-existing folders or change sharing. Private assets must first be migrated by the local setup helper. Public code contains no private file IDs, videos, weights, refresh tokens or plaintext account email.

Transfers use resumable uploads and verify actual server checksums or a streamed download SHA256. On upload failure, outputs stay in the runtime. Rerunning cell 2 retries a recorded pending publication without invoking the models. Preserve the printed runtime folder manually if local publication failed before a recovery pointer was written. Original assets and earlier checkpoints are never deleted.

The current External/Testing OAuth grant lasts about seven days. Colab Secrets prevents repeated mount dialogs during that grant; it cannot bypass renewal, revocation, account limits or Google consent. Refresh credentials are security-sensitive: enable notebook access only for reviewed code.

The existing upright LCD corner assumption, unmasked speed input and raw ungated readings remain diagnostic limitations. Inspect model accuracy separately from throughput. Reports distinguish reader FPS, decode/inference/render/encode FPS, playback FPS and elapsed evaluation time; Drive transfer time is additional.

`build_notebook.py --code-commit <full SHA>` builds two code cells and runtime-free download instructions. All six Python runtime files are pinned by SHA256 and byte count. Imports do no network or model work. Offline tests use fake Drive objects; training and video inference are Colab-only.

References: [Drive resumable uploads](https://developers.google.com/workspace/drive/api/guides/manage-uploads), [Drive scopes](https://developers.google.com/workspace/drive/api/guides/api-specific-auth), [Colab Secrets API](https://github.com/googlecolab/colabtools/blob/main/google/colab/userdata.py).
