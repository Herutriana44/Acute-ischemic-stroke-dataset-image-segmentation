from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from flask import Flask, Response, abort, flash, jsonify, make_response, redirect, render_template, request, send_from_directory, url_for

from webapp.services.archive_service import extract_archive, find_dicom_series_dir
from webapp.services.inference_service import InferenceError, run_inference


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-key-change-me")

    root_dir = Path(__file__).resolve().parent
    uploads_dir = root_dir / "uploads"
    runs_dir = root_dir / "runs"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    app.config["UPLOAD_DIR"] = uploads_dir
    app.config["RUNS_DIR"] = runs_dir
    app.config["MODEL_PATH"] = root_dir.parent / "best_unet.pt"

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/runs/<run_id>/<path:filename>", methods=["GET"])
    def runs_file(run_id: str, filename: str):
        runs_dir: Path = app.config["RUNS_DIR"]
        safe_run_id = "".join([c for c in run_id if c.isalnum() or c in ("-", "_")])
        if safe_run_id != run_id:
            abort(404)
        run_dir = (runs_dir / run_id).resolve()
        if runs_dir.resolve() not in run_dir.parents:
            abort(404)
        send_kw: dict = dict(as_attachment=False, conditional=True, max_age=0)
        lower = filename.lower()
        if lower.endswith(".nii.gz"):
            send_kw["mimetype"] = "application/gzip"
        elif lower.endswith(".nii"):
            send_kw["mimetype"] = "application/octet-stream"
        elif lower.endswith(".json"):
            send_kw["mimetype"] = "application/json"
        elif lower.endswith(".ply"):
            send_kw["mimetype"] = "application/vnd.ply"
        elif lower.endswith(".zip"):
            send_kw["mimetype"] = "application/zip"
        elif lower.endswith(".obj"):
            send_kw["mimetype"] = "model/obj"
        elif lower.endswith(".mtl"):
            send_kw["mimetype"] = "model/mtl"

        resp = send_from_directory(run_dir, filename, **send_kw)
        # Make remote-viewer loads (ngrok) more reliable.
        resp = make_response(resp)
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    @app.route("/runs/<run_id>/dicom_manifest", methods=["GET"])
    def dicom_manifest(run_id: str):
        """Return JSON list of DICOM URLs for Papaya (no manual folder selection)."""
        runs_dir: Path = app.config["RUNS_DIR"]
        safe_run_id = "".join([c for c in run_id if c.isalnum() or c in ("-", "_")])
        if safe_run_id != run_id:
            abort(404)

        run_dir = (runs_dir / run_id).resolve()
        if runs_dir.resolve() not in run_dir.parents:
            abort(404)

        dicom_dir = run_dir / "dicom_series"
        if not dicom_dir.exists() or not dicom_dir.is_dir():
            abort(404)

        order_path = dicom_dir / "slice_order.json"
        files: list[Path]
        if order_path.is_file():
            try:
                spec = json.loads(order_path.read_text(encoding="utf-8"))
                names = spec.get("ordered_filenames") or []
                ordered = [dicom_dir / n for n in names if (dicom_dir / n).is_file()]
                seen = {p.name for p in ordered}
                extra = sorted(
                    [p for p in dicom_dir.glob("*.dcm") if p.is_file() and p.name not in seen]
                )
                files = ordered + extra
            except (json.JSONDecodeError, OSError):
                files = sorted([p for p in dicom_dir.glob("*.dcm") if p.is_file()])
        else:
            files = sorted([p for p in dicom_dir.glob("*.dcm") if p.is_file()])
        # Fallback: if extensions are missing, include all files.
        if not files:
            files = sorted([p for p in dicom_dir.iterdir() if p.is_file()])

        # IMPORTANT: return relative URLs to avoid mixed-content issues behind reverse proxies (e.g. ngrok).
        # The browser will resolve them with the current page's scheme (HTTPS).
        urls = [
            url_for(
                "runs_file",
                run_id=run_id,
                filename=f"dicom_series/{p.name}",
            )
            for p in files
        ]
        resp = jsonify({"run_id": run_id, "count": len(urls), "dicom_urls": urls})
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _safe_run_dir(run_id: str) -> Path | None:
        runs_dir: Path = app.config["RUNS_DIR"]
        safe_run_id = "".join([c for c in run_id if c.isalnum() or c in ("-", "_")])
        if safe_run_id != run_id:
            return None
        run_dir = (runs_dir / run_id).resolve()
        if runs_dir.resolve() not in run_dir.parents:
            return None
        return run_dir

    def _read_run_spacing(run_dir: Path) -> tuple[float, float, float]:
        """Spacing (ps_row, ps_col, ps_z) seperti di result.json inference."""
        rj = run_dir / "result.json"
        if rj.is_file():
            try:
                data = json.loads(rj.read_text(encoding="utf-8"))
                sp = data.get("spacing")
                if sp and len(sp) == 3:
                    return float(sp[0]), float(sp[1]), float(sp[2])
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass
        return 1.0, 1.0, 1.0

    def _vtk_load_volumes(
        run_dir: Path, max_dim: int | None
    ) -> tuple[np.ndarray, np.ndarray, int, tuple[float, float, float]] | None:
        hu_path = run_dir / "hu_volume.npy"
        mask_path = run_dir / "mask_pred.npy"
        if not hu_path.is_file() or not mask_path.is_file():
            return None
        hu = np.load(hu_path)
        mask = np.load(mask_path)
        ps_row, ps_col, ps_z = _read_run_spacing(run_dir)
        stride = 1
        limit = max_dim if max_dim and max_dim > 0 else None
        if limit is not None and max(hu.shape) > limit:
            stride = max(1, int(math.ceil(max(hu.shape) / limit)))
            hu = hu[::stride, ::stride, ::stride]
            mask = mask[::stride, ::stride, ::stride]
        eff = (ps_row * stride, ps_col * stride, ps_z * stride)
        return hu.astype(np.float32), mask.astype(np.uint8), stride, eff

    @app.route("/runs/<run_id>/vtk_meta", methods=["GET"])
    def vtk_meta(run_id: str):
        """Metadata untuk vtk.js: dimensi grid (X,Y,Z), spacing fisik (mm), stride downsample."""
        run_dir = _safe_run_dir(run_id)
        if run_dir is None:
            abort(404)
        try:
            max_dim = int(request.args.get("max", "192"))
        except ValueError:
            max_dim = 192
        loaded = _vtk_load_volumes(run_dir, max_dim)
        if loaded is None:
            abort(404)
        hu, _mask, stride, eff = loaded
        nz, ny, nx = int(hu.shape[0]), int(hu.shape[1]), int(hu.shape[2])
        ps_row, ps_col, ps_z = eff
        resp = jsonify(
            {
                "run_id": run_id,
                "dims_xyz": [nx, ny, nz],
                "spacing_xyz_mm": [ps_col, ps_row, ps_z],
                "stride": stride,
                "hu_dtype": "float32",
                "mask_dtype": "uint8",
            }
        )
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/runs/<run_id>/vtk_hu.bin", methods=["GET"])
    def vtk_hu_binary(run_id: str):
        run_dir = _safe_run_dir(run_id)
        if run_dir is None:
            abort(404)
        try:
            max_dim = int(request.args.get("max", "192"))
        except ValueError:
            max_dim = 192
        loaded = _vtk_load_volumes(run_dir, max_dim)
        if loaded is None:
            abort(404)
        hu, _mask, _stride, _eff = loaded
        body = memoryview(hu)
        resp = Response(body, mimetype="application/octet-stream")
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    @app.route("/runs/<run_id>/vtk_mask.bin", methods=["GET"])
    def vtk_mask_binary(run_id: str):
        run_dir = _safe_run_dir(run_id)
        if run_dir is None:
            abort(404)
        try:
            max_dim = int(request.args.get("max", "192"))
        except ValueError:
            max_dim = 192
        loaded = _vtk_load_volumes(run_dir, max_dim)
        if loaded is None:
            abort(404)
        _hu, mask, _stride, _eff = loaded
        body = memoryview(mask)
        resp = Response(body, mimetype="application/octet-stream")
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    @app.route("/predict", methods=["POST"])
    def predict():
        archive_file = request.files.get("dicom_archive")
        if archive_file is None or archive_file.filename == "":
            flash("Pilih file arsip DICOM CT terlebih dahulu.")
            return redirect(url_for("index"))

        try:
            extracted_dir, run_id = extract_archive(archive_file, app.config["UPLOAD_DIR"])
            series_dir = find_dicom_series_dir(extracted_dir)
            result = run_inference(
                dicom_dir=series_dir,
                run_id=run_id,
                model_path=app.config["MODEL_PATH"],
                runs_dir=app.config["RUNS_DIR"],
            )
        except InferenceError as exc:
            return render_template("error.html", error_message=str(exc)), 400
        except Exception as exc:  # pragma: no cover - guard for runtime issues
            return render_template("error.html", error_message=f"Gagal memproses data: {exc}"), 500

        return render_template("result.html", result=result)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
