from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, abort, flash, make_response, redirect, render_template, request, send_from_directory, url_for

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
        resp = send_from_directory(run_dir, filename, as_attachment=False, conditional=True, max_age=0)
        # Make remote-viewer loads (ngrok) more reliable.
        resp = make_response(resp)
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    @app.route("/predict", methods=["POST"])
    def predict():
        archive_file = request.files.get("dicom_archive")
        if archive_file is None or archive_file.filename == "":
            flash("Pilih file archive DICOM terlebih dahulu.")
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
