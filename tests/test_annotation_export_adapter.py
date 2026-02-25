from pathlib import Path
from types import SimpleNamespace

from src.adapters.annotation_export import AnnotationExportAdapter


def test_convert_annotations_returns_error_when_script_missing(tmp_path):
    adapter = AnnotationExportAdapter(project_path=str(tmp_path / "missing_project"))
    result = adapter.convert_annotations(
        input_dir=str(tmp_path / "in"),
        output_path=str(tmp_path / "out.jsonl"),
    )
    assert result["success"] is False
    assert "脚本不存在" in (result.get("error") or "")


def test_convert_annotations_builds_command_and_parses_output_path(tmp_path, monkeypatch):
    project = tmp_path / "project"
    script = project / "scripts" / "transformation" / "convert_annotations.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# stub\n", encoding="utf-8")

    captured = {}

    def _fake_run(cmd, cwd, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["timeout"] = timeout
        return SimpleNamespace(
            returncode=0,
            stdout="所有转换结果已保存到:\n/tmp/final_output.jsonl",
            stderr="",
        )

    monkeypatch.setattr("src.adapters.annotation_export.subprocess.run", _fake_run)
    monkeypatch.setattr("src.adapters.annotation_export.settings.USE_LOCAL_MODULES", True)
    monkeypatch.setattr("src.adapters.annotation_export.settings.PYTHON_UNIFIED", "/usr/bin/python3")

    adapter = AnnotationExportAdapter(project_path=str(project))
    result = adapter.convert_annotations(
        input_dir="/tmp/input",
        output_path="/tmp/out.jsonl",
        image_dir="/tmp/images",
        filename="P_1001.PV.json",
        model_family="chatts",
        csv_src_dir="/tmp/csv",
    )

    assert result["success"] is True
    assert result["output_path"] == "/tmp/final_output.jsonl"
    assert captured["cwd"] == str(project)
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["timeout"] == 600

    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/python3"
    assert Path(cmd[1]) == script
    assert "--input-dir" in cmd and "/tmp/input" in cmd
    assert "--image-dir" in cmd and "/tmp/images" in cmd
    assert "--output" in cmd and "/tmp/out.jsonl" in cmd
    assert "--format" in cmd and "chatts" in cmd
    assert "--csv-src" in cmd and "/tmp/csv" in cmd
    assert "--file" in cmd and "P_1001.PV.json" in cmd
