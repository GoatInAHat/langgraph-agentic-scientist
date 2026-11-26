from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def derive() -> ModuleType:
    path = REPO_ROOT / "tools" / "derive_prompts.py"
    spec = importlib.util.spec_from_file_location("_tool_derive_extractors", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _extraction(derive: ModuleType, upstream: str, lines: str = "1", **fields: object) -> object:
    raw = {"upstream": upstream, "lines": lines, **fields}
    return derive._locator(raw, "test.toml", "upstream", "extracted/x.md")


def _text(
    derive: ModuleType, tmp_path: Path, file: str, source: str, *args: str, **fields: object
) -> str:
    (tmp_path / file).write_text(source, encoding="utf-8")
    value, _ = derive.text_of(_extraction(derive, file, *args, **fields), tmp_path)
    return value


def test_assignments_are_folded_and_a_dictionary_value_is_found_by_key(
    derive: ModuleType, tmp_path: Path
) -> None:
    source = 'A = "one " "two"\nB = "x" + A\nD: dict[str, str] = {"k": "v " + "w", "m": "n"}\n'
    assert _text(derive, tmp_path, "m.py", source, kind="assignment", name="A") == "one two"
    assert _text(derive, tmp_path, "m.py", source, kind="dict_value", name="D", key="k") == "v w"
    with pytest.raises(derive.DerivationError, match="no single key"):
        _text(derive, tmp_path, "m.py", source, kind="dict_value", name="D", key="z")
    with pytest.raises(derive.DerivationError, match="one module-level assignment"):
        _text(derive, tmp_path, "m.py", source, kind="assignment", name="Q")


def test_a_packed_assignment_is_decoded(derive: ModuleType, tmp_path: Path) -> None:
    import base64
    import bz2
    import json

    packed = base64.b85encode(bz2.compress(json.dumps({"review": "R"}).encode())).decode()
    source = f'PACKED = "{packed}"\n'
    value = _text(
        derive, tmp_path, "m.py", source, kind="assignment", name="PACKED", decode="base85-bz2"
    )
    assert json.loads(value) == {"review": "R"}
    value = _text(
        derive,
        tmp_path,
        "m.py",
        source,
        kind="assignment",
        name="PACKED",
        decode="base85-bz2",
        json_key="review",
    )
    assert value == "R"


def test_elements_fstrings_and_named_expressions_in_python(
    derive: ModuleType, tmp_path: Path
) -> None:
    source = (
        "def f(items, cats):\n"
        '    parts = ["a", "b: " + ", ".join(cats)]\n'
        '    return "\\n".join(f"- {item}" for item in items)\n'
    )
    names = {"', '.join(cats)": "CATS"}
    assert (
        _text(derive, tmp_path, "m.py", source, "2", kind="elements", names=names)
        == "a\nb: {{CATS}}"
    )
    assert _text(derive, tmp_path, "m.py", source, "3", kind="fstring") == "- {{ITEM}}"
    with pytest.raises(derive.DerivationError, match="name it"):
        _text(derive, tmp_path, "m.py", source, "2", kind="elements")


def test_embedded_python_snippets(derive: ModuleType, tmp_path: Path) -> None:
    source = (
        "python - <<'PY'\n"
        "    x = (\n"
        '        "one\\n"\n'
        '        f"two {name}"\n'
        "    )\n"
        '    print("p " f"{value}")\n'
        '    lines = ["l1", "l2"]\n'
        "PY\n"
    )
    assert _text(derive, tmp_path, "s.sh", source, "3-4", kind="expression") == "one\ntwo {{NAME}}"
    assert _text(derive, tmp_path, "s.sh", source, "6", kind="print") == "p {{VALUE}}"
    assert (
        _text(
            derive,
            tmp_path,
            "s.sh",
            source,
            "7",
            kind="elements",
            strip_prefix="lines = [",
            strip_suffix="]",
        )
        == "l1\nl2"
    )


def test_shell_strings_are_decoded_strictly(derive: ModuleType, tmp_path: Path) -> None:
    source = (
        'A="say \\"hi\\" \\$HOME and \\`tick\\` ${who#X:}"\n'
        "B='raw $x'\n"
        'C="$unnamed"\n'
        'D="`date`"\n'
        "E='it'\\''s'\n"
    )
    fields = {"kind": "shell_string", "strip_prefix": 'A="', "strip_suffix": '"'}
    assert (
        _text(derive, tmp_path, "s.sh", source, "1", names={"${who#X:}": "WHO"}, **fields)
        == 'say "hi" $HOME and `tick` {{WHO}}'
    )
    assert (
        _text(
            derive,
            tmp_path,
            "s.sh",
            source,
            "2",
            kind="shell_string",
            quote="single",
            strip_prefix="B='",
            strip_suffix="'",
        )
        == "raw $x"
    )
    for line, prefix, pattern in (
        ("3", 'C="', "name the expansion"),
        ("4", 'D="', "backtick"),
    ):
        with pytest.raises(derive.DerivationError, match=pattern):
            _text(
                derive,
                tmp_path,
                "s.sh",
                source,
                line,
                kind="shell_string",
                strip_prefix=prefix,
                strip_suffix='"',
            )
    with pytest.raises(derive.DerivationError, match="single quote"):
        _text(
            derive,
            tmp_path,
            "s.sh",
            source,
            "5",
            kind="shell_string",
            quote="single",
            strip_prefix="E='",
            strip_suffix="'",
        )
    with pytest.raises(derive.DerivationError, match="does not start with"):
        _text(derive, tmp_path, "s.sh", source, "1", kind="shell_string", strip_prefix="Z=")


def test_shell_appends_and_heredocs(derive: ModuleType, tmp_path: Path) -> None:
    source = (
        "p+=$'a\\n\\'b\\''\n"
        'p+="$d"\n'
        "p+=$'`\\n'\n"
        "cat <<'EOF'\n"
        "body $x\n"
        "EOF\n"
        "cat <<EOF\n"
        "unquoted\n"
        "EOF\n"
    )
    assert (
        _text(derive, tmp_path, "s.sh", source, "1-3", kind="shell_concat", names={"$d": "D"})
        == "a\n'b'{{D}}`\n"
    )
    assert _text(derive, tmp_path, "s.sh", source, "5", kind="heredoc") == "body $x\n"
    with pytest.raises(derive.DerivationError, match="quoted heredoc opener"):
        _text(derive, tmp_path, "s.sh", source, "8", kind="heredoc")
    with pytest.raises(derive.DerivationError, match="not an append"):
        _text(derive, tmp_path, "s.sh", source, "5", kind="shell_concat")
    with pytest.raises(derive.DerivationError, match="delimiter"):
        _text(derive, tmp_path, "s.sh", source, "5-6", kind="heredoc")


def test_an_unknown_kind_is_refused(derive: ModuleType) -> None:
    with pytest.raises(derive.DerivationError, match="unknown extraction kind"):
        _extraction(derive, "x.py", kind="magic")


def test_every_extracted_file_is_declared(derive: ModuleType) -> None:
    extractions, _ = derive.load_changes()
    declared = {item.output for item in extractions}
    present = {
        path.relative_to(derive.THIRD_PARTY).as_posix()
        for path in (derive.THIRD_PARTY / "extracted").rglob("*")
        if path.is_file() and path.suffix == ".md"
    }
    assert present == declared


def _copy_repo_subset(derive: ModuleType, tmp_path: Path) -> None:
    for name in ("THIRD_PARTY", "PACKAGE"):
        source = getattr(derive, name)
        target = tmp_path / source.relative_to(derive.REPO_ROOT)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        setattr(derive, name, target)
    derive.CHANGES_DIR = derive.PACKAGE / "prompts" / "changes"


def test_a_one_character_change_to_upstream_or_the_package_fails_the_check(
    derive: ModuleType, tmp_path: Path
) -> None:
    saved = (derive.THIRD_PARTY, derive.PACKAGE, derive.CHANGES_DIR)
    try:
        _copy_repo_subset(derive, tmp_path)
        assert derive.run(check=True) == []
        constant = derive.PACKAGE / "posthoc" / "rubric_contract.py"
        original = constant.read_text(encoding="utf-8")
        constant.write_text(original.replace("exists and explains", "exist and explains"))
        problems = derive.run(check=True)
        assert any("rubric_contract.py" in problem for problem in problems)
        constant.write_text(original)
        shell = derive.THIRD_PARTY / "src" / "scripts" / "paper.sh"
        text = shell.read_text(encoding="utf-8")
        shell.write_text(
            text.replace("is expected to be available", "is expected to be avaiable", 1)
        )
        problems = derive.run(check=True)
        assert any("openai_key_available" in problem for problem in problems)
    finally:
        derive.THIRD_PARTY, derive.PACKAGE, derive.CHANGES_DIR = saved
