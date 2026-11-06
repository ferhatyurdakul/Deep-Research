"""
The one-time carry-over from the former `deep-research` name.

Every test points the XDG dirs at a temp folder and uses the in-memory keyring
from conftest, so nothing here can see the developer's real files or keys.
"""
from __future__ import annotations

import pytest

from synapse_deep_research import entry, legacy_install
from synapse_deep_research.legacy_install import MARKER, NEW_NAME, OLD_NAME


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """(config_home, data_home) with the state overrides cleared, as for a real launch."""
    for name in legacy_install._OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "config", tmp_path / "data"


def _old_install(config_home, data_home):
    old_config, old_data = config_home / OLD_NAME, data_home / OLD_NAME
    old_config.mkdir(parents=True)
    old_data.mkdir(parents=True)
    (old_config / "config.json").write_text('{"provider": "opencode-go"}')
    (old_config / ".env").write_text("TAVILY_API_KEY=tvly-old\n")
    (old_config / "repl_history").write_text("+/depth deep\n")
    (old_data / "research.db").write_bytes(b"SQLite format 3\x00old sessions")
    (old_data / "research.db-wal").write_bytes(b"wal")
    return old_config, old_data


class TestCarryOver:
    def test_copies_settings_history_and_database(self, homes):
        config_home, data_home = homes
        _old_install(config_home, data_home)

        carried = legacy_install.carry_over(config_home, data_home)

        new_config, new_data = config_home / NEW_NAME, data_home / NEW_NAME
        assert (new_config / "config.json").read_text() == '{"provider": "opencode-go"}'
        assert (new_config / ".env").read_text() == "TAVILY_API_KEY=tvly-old\n"
        assert (new_config / "repl_history").read_text() == "+/depth deep\n"
        assert (new_data / "research.db").read_bytes() == b"SQLite format 3\x00old sessions"
        assert (new_data / "research.db-wal").read_bytes() == b"wal"
        assert carried and len(carried.files) == 5

    def test_the_old_files_are_left_exactly_as_they_were(self, homes):
        config_home, data_home = homes
        old_config, old_data = _old_install(config_home, data_home)

        legacy_install.carry_over(config_home, data_home)

        assert (old_config / "config.json").read_text() == '{"provider": "opencode-go"}'
        assert (old_data / "research.db").read_bytes() == b"SQLite format 3\x00old sessions"
        assert sorted(p.name for p in old_config.iterdir()) == [".env", "config.json", "repl_history"]
        assert sorted(p.name for p in old_data.iterdir()) == ["research.db", "research.db-wal"]

    def test_never_overwrites_what_the_new_name_already_has(self, homes):
        config_home, data_home = homes
        _old_install(config_home, data_home)
        new_config = config_home / NEW_NAME
        new_config.mkdir(parents=True)
        (new_config / "config.json").write_text('{"provider": "zai"}')

        legacy_install.carry_over(config_home, data_home)

        assert (new_config / "config.json").read_text() == '{"provider": "zai"}'
        assert (new_config / ".env").exists()  # the files it did not have are still filled in

    def test_an_existing_session_database_is_not_replaced_or_spliced(self, homes):
        config_home, data_home = homes
        _old_install(config_home, data_home)
        new_data = data_home / NEW_NAME
        new_data.mkdir(parents=True)
        (new_data / "research.db").write_bytes(b"live sessions")

        legacy_install.carry_over(config_home, data_home)

        assert (new_data / "research.db").read_bytes() == b"live sessions"
        assert not (new_data / "research.db-wal").exists()  # an old WAL must not meet a different database

    def test_runs_once(self, homes):
        config_home, data_home = homes
        _old_install(config_home, data_home)
        assert legacy_install.carry_over(config_home, data_home)
        assert (config_home / NEW_NAME / MARKER).exists()

        (config_home / OLD_NAME / "extra.json").write_text("{}")
        (config_home / OLD_NAME / "config.json").write_text('{"provider": "changed-later"}')
        (config_home / NEW_NAME / "config.json").unlink()

        again = legacy_install.carry_over(config_home, data_home)

        assert not again
        assert not (config_home / NEW_NAME / "config.json").exists()  # not re-copied

    def test_a_fresh_machine_has_nothing_to_carry_and_says_nothing(self, homes):
        config_home, data_home = homes

        carried = legacy_install.carry_over(config_home, data_home)

        assert not carried and carried.files == [] and carried.keys == []
        assert (config_home / NEW_NAME / MARKER).exists()  # so it does not look again next launch
        assert not (data_home / NEW_NAME).exists()  # and made no empty folders for nothing

    @pytest.mark.parametrize("override", legacy_install._OVERRIDES)
    def test_does_nothing_when_the_environment_redirects_state(self, homes, monkeypatch, override):
        config_home, data_home = homes
        _old_install(config_home, data_home)
        monkeypatch.setenv(override, "/somewhere/else")

        carried = legacy_install.carry_over(config_home, data_home)

        assert not carried
        assert not (config_home / NEW_NAME).exists()


class TestApiKeys:
    def test_copies_keys_from_the_old_keyring_service(self, homes, fake_keyring):
        fake_keyring.set_password(OLD_NAME, "ZAI_API_KEY", "zai-old-secret")
        fake_keyring.set_password(OLD_NAME, "TAVILY_API_KEY", "tvly-old-secret")

        carried = legacy_install.carry_over(*homes)

        assert sorted(carried.keys) == ["TAVILY_API_KEY", "ZAI_API_KEY"]
        assert fake_keyring.get_password(NEW_NAME, "ZAI_API_KEY") == "zai-old-secret"
        assert fake_keyring.get_password(NEW_NAME, "TAVILY_API_KEY") == "tvly-old-secret"

    def test_leaves_the_old_keyring_entries_in_place(self, homes, fake_keyring):
        fake_keyring.set_password(OLD_NAME, "ZAI_API_KEY", "zai-old-secret")

        legacy_install.carry_over(*homes)

        assert fake_keyring.get_password(OLD_NAME, "ZAI_API_KEY") == "zai-old-secret"

    def test_does_not_replace_a_key_already_stored_under_the_new_name(self, homes, fake_keyring):
        fake_keyring.set_password(OLD_NAME, "ZAI_API_KEY", "zai-old-secret")
        fake_keyring.set_password(NEW_NAME, "ZAI_API_KEY", "zai-new-secret")

        carried = legacy_install.carry_over(*homes)

        assert carried.keys == []
        assert fake_keyring.get_password(NEW_NAME, "ZAI_API_KEY") == "zai-new-secret"

    def test_only_copies_the_keys_this_tool_manages(self, homes, fake_keyring):
        fake_keyring.set_password(OLD_NAME, "SOMETHING_ELSE", "not-ours")

        carried = legacy_install.carry_over(*homes)

        assert carried.keys == []
        assert fake_keyring.get_password(NEW_NAME, "SOMETHING_ELSE") is None

    def test_a_broken_keyring_does_not_stop_the_files_being_copied(self, homes, monkeypatch):
        config_home, data_home = homes
        _old_install(config_home, data_home)

        def boom():
            raise RuntimeError("keyring locked")

        monkeypatch.setattr(legacy_install.credentials, "_backend", boom)

        carried = legacy_install.carry_over(config_home, data_home)

        assert carried.keys == []
        assert (config_home / NEW_NAME / "config.json").exists()


class TestDescribe:
    def test_names_what_was_copied_without_paths_or_values(self, homes, fake_keyring):
        config_home, data_home = homes
        _old_install(config_home, data_home)
        fake_keyring.set_password(OLD_NAME, "ZAI_API_KEY", "zai-old-secret")

        text = legacy_install.describe(legacy_install.carry_over(config_home, data_home))

        assert text == "settings, session history, the global .env, 1 API key"
        assert "zai-old-secret" not in text and str(config_home) not in text

    def test_pluralises_keys(self):
        carried = legacy_install.CarriedOver(keys=["ZAI_API_KEY", "TAVILY_API_KEY"])
        assert legacy_install.describe(carried) == "2 API keys"


class TestEntryPoint:
    def test_carries_over_then_starts_the_cli(self, monkeypatch, capsys):
        calls = []
        carried = legacy_install.CarriedOver(files=[], keys=["ZAI_API_KEY"], old_dirs=["/old/place"])
        monkeypatch.setattr(legacy_install, "carry_over", lambda: calls.append("carry") or carried)
        monkeypatch.setattr("synapse_deep_research.main.main", lambda: calls.append("main"))

        entry.run()

        captured = capsys.readouterr()
        out = " ".join(captured.err.split())  # the terminal wraps long lines
        assert captured.out == ""  # stdout stays free for a --json run
        assert calls == ["carry", "main"]
        assert "Carried over" in out and "1 API key" in out
        assert "/old/place" in out and "keyring entries" in out and "left in place" in out

    def test_says_nothing_when_there_was_nothing_to_carry(self, monkeypatch, capsys):
        monkeypatch.setattr(legacy_install, "carry_over", lambda: legacy_install.CarriedOver())
        monkeypatch.setattr("synapse_deep_research.main.main", lambda: None)

        entry.run()

        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_a_failing_carry_over_never_stops_the_cli_starting(self, monkeypatch):
        started = []

        def boom():
            raise OSError("disk on fire")

        monkeypatch.setattr(legacy_install, "carry_over", boom)
        monkeypatch.setattr("synapse_deep_research.main.main", lambda: started.append(True))

        entry.run()

        assert started == [True]
