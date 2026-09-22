"""Fail-closed offline wheel contract and full environment audit (selected Python)."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import tomllib
import zipfile
from email.parser import BytesParser

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def inventory():
    result = {}
    for dist in metadata.distributions():
        name = canonicalize_name(dist.metadata['Name'])
        if name in result:
            raise ValueError(f'duplicate distribution: {name}')
        result[name] = dist.version
    return result


def validate_wheels(manifest, wheel_dir, lock):
    assert platform.machine() == manifest['architecture'], 'unsupported architecture'
    assert platform.python_version() == manifest['python_version'], 'unsupported Python'
    expected = []
    names = set()
    for wheel in manifest['wheels']:
        name = canonicalize_name(wheel['name'])
        assert name not in names, 'duplicate wheel'
        names.add(name)
        filename = wheel['filename']
        assert Path(filename).name == filename and filename.endswith('.whl')
        assert wheel['url'].startswith('https://files.pythonhosted.org/packages/')
        assert wheel['url'].endswith('/' + filename)
        data = (wheel_dir / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == wheel['sha256'], filename
        parsed_name, version, _, tags = parse_wheel_filename(filename)
        assert parsed_name == name and version == Version(wheel['version'])
        assert tags.intersection(sys_tags()), f'unsupported wheel: {filename}'
        with zipfile.ZipFile(wheel_dir / filename) as archive:
            paths = [p for p in archive.namelist() if p.endswith('.dist-info/METADATA')]
            assert len(paths) == 1
            message = BytesParser().parsebytes(archive.read(paths[0]))
            assert canonicalize_name(message['Name']) == name
            assert Version(message['Version']) == version
        expected.append(f"{name}=={wheel['version']} --hash=sha256:{wheel['sha256']}")
    assert lock.read_text().splitlines() == expected, 'lock/manifest mismatch'
    assert {p.name for p in wheel_dir.iterdir()} == {w['filename'] for w in manifest['wheels']}, 'unexpected wheel input'


def records():
    return {canonicalize_name(d.metadata['Name']): hashlib.sha256(
        (d.read_text('RECORD') or '').encode()).hexdigest() for d in metadata.distributions()}


def conflicts(versions, requirements):
    environment = default_environment()
    extras = {n: {''} for n in requirements}
    pending = list(extras)
    while pending:
        owner = pending.pop()
        for text in requirements.get(owner, []):
            req = Requirement(text)
            if req.marker and not any(req.marker.evaluate({**environment, 'extra': extra}) for extra in extras[owner]):
                continue
            name = canonicalize_name(req.name)
            if name in requirements:
                new = set(req.extras) - extras[name]
                if new:
                    extras[name].update(new)
                    pending.append(name)
    found = []
    for owner, texts in requirements.items():
        for text in texts:
            req = Requirement(text)
            active = sorted(extra for extra in extras[owner]
                            if not req.marker or req.marker.evaluate({**environment, 'extra': extra}))
            if not active:
                continue
            installed = versions.get(canonicalize_name(req.name))
            if installed is None or not req.specifier.contains(installed, prereleases=True):
                found.append(dict(distribution=owner, version=versions[owner], requirement=text,
                                  installed=installed, active_extras=active))
    return sorted(found, key=lambda row: (row['distribution'], row['requirement']))


def audit(manifest, before, after, requirements, baseline, before_records, after_records):
    approved = {canonicalize_name(w['name']): w['version'] for w in manifest['wheels']}
    changes = {n: [before.get(n), after.get(n)] for n in before.keys() | after.keys()
               if before.get(n) != after.get(n)}
    assert set(changes) <= approved.keys(), ('unapproved distribution changes', changes)
    assert all(after.get(n) == v for n, v in approved.items()), 'override versions'
    record_changes = {n for n in before_records.keys() | after_records.keys()
                      if before_records.get(n) != after_records.get(n)}
    assert record_changes <= approved.keys(), ('unapproved RECORD changes', record_changes)
    expected = sorted(baseline['conflicts'] + manifest['intentional_overrides'],
                      key=lambda row: (row['distribution'], row['requirement']))
    observed = conflicts(after, requirements)
    assert observed == expected, ('post-install conflicts differ from exact contract', observed, expected)
    return changes


def check_baseline(baseline, versions, requirements, record_hashes):
    assert conflicts(versions, requirements) == baseline['conflicts'], 'pre-install conflicts differ from immutable baseline'
    protected = {n: {'version': versions.get(n), 'record_sha256': record_hashes.get(n)}
                 for n in baseline['protected_records']}
    assert protected == baseline['protected_records'], 'protected base distribution drift'


def installed_requirements():
    return {canonicalize_name(d.metadata['Name']): list(d.requires or []) for d in metadata.distributions()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['before', 'after'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / 'manifest.json').read_text())
    baseline = json.loads((root / 'baseline.json').read_text())
    assert baseline['base'] == manifest['base'], 'baseline base identity'
    environment = default_environment()
    assert all(environment[k] == v for k, v in baseline['environment'].items()), 'baseline platform'
    if args.mode == 'before':
        validate_wheels(manifest, root / 'wheels', root / 'requirements.lock')
        versions = inventory()
        check_baseline(baseline, versions, installed_requirements(), records())
        (root / 'before.json').write_text(json.dumps(versions, sort_keys=True))
        (root / 'before-records.json').write_text(json.dumps(records(), sort_keys=True))
    else:
        after = inventory()
        before = json.loads((root / 'before.json').read_text())
        source = tomllib.loads(args.source.read_text())['project']['dependencies']
        requirements = installed_requirements()
        requirements['sglang'] = source
        changes = audit(manifest, before, after, requirements, baseline,
                        json.loads((root / 'before-records.json').read_text()), records())
        receipt = {'before': before, 'after': after, 'changes': changes,
                   'intentional_overrides': manifest['intentional_overrides'],
                   'inherited_conflicts': baseline['conflicts'], 'records': records()}
        (root / 'inventory.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        print('DEPENDENCY_OVERLAY_PASS', json.dumps(changes), flush=True)


if __name__ == '__main__':
    main()
