"""Real STEP smoke check; run with: python -m tests.check_a3 <STEP file>.

STEP contains no authored kinematic tree. This checks all imported solids and
collision resources as one rigid assembly; joint behavior has analytical tests.
"""
import argparse
import io
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from workbench.collision_geometry import collision_meshes
from workbench.exporters import export_zip
from workbench.model import Link, Project, frame
from workbench.step import import_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('step', type=Path)
    parser.add_argument('--cached-import', type=Path)
    parser.add_argument('--output', type=Path, default=Path('exports/a3-collision-check'))
    args = parser.parse_args()
    started = time.perf_counter()
    if args.cached_import:
        parts = json.loads(args.cached_import.read_text(encoding='utf-8'))['parts']
    else:
        parts, _ = import_step(args.step, args.step.stem)
    args.output.mkdir(parents=True, exist_ok=True)
    total, report = 0, []
    # Generate in small batches so progress remains visible for large CAD files.
    for index, part in enumerate(parts.values()):
        link = Link(id=f'p{index}', name=f'part_{index}', parts=[part['id']])
        meshes = collision_meshes(link, parts, frame())
        assert meshes and all(mesh.is_volume and mesh.is_convex for mesh in meshes), (
            index, part['id'], part['name'], [(mesh.is_volume, mesh.is_convex, mesh.volume) for mesh in meshes])
        report.append({'part': part['id'], 'name': part['name'], 'volumes': len(meshes)})
        total += len(meshes)
        if index % 25 == 0:
            print(f'{index + 1}/{len(parts)} solids; {total} convex volumes', flush=True)
    project = Project(name='A3_collision_smoke', links=[
        Link(id='base', name='base', parts=list(parts), collision='convex')])
    archive = export_zip(project, parts, ['urdf'], merge=False)
    path = args.output / 'A3_collision_smoke.zip'
    path.write_bytes(archive)
    with ZipFile(io.BytesIO(archive)) as package:
        robot = ET.fromstring(package.read('A3_collision_smoke.urdf'))
        assert len(robot.findall('.//collision')) == total
        assert len(robot.findall('.//visual')) == len(parts)
        for mesh in robot.findall('.//mesh'):
            assert package.getinfo(mesh.attrib['filename']).file_size > 84
    result = {'file': args.step.name, 'solids': len(parts),
              'triangles': sum(len(part['faces']) for part in parts.values()),
              'convex_volumes': total, 'seconds': time.perf_counter() - started,
              'export': str(path), 'scope': 'rigid assembly geometry and URDF resources; no inferred six-axis mechanism',
              'parts': report}
    (args.output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'parts'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
