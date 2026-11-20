"""Developer inspection of product names and geometry in the supplied A3 STEP."""
import json
import re
from pathlib import Path


def product_names(path):
    entities = {}
    for line in path.open(encoding='latin1'):
        match = re.match(r"#(\d+) = (PRODUCT|PRODUCT_DEFINITION|PRODUCT_DEFINITION_FORMATION(?:_WITH_SPECIFIED_SOURCE)?|NEXT_ASSEMBLY_USAGE_OCCURRENCE)\s*\((.*)\)\s*;", line)
        if match:
            entities[match[1]] = (match[2], match[3])

    def product(key):
        kind, value = entities[key]
        if kind == 'PRODUCT':
            name = re.findall(r"'([^']*)'", value)[0]
            return name.encode('latin1').decode('gb18030', errors='replace')
        for reference in re.findall(r'#(\d+)', value):
            if reference in entities and entities[reference][0] in ('PRODUCT', 'PRODUCT_DEFINITION_FORMATION', 'PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE'):
                return product(reference)

    result = {}
    for kind, value in entities.values():
        if kind == 'NEXT_ASSEMBLY_USAGE_OCCURRENCE':
            name = re.findall(r"'([^']*)'", value)[0]
            references = re.findall(r'#(\d+)', value)
            result[name] = product(references[1])
    return result


if __name__ == '__main__':
    names = product_names(next(Path('.').glob('*.STEP')))
    output = Path('exports/a3-motion-check')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'product_names.json').write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding='utf-8')
    for key, name in names.items():
        print(key, name)
