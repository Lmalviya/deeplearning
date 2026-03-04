import json, sys

with open(r'c:\Users\23add\workspace\deeplearning\notes\01_neural_building_blocks.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
output_lines = []
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        lines = src.split('\n')
        headings = [l.strip() for l in lines if l.strip().startswith('#')]
        if headings:
            for h in headings:
                output_lines.append(f'Cell {i}: {h}')
        else:
            output_lines.append(f'Cell {i} [md]: {lines[0][:80]}')
    else:
        src = ''.join(cell['source'])
        output_lines.append(f'Cell {i} [code]: {src.split(chr(10))[0][:80]}')

with open('nb_outline.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print("Done - written to nb_outline.txt")
print(f"Total cells: {len(cells)}")
