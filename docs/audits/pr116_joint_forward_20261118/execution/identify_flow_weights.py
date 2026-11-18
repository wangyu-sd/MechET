"""Recognize the official FlowER checkpoint format by its inert pickle opcodes and training step."""
from concurrent.futures import ThreadPoolExecutor
import json,pickletools,time,zipfile
from pathlib import Path
work=Path(__file__).resolve().parent
expected=('model.2880000_95.pt','model.2940000_97.pt','model.2370000_78.pt','model.1440000_47.pt')
final=work/'whaley_all_weight_paths.txt'
partial=work/'whaley_all_weight_paths.part'
report=work/'identified_flow_weights.jsonl'


def checkpoint_step(stream):
    """Read the author's configuration, training step and tensor-dictionary keys without loading it."""
    header = stream.read(4096)
    if b'settings' not in header or b'Args' not in header:
        return None
    stream.seek(0)
    integers = {'INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4'}
    memo = {'MEMOIZE', 'BINPUT', 'LONG_BINPUT', 'PUT', 'FRAME'}
    found_configuration = False
    expecting_step = False
    step = None
    for op, arg, _ in pickletools.genops(stream):
        if expecting_step:
            if op.name in integers and type(arg) is int:
                step = arg
            if op.name not in memo:
                expecting_step = False
        if arg in ('args', 'settings'):
            found_configuration = True
        if arg == 'total_step':
            expecting_step = True
        if arg == 'state_dict' and found_configuration and step is not None:
            return step
    return None


def inspect(name):
    p=Path(name)
    # Spectral sample shards use SHA-named tensors. Their .pt suffix is the
    # dataset serialization convention, so skip the metadata read per sample.
    if '/msnlib/' in name and '/shards/' in name and p.suffix=='.pt' and len(p.stem)==64:
        return
    if p.suffix.lower() not in ('.zip','.pt','.pth','.ckpt','.bin') and p.name not in ('55904912','65102577'):
        return
    try:
        size=p.stat().st_size
        suffix=p.suffix.lower()
        if p.name in ('55904912','65102577') or (suffix=='.zip' and size in (444244997,179207131)):
            return dict(kind='release_size_candidate',path=name,size=size)
        if suffix=='.zip':
            if size<48_000_000:return
            with zipfile.ZipFile(p) as z:
                members=[k for k in z.namelist() if k.split('/')[-1] in expected or ('flower' in k.lower() and k.lower().endswith(('.pth','.pt','.zip')))]
                if members:return dict(kind='archive_member',path=name,members=members[:40])
        elif suffix in ('.pt','.pth','.ckpt','.bin') and 48_000_000<size<1_500_000_000:
            with zipfile.ZipFile(p) as z:
                member=next((m for m in z.infolist() if m.filename.rsplit('/',1)[-1]=='data.pkl'),None)
                if not member or member.file_size>16<<20:return
                with z.open(member) as f:step=checkpoint_step(f)
                if step is not None:
                    return dict(kind='author_format',path=name,step=step,paper_step=step==2880000,may_2026_step=step==2940000,size=size)
    except (zipfile.BadZipFile,ValueError,StopIteration,OSError,EOFError,RuntimeError):
        return
def main():
    previous = [json.loads(line) for line in report.read_text().splitlines()] if report.exists() else []
    found_paths = {row['path'] for row in previous}
    total = 0
    source = (final if final.exists() else partial).open('rb')
    with source, ThreadPoolExecutor(max_workers=6) as pool, report.open('a') as log:
        while True:
            batch = []
            for _ in range(4096):
                position = source.tell()
                line = source.readline()
                if not line.endswith(b'\n'):
                    source.seek(position)
                    break
                path = line[:-1].decode(errors='surrogateescape')
                if path not in found_paths:
                    batch.append(path)
            total += len(batch)
            for candidate in pool.map(inspect, batch):
                if candidate:
                    found_paths.add(candidate['path'])
                    log.write(json.dumps(candidate) + '\n')
                    log.flush()
                    if candidate.get('paper_step') or candidate.get('may_2026_step') or candidate['kind'] != 'author_format':
                        print(candidate, flush=True)
            if not line.endswith(b'\n'):
                if final.exists():
                    break
                time.sleep(8)
    print('inspected named artifacts', total, 'candidates', len(found_paths), flush=True)


if __name__ == '__main__':
    main()
