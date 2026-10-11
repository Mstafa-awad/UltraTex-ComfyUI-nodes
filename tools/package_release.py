"""Build an installable ZIP with a single custom_nodes folder and verify its contents."""
import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--output",required=True)
args=p.parse_args()
package=Path(__file__).resolve().parents[1]
output=Path(args.output).resolve()
files=sorted(f for f in package.rglob("*") if f.is_file() and
             not {"__pycache__",".pytest_cache",".git"}.intersection(f.relative_to(package).parts) and
             f.suffix not in (".pyc",".pyo") and f.name!="SHA256SUMS.json")
for path in files:
    if path.suffix==".py":
        ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    elif path.suffix==".json":
        json.loads(path.read_text(encoding="utf-8"))
hashes={str(path.relative_to(package)).replace("\\","/"):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
checksums=package/"SHA256SUMS.json"
checksums.write_text(json.dumps(hashes,indent=2),encoding="utf-8")
files.append(checksums)
with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for path in files:
        archive.write(path,str(Path(package.name)/path.relative_to(package)))
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    assert all(name.startswith(package.name+"/") for name in archive.namelist())
    for name,digest in hashes.items():
        assert hashlib.sha256(archive.read(package.name+"/"+name)).hexdigest()==digest
digest=hashlib.sha256(output.read_bytes()).hexdigest()
output.with_suffix(".zip.sha256").write_text(digest+"  "+output.name+"\n",encoding="ascii")
print(json.dumps({"zip":str(output),"files":len(files),"bytes":output.stat().st_size,"sha256":digest},indent=2))
