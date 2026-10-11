"""Run unit tests and copy integration reports into a release, shortening local paths."""
import argparse
import io
import json
import platform
import sys
import unittest
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--reports",required=True)
args=p.parse_args()
package=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(package.parent))
folder=package/"validation"
folder.mkdir(exist_ok=True)
suite=unittest.defaultTestLoader.loadTestsFromName("comfyui_ultratex_modular.tests.test_core")
log=io.StringIO()
result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
(folder/"unit_tests.txt").write_text(log.getvalue(),encoding="utf-8")
(folder/"unit_tests.json").write_text(json.dumps({"python":platform.python_version(),"tests_run":result.testsRun,
    "passed":result.wasSuccessful(),"failures":len(result.failures),"errors":len(result.errors)},indent=2),encoding="utf-8")
if not result.wasSuccessful():
    print(log.getvalue())
    raise SystemExit(1)
def clean(value):
    if isinstance(value,list):
        return [clean(item) for item in value]
    if isinstance(value,dict):
        return {key:("<temporary-workset-manifest>" if key=="manifest" else
                     "<process-local-encoder>" if key=="encoder_id" else clean(item)) for key,item in value.items()}
    if isinstance(value,str) and ":\\" in value:
        return value.replace("\\","/").split("/")[-1]
    return value
names=("model_report.json","sampling_report.json","native_2k_report.json","native_2k_optimized_report.json",
       "smoke_report.json","smoke_optimized_report.json","pbr_512_report.json","text_report.json")
for name in names:
    source=Path(args.reports)/name
    (folder/name).write_text(json.dumps(clean(json.loads(source.read_text(encoding="utf-8"))),indent=2),encoding="utf-8")
print(json.dumps({"unit_tests":result.testsRun,"passed":True,"integration_reports":len(names)},indent=2))
