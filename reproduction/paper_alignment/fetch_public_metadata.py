"""Pin public release metadata without downloading multi-GB model weights."""
import json,urllib.request,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'evidence'/'release_metadata';ROOT.mkdir(parents=True,exist_ok=True)
manifest={}
for repo,kind in [('OpenHLM/OpenHLM-data','datasets'),('OpenHLM/OpenHLM-ckpts','models')]:
    with urllib.request.urlopen('https://huggingface.co/api/'+kind+'/'+repo,timeout=60) as r:info=json.load(r)
    sha=info['sha'];manifest[kind]={'repository':repo,'revision':sha,'files':[]}
    if kind=='models':continue
    for name in ['g1_HLM-12_full_teleop','g1_HLM-12_humi','g1_HLM-12_stationary_teleop','long_g1_5_fruits_full_teleop','long_g1_5_fruits_humi']:
        for file in ['info.json','tasks.jsonl','episodes.jsonl']:
            relative=f'{name}/meta/{file}';url=f'https://huggingface.co/datasets/{repo}/resolve/{sha}/{relative}'
            try:
                with urllib.request.urlopen(url,timeout=90) as r:data=r.read()
                path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                manifest[kind]['files'].append({'path':relative,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
            except Exception as e:manifest[kind]['files'].append({'path':relative,'error':str(e)})
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
