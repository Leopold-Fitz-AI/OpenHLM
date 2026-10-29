"""Download pinned public checkpoints and actual embedded-JPEG reference data."""
import os,json,argparse,hashlib,shutil,time
from pathlib import Path
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com');os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120');os.environ.setdefault('HF_HUB_ETAG_TIMEOUT','60')
from huggingface_hub import hf_hub_download,snapshot_download
p=argparse.ArgumentParser();p.add_argument('--references-only',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=root/'public_data/paper_reference';out.mkdir(parents=True,exist_ok=True)
manifest={'pid':os.getpid(),'status':'running','data_revision':'fe7f13138d0a22a2a2b06b775e4effef9eb54903','model_revision':'f3a109f817c65ea76a659d0b7b6f7cfff6d6825c','completed':[]}
def save():
 (out/'download_status.tmp').write_text(json.dumps(manifest,indent=2));(out/'download_status.tmp').replace(out/'download_status.json')
save()
try:
 if shutil.disk_usage(root).free<50*1024**3:raise RuntimeError('Need 50 GiB free for the selected release assets')
 for episode in [80]:
  file=f'g1_HLM-12_full_teleop/data/chunk-000/episode_{episode:06d}.parquet';print('Downloading reference',file,flush=True)
  path=hf_hub_download('OpenHLM/OpenHLM-data',file,repo_type='dataset',revision=manifest['data_revision'],local_dir=out)
  digest=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()
  manifest['completed'].append({'type':'reference','path':path,'sha256':digest});save();print('Reference ready',path,flush=True)
 if not a.references_only:
  folders=['12tasks_8full-teleop_4humi','12tasks_8full-teleop_4stationary_teleop','20fruit-arrangement_20full-teleop','20fruit-arrangement_6full-teleop_14humi']
  for name in folders:
   manifest['current']=name;save();print('Downloading checkpoint',name,flush=True)
   snapshot_download('OpenHLM/OpenHLM-ckpts',revision=manifest['model_revision'],allow_patterns=[name+'/*'],local_dir=root/'models/paper_release',max_workers=2)
   manifest['completed'].append({'type':'checkpoint','name':name});save();print('Checkpoint ready',name,flush=True)
 manifest['status']='complete';save()
except Exception as e:
 manifest['status']='failed';manifest['error']=str(e);save();raise
