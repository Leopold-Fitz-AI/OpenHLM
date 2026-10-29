"""Run on the camera host to export the exact recording profile, read-only.

Requires pyrealsense2 and attached cameras. No robot commands are sent. Side
assignment must use the hardware serial mapping; sorting devices is not proof.
"""
import argparse,json
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('d405_intrinsics.json'))
parser.add_argument('--left-serial');parser.add_argument('--right-serial')
parser.add_argument('--width',type=int,default=640);parser.add_argument('--height',type=int,default=480)
args=parser.parse_args()
import pyrealsense2 as rs
records=[]
for device in rs.context().query_devices():
    name=device.get_info(rs.camera_info.name)
    if 'D405' not in name:continue
    serial=device.get_info(rs.camera_info.serial_number)
    pipeline=rs.pipeline();config=rs.config();config.enable_device(serial)
    config.enable_stream(rs.stream.color,args.width,args.height,rs.format.rgb8,30)
    try:
        profile=pipeline.start(config)
        for _ in range(10):pipeline.wait_for_frames()
        intr=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        records.append({'name':name,'serial':serial,
            'side':'left' if serial==args.left_serial else 'right' if serial==args.right_serial else 'unassigned',
            'width':intr.width,'height':intr.height,'fps':30,
            'fx':intr.fx,'fy':intr.fy,'ppx':intr.ppx,'ppy':intr.ppy,
            'distortion_model':str(intr.model),'distortion_coefficients':list(intr.coeffs),
            'fov_degrees':list(rs.rs2_fov(intr)),
            'wrist_to_camera_extrinsics':None})
    finally:pipeline.stop()
if not records:raise RuntimeError('No D405 camera was found; run on the camera host')
args.output.write_text(json.dumps({'source':'device_stream_profile_not_catalog_fov','devices':records},indent=2))
print(args.output)
