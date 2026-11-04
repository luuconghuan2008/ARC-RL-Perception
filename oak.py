import cv2
import depthai as dai
import numpy as np
from camera import Camera

for device in dai.Device.getAllAvailableDevices():
    print(f"{device.getDeviceId()} {device.state}")

pipeline = dai.Pipeline()

oakD = Camera("14442C1001CA5ED700", ['B', 'C'], pipeline)
calib = pipeline.getCalibrationData()
oakD.create_pipeline(calib)

with pipeline:
    pipeline.start()

    while pipeline.isRunning():
        in_b = oakD.q_frame["B"].tryGet()
        in_c = oakD.q_frame["C"].tryGet()
        frame_b = None
        frame_c = None
        if in_b is not None:
            frame_b = in_b.getCvFrame() # type: ignore
        if in_c is not None:
            frame_c = in_c.getCvFrame() # type: ignore

        if frame_b is not None and frame_c is not None:
            pose_b, warp_b, tag_pos_b = oakD.process_frame(frame_b, "B")
            pose_c, warp_c, tag_pos_c = oakD.process_frame(frame_c, "C")
            video_display = np.hstack((pose_b, pose_c))
            if tag_pos_b and tag_pos_c:
                pos = {}
                common_ids = set(tag_pos_b) & set(tag_pos_c)
                for tag_id in common_ids:
                    pos_b = tag_pos_b[tag_id]
                    pos_c = tag_pos_c[tag_id]

                    px = int((pos_b[0] + pos_c[0]) / 2)
                    py = int((pos_b[1] + pos_c[1]) / 2)

                    pos[tag_id] = (px, py)
            else:
                if tag_pos_b:
                    pos = tag_pos_b
                else:
                    pos = tag_pos_c

            field_map = oakD.draw_field_map(pos)
            warp_b_fmt = oakD.format_warped_for_stack(warp_b)
            warp_c_fmt = oakD.format_warped_for_stack(warp_c)
            warp_display = np.hstack((warp_b_fmt, warp_c_fmt))
            cv2.imshow("Display", video_display)
            cv2.imshow("Warp", warp_display)
            cv2.imshow("Map", field_map)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break