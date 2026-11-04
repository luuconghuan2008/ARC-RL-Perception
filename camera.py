import time
import depthai as dai
import cv2
from config import *
from tag_utils import parse_tag_info, get_relative_poses

class Camera():
    def __init__(self, cam_id, cams, pipeline):
        self.id = cam_id
        self.cams = cams
        self.q_frame = {}
        self.cams_matrix = {}
        self.cams_dist_coeffs = {}
        self.pipeline = pipeline

        self.prev_positions = {}
        self.filtered_positions = {}
        self.prev_time = time.time()
        self.tag_speeds = {}
        self.alpha = 0.3

    def create_pipeline(self, calib):
        sockets = {
            'A': dai.CameraBoardSocket.CAM_A,
            'B': dai.CameraBoardSocket.CAM_B,
            'C': dai.CameraBoardSocket.CAM_C
        }

        for cam in self.cams:
            if cam not in sockets:
                continue

            socket = sockets[cam]
            camera = self.pipeline.create(dai.node.Camera).build(socket)
            output = camera.requestOutput((640, 400))
            queue = output.createOutputQueue(maxSize=1, blocking=False)

            matrix = np.array(calib.getCameraIntrinsics(socket, 640, 400))

            dist = np.array(calib.getDistortionCoefficients(socket))

            self.q_frame[cam] = queue
            self.cams_matrix[cam] = matrix
            self.cams_dist_coeffs[cam] = dist

    def detect_tag(self, color_frame, gray_frame, cam):
        camera_matrix = self.cams_matrix[cam]
        dist_coeffs = self.cams_dist_coeffs[cam]
        detections = detector.detect(gray_frame)
        parsed_tags = {}
        for tag in detections: # type: ignore
            info = parse_tag_info(tag, camera_matrix, dist_coeffs, OBJECT_POINTS)
            if info:
                parsed_tags[info["id"]] = info

        for tag_id, info in parsed_tags.items():
            rvec = info["rvec"]
            tvec = info["tvec"]
            image_points = info["corners"]

            R, _ = cv2.Rodrigues(rvec)
            R_corrected = R @ R_FLIP
            rvec_corrected, _ = cv2.Rodrigues(R_corrected)

            if tag_id == 0:
                cv2.drawFrameAxes(color_frame, camera_matrix, dist_coeffs, 
                                  rvec_corrected, tvec, HALF_SIZE, thickness=2)
            
            color = (0, 255, 0) if tag_id in WARP_CORNER_IDS else (0, 255, 255)
            pts = image_points.astype(int)
            cv2.polylines(color_frame, [pts], isClosed=True, color=color, thickness=2)

        return parsed_tags, color_frame

    def draw_field_map(self, tag_pos = None):
        field_w_px = int(FIELD_WIDTH * MAP_SCALE_PX_PER_M)
        field_h_px = int(FIELD_LENGTH * MAP_SCALE_PX_PER_M)

        map_w = field_w_px + 2 * MARGIN_PX
        map_h = field_h_px + 2 * MARGIN_PX
        field_map = np.zeros((map_h, map_w, 3), dtype=np.uint8)

        top_left = (MARGIN_PX, MARGIN_PX)                              # Tag 0 (0,0)
        top_right = (MARGIN_PX + field_w_px, MARGIN_PX)                 # Tag 1
        bottom_right = (MARGIN_PX + field_w_px, MARGIN_PX + field_h_px) # Tag 2
        bottom_left = (MARGIN_PX, MARGIN_PX + field_h_px)               # Tag 3

        cv2.rectangle(field_map, top_left, bottom_right, (80, 80, 80), 2)

        cv2.circle(field_map, top_left, 5, (0, 0, 255), -1)     # Tag 0 (Red Origin)
        cv2.putText(field_map, "0", (top_left[0] - 25, top_left[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        cv2.circle(field_map, top_right, 5, (0, 255, 0), -1)    # Tag 1
        cv2.putText(field_map, "1", (top_right[0] + 5, top_right[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        cv2.circle(field_map, bottom_right, 5, (0, 255, 0), -1) # Tag 2
        cv2.putText(field_map, "2", (bottom_right[0] + 5, bottom_right[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        cv2.circle(field_map, bottom_left, 5, (0, 255, 0), -1)  # Tag 3
        cv2.putText(field_map, "3", (bottom_left[0] - 25, bottom_left[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        if tag_pos:
            for tag_id, pose in tag_pos.items():
                px, py = pose
                if 0 <= px < map_w and 0 <= py < map_h:
                    cv2.circle(field_map, (px, py), 4, (0, 255, 255), 1)
                    cv2.circle(field_map, (px, py), 1, (0, 255, 255), 1)
                    cv2.putText(field_map, f"ID {tag_id}", (px + 8, py + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return field_map

    def get_rel_pos_map(self, color_frame, parsed_tags):
        field_w_px = int(FIELD_WIDTH * MAP_SCALE_PX_PER_M)
        field_h_px = int(FIELD_LENGTH * MAP_SCALE_PX_PER_M)

        top_left = (MARGIN_PX, MARGIN_PX)                              # Tag 0 (0,0)
        top_right = (MARGIN_PX + field_w_px, MARGIN_PX)                 # Tag 1
        bottom_right = (MARGIN_PX + field_w_px, MARGIN_PX + field_h_px) # Tag 2
        bottom_left = (MARGIN_PX, MARGIN_PX + field_h_px)               # Tag 3

        tag_pos = {}
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        if 0 in parsed_tags:
            relative_poses = get_relative_poses(parsed_tags, origin_id=0)

            if relative_poses:
                for tag_id, rel_info in relative_poses.items():
                    if tag_id in WARP_CORNER_IDS:
                        continue

                    raw_rel_pos = rel_info["rel_position"]

                    if tag_id in self.filtered_positions:
                        filtered_pos = self.alpha * raw_rel_pos + (1 - self.alpha) * self.filtered_positions[tag_id]
                    else:
                        filtered_pos = raw_rel_pos

                    self.filtered_positions[tag_id] = filtered_pos

                    if dt > 0 and tag_id in self.prev_positions:
                        prev_pos, prev_t = self.prev_positions[tag_id]
                        dt_tag = current_time - prev_t
                        if dt_tag > 0:
                            distance = np.linalg.norm(filtered_pos[:2] - prev_pos[:2])
                            self.tag_speeds[tag_id] = distance / dt_tag

                    self.prev_positions[tag_id] = (filtered_pos, current_time)

                    px = int(top_left[0] + filtered_pos[0] * MAP_SCALE_PX_PER_M)
                    py = int(top_left[1] + filtered_pos[1] * MAP_SCALE_PX_PER_M)

                    tag_pos[tag_id] = (px, py)

                    if tag_id in parsed_tags:
                        corner_pt = parsed_tags[tag_id]["corners"][0].astype(int)
                        speed_str = f"{self.tag_speeds.get(tag_id, 0.0):.2f} m/s" if tag_id in self.tag_speeds else "0.00 m/s"
                        cv2.putText(
                            color_frame,
                            f"({filtered_pos[0] * 100.0:.1f}, {filtered_pos[1] * 100.0:.1f}, {speed_str})",
                            (corner_pt[0], corner_pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 255),
                            1
                        )
        return color_frame, tag_pos

    def warp_view(self, parsed_tags, color_frame):
        valid = all(tag_id in parsed_tags for tag_id in WARP_CORNER_IDS)
        if valid:
            t0 = parsed_tags[0]["position_3d"]
            t1 = parsed_tags[1]["position_3d"]
            t2 = parsed_tags[2]["position_3d"]
            t3 = parsed_tags[3]["position_3d"]

            w_top = np.linalg.norm(t1 - t0)
            w_bot = np.linalg.norm(t2 - t3)
            h_left = np.linalg.norm(t3 - t0)
            h_right = np.linalg.norm(t2 - t1)

            tag_field_w_m = (w_top + w_bot) / 2.0
            tag_field_h_m = (h_left + h_right) / 2.0
        
            warp_width = max(20, int((tag_field_w_m + TAG_SIZE_METERS) * WARP_SCALE_PX_PER_M))
            warp_height = max(20, int((tag_field_h_m + TAG_SIZE_METERS) * WARP_SCALE_PX_PER_M))

            src_pts = np.array(
                [parsed_tags[WARP_CORNER_IDS[0]]["corners"][3],
                parsed_tags[WARP_CORNER_IDS[1]]["corners"][2],
                parsed_tags[WARP_CORNER_IDS[2]]["corners"][1],
                parsed_tags[WARP_CORNER_IDS[3]]["corners"][0]],
                dtype=np.float32
            )
            dst_pts = np.array([
                [0, 0],
                [warp_width - 1, 0],
                [warp_width - 1, warp_height - 1],
                [0, warp_height - 1]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_frame = cv2.warpPerspective(color_frame, M, (warp_width, warp_height))
        else:
            warped_frame = None
        return warped_frame
    
    def process_frame(self, frame, cam, detect_tag = True, warp = True):
        if frame is None:
            return None, None, None
        
        if len(frame.shape) == 3:
            color_frame = frame
        else:
            gray_frame = frame
            color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if detect_tag:
            parsed_tags, color_frame = self.detect_tag(color_frame, gray_frame, cam)  
            color_frame, tag_pos = self.get_rel_pos_map(color_frame, parsed_tags)
        else:
            tag_pos = None

        if warp:
            warped_frame = self.warp_view(parsed_tags, color_frame)
        else:
            warped_frame = None
        

        return color_frame, warped_frame, tag_pos

    def format_warped_for_stack(self, frame, target_height=300):
        if frame is None:
            return np.zeros((target_height, target_height, 3), dtype=np.uint8)
        
        h, w = frame.shape[:2]
        new_width = max(1, int(target_height * (w / h)))
        return cv2.resize(frame, (new_width, target_height))
