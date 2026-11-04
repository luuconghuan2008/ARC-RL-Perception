import cv2
import numpy as np

def parse_tag_info(tag, camera_matrix, dist_coeffs, object_points):
    try:
        image_points = tag.corners.astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if not success:
            return None

        R, _ = cv2.Rodrigues(rvec)
        
        return {
            "id": tag.tag_id,
            "center": (float(tag.center[0]), float(tag.center[1])),
            "corners": image_points,
            "position_3d": tvec.flatten(),
            "rotation_matrix": R,
            "rvec": rvec,
            "tvec": tvec,
            "distance": float(np.linalg.norm(tvec))
        }
    except Exception as e:
        tag_id = getattr(tag, 'tag_id', 'unknown')
        print(f"Error processing tag {tag_id}: {e}")
        return None


def get_relative_poses(parsed_tags, origin_id=0):
    if origin_id not in parsed_tags:
        return None

    R_0 = parsed_tags[origin_id]["rotation_matrix"]
    t_0 = parsed_tags[origin_id]["position_3d"]

    relative_tags = {}
    for tag_id, tag_data in parsed_tags.items():
        t_i = tag_data["position_3d"]
        R_i = tag_data["rotation_matrix"]

        t_rel = R_0.T @ (t_i - t_0)
        R_rel = R_0.T @ R_i

        relative_tags[tag_id] = {
            "rel_position": t_rel,
            "rel_rotation": R_rel,
            "rel_distance": float(np.linalg.norm(t_rel))
        }

    return relative_tags