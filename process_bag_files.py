import rosbag
import numpy as np
from ros_numpy import numpify
import cv2
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion
import matplotlib.pyplot as plt
from PIL import Image as PILImage

def process_bag(file, output_file=None, annotations_folder=None):
    assert file.endswith('.bag')
    bag = rosbag.Bag(file)

    start = bag.get_start_time()
    end = bag.get_end_time()
    info = bag.get_type_and_topic_info()[1]
    num_imgs = info['/camera/color/image_raw_throttled'].message_count
    hz = num_imgs / (end - start)

    # Annotation stuff
    annotations_index = []
    if annotations_folder is not None:
        imgs_to_annotate = 15
        annotations_index = np.linspace(0.20 * num_imgs, num_imgs - 1, num=imgs_to_annotate).astype(np.int32)

    pose_info = []
    queued_diagnostic_img = None
    video_writer = None
    img_idx = 0
    annotated_imgs = 0

    if output_file is None:
        output_file = file.replace('.bag', '.avi')
    try:
        for topic, msg, t in bag:
            if topic == '/camera/color/image_raw_throttled':

                msg.__class__ = Image
                img = numpify(msg)
                if video_writer is None:
                    h, w = img.shape[:2]
                    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), int(hz), (2*w, h))

                if queued_diagnostic_img is None:
                    queued_diagnostic_img = np.zeros(img.shape, dtype=np.uint8)

                final_img = np.concatenate([cv2.cvtColor(img, cv2.COLOR_RGB2BGR), queued_diagnostic_img], axis=1)
                video_writer.write(final_img)

                if img_idx in annotations_index:
                    mod_img = img.copy()
                    cv2.line(mod_img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (0, 0, 0), 2)
                    PILImage.fromarray(mod_img).save(os.path.join(annotations_folder, f'{annotated_imgs}.png'))
                    annotated_imgs += 1

                img_idx += 1


            elif topic == 'diagnostic_img':
                msg.__class__ = Image
                img_array = numpify(msg)
                queued_diagnostic_img = process_diagnostic_img(img_array)

            elif topic == 'tool_pose':
                new_msg = Pose()

                o = msg.pose.orientation
                new_msg.orientation = Quaternion(o.x, o.y, o.z, o.w)
                p = msg.pose.position
                new_msg.position = Point(p.x, p.y, p.z)
                mat = numpify(new_msg)
                pose_info.append(mat)
    finally:
        if video_writer is not None:
            video_writer.release()

    # Post-process the pose info
    processed_pos = []
    first_pose = pose_info[0]
    inv_tf = np.linalg.inv(first_pose)
    for pose in pose_info:
        homog_pos = pose[:,3]
        tfed_pos = (inv_tf @ homog_pos)[:3]
        processed_pos.append(tfed_pos)

    return processed_pos

def process_diagnostic_img(img):
    w = img.shape[1] // 2
    h = img.shape[0] // 2

    rgb = img[:h,:w]
    base_mask = img[:h, w:2*w]
    mask = (img[:h, w:2*w].mean(axis=2)).astype(np.uint8)

    from LeaderDetector import LeaderDetector
    images = {
        'RGB0': rgb,
        'Mask': mask,
        'Edge': cv2.Canny(mask, 50, 150, apertureSize=3)
    }

    mask_overlay_factor = 0.7
    base_img = mask_overlay_factor * base_mask + (1 - mask_overlay_factor) * rgb

    arrows = []
    leader_detect = LeaderDetector(images, b_output_debug=False, b_recalc=True)
    if leader_detect.vertical_leader_quads:
        quad = leader_detect.vertical_leader_quads[-1]
        ts = np.linspace(0, 1, 101)
        pts = quad.pt_axis(ts).astype(np.int32)
        idx_closest = np.argmin(np.abs(pts[:,1] - h / 2))
        target_pt = pts[idx_closest]
        gradient = quad.tangent_axis(ts[idx_closest])
        gradient /= np.linalg.norm(gradient)
        if gradient[1] < 0:
            gradient *= -1

        scan_vel = 0.05
        correction_term = np.radians(69.8) * (target_pt[0] - w / 2) / w * 0.8
        vel = gradient * scan_vel + np.array([correction_term, 0])
        vel_norm = vel / np.linalg.norm(vel)

        vl = 75
        arrows = [
            [tuple(target_pt.astype(np.int64)), tuple((target_pt + gradient * vl).astype(np.int64)), (0, 255, 0)],
            [tuple(target_pt.astype(np.int64)), tuple((target_pt + np.array([(correction_term / scan_vel), 0]) * vl).astype(np.int64)), (0, 0, 255)],
            [tuple(target_pt.astype(np.int64)), tuple((target_pt + vel_norm * vl).astype(np.int64)), (0, 255, 255)],
        ]
        cv2.polylines(base_img, [pts.reshape((-1,1,2))], False, (0, 0, 255), 2)

    cv2.line(base_img, (w // 2, 0), (w // 2, h), (0, 0, 0), 2)
    cv2.line(base_img, (0, h // 2), (w, h // 2), (0, 0, 0), 2)
    cv2.circle(base_img, (w // 2, h // 2), 4, (0, 255, 0), -1)

    for start, end, color in arrows:
        cv2.arrowedLine(base_img, start, end, color, 2)

    return base_img.astype(np.uint8)


if __name__ == '__main__':
    base_folder = os.path.join(os.path.expanduser('~'), 'data', 'follow_the_leader')
    groups = [
        {'gs_fast': '0.bag', 'gs_slow': '1.bag', 'ngs_fast': '2.bag', 'ngs_slow': '3.bag'},
        {'gs_fast': '7.bag', 'gs_slow': '8.bag', 'ngs_fast': '4.bag', 'ngs_slow': '5.bag'},
        {'gs_fast': '9.bag', 'gs_slow': '10.bag', 'ngs_fast': '11.bag', 'ngs_slow': '12.bag'},
    ]

    cat_map = {
        'gs_fast': 'GS, 10 cm/s',
        'gs_slow': 'GS, 2.5 cm/s',
        'ngs_fast': 'No GS, 10 cm/s',
        'ngs_slow': 'No GS, 2.5 cm/s'
    }

    mpl_settings = {
        'gs_fast': {'color': 'green', 'linestyle': 'dashed'},
        'gs_slow': {'color': 'green', 'linestyle': 'solid'},
        'ngs_fast': {'color': 'blue', 'linestyle': 'dashed'},
        'ngs_slow': {'color': 'blue', 'linestyle': 'solid'}
    }


    for i, group in enumerate(groups):
        for cat, file in group.items():
            input_path = os.path.join(base_folder, file)
            output_path = os.path.join(base_folder, 'outputs', f'{i+1}_{cat}.avi')
            annotations_folder = os.path.join(base_folder, 'annotation', f'{i+1}_{cat}')
            if not os.path.exists(annotations_folder):
                os.mkdir(annotations_folder)
            output_pos = process_bag(input_path, output_path, annotations_folder)
            output_pos = np.array(output_pos)[:,:2]
            output_pos[:,1] *= -1

        #     plt.plot(output_pos[:,0], output_pos[:,1], label=cat_map[cat], **mpl_settings[cat])
        #
        # plt.legend()
        # plt.axis('equal')
        # plt.show()