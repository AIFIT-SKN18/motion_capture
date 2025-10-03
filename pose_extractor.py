import cv2
import mediapipe as mp
import numpy as np


class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # NTU RGB+D 25개 관절점 순서 (0-based index)
        self.ntu_joint_names = [
            'spine_base',      # 0  - hip center
            'spine_mid',       # 1  - spine
            'neck',            # 2  - neck
            'head',            # 3  - head
            'left_shoulder',   # 4
            'left_elbow',      # 5
            'left_wrist',      # 6
            'left_hand',       # 7
            'right_shoulder',  # 8
            'right_elbow',     # 9
            'right_wrist',     # 10
            'right_hand',      # 11
            'left_hip',        # 12
            'left_knee',       # 13
            'left_ankle',      # 14
            'left_foot',       # 15
            'right_hip',       # 16
            'right_knee',      # 17
            'right_ankle',     # 18
            'right_foot',      # 19
            'spine_shoulder',  # 20 - shoulder center
            'left_hand_tip',   # 21
            'left_thumb',      # 22
            'right_hand_tip',  # 23
            'right_thumb'      # 24
        ]

    def mediapipe_to_ntu(self, pose_landmarks):
        """
        MediaPipe 포즈를 NTU RGB+D 25개 관절점으로 변환
        """
        joints = np.zeros((25, 3))
        landmarks = pose_landmarks.landmark


        # 0: spine_base (hip center) - 양쪽 힙의 중점
        joints[0] = [(landmarks[23].x + landmarks[24].x) / 2,
                     (landmarks[23].y + landmarks[24].y) / 2,
                     (landmarks[23].z + landmarks[24].z) / 2]


        hip_center = joints[0]
        shoulder_center = [(landmarks[11].x + landmarks[12].x) / 2,
                          (landmarks[11].y + landmarks[12].y) / 2,
                          (landmarks[11].z + landmarks[12].z) / 2]
        joints[1] = [(hip_center[0] + shoulder_center[0]) / 2,
                     (hip_center[1] + shoulder_center[1]) / 2,
                     (hip_center[2] + shoulder_center[2]) / 2]

        joints[2] = [(landmarks[11].x + landmarks[12].x) / 2,
                     (landmarks[11].y + landmarks[12].y) / 2 - 0.05,  # 약간 위로
                     (landmarks[11].z + landmarks[12].z) / 2]

        # 3: head - nose
        joints[3] = [landmarks[0].x, landmarks[0].y, landmarks[0].z]

        # 4: left_shoulder
        joints[4] = [landmarks[11].x, landmarks[11].y, landmarks[11].z]

        # 5: left_elbow
        joints[5] = [landmarks[13].x, landmarks[13].y, landmarks[13].z]

        # 6: left_wrist
        joints[6] = [landmarks[15].x, landmarks[15].y, landmarks[15].z]

        # 7: left_hand - left_wrist와 동일
        joints[7] = [landmarks[15].x, landmarks[15].y, landmarks[15].z]

        # 8: right_shoulder
        joints[8] = [landmarks[12].x, landmarks[12].y, landmarks[12].z]

        # 9: right_elbow
        joints[9] = [landmarks[14].x, landmarks[14].y, landmarks[14].z]

        # 10: right_wrist
        joints[10] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]

        # 11: right_hand - right_wrist와 동일
        joints[11] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]

        # 12: left_hip
        joints[12] = [landmarks[23].x, landmarks[23].y, landmarks[23].z]

        # 13: left_knee
        joints[13] = [landmarks[25].x, landmarks[25].y, landmarks[25].z]

        # 14: left_ankle
        joints[14] = [landmarks[27].x, landmarks[27].y, landmarks[27].z]

        # 15: left_foot
        joints[15] = [landmarks[31].x, landmarks[31].y, landmarks[31].z]

        # 16: right_hip
        joints[16] = [landmarks[24].x, landmarks[24].y, landmarks[24].z]

        # 17: right_knee
        joints[17] = [landmarks[26].x, landmarks[26].y, landmarks[26].z]

        # 18: right_ankle
        joints[18] = [landmarks[28].x, landmarks[28].y, landmarks[28].z]

        # 19: right_foot
        joints[19] = [landmarks[32].x, landmarks[32].y, landmarks[32].z]

        # 20: spine_shoulder - shoulder center
        joints[20] = shoulder_center

        # 21: left_hand_tip - left_pinky
        joints[21] = [landmarks[17].x, landmarks[17].y, landmarks[17].z]

        # 22: left_thumb
        joints[22] = [landmarks[21].x, landmarks[21].y, landmarks[21].z]

        # 23: right_hand_tip - right_pinky
        joints[23] = [landmarks[18].x, landmarks[18].y, landmarks[18].z]

        # 24: right_thumb
        joints[24] = [landmarks[22].x, landmarks[22].y, landmarks[22].z]

        return joints

    def extract_from_video(self, video_path, max_frames=300):
        """
        비디오에서 NTU 형식의 스켈레톤 시퀀스 추출
        """
        cap = cv2.VideoCapture(video_path)
        skeleton_sequence = []
        frame_count = 0

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                joints = self.mediapipe_to_ntu(results.pose_landmarks)
                skeleton_sequence.append(joints)
            else:
                # 포즈를 찾지 못한 경우 이전 프레임 복사
                if skeleton_sequence:
                    skeleton_sequence.append(skeleton_sequence[-1].copy())
                else:
                    skeleton_sequence.append(np.zeros((25, 3)))

            frame_count += 1

        cap.release()

        if not skeleton_sequence:
            return np.zeros((1, 25, 3))

        return np.array(skeleton_sequence)

    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()
