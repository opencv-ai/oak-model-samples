#!/bin/bash

MODELS=(
"landmarks_regression_retail;draw_landmarks_one_frame"
"emotion_recognition_retail;draw_emotion_recognition_one_frame"
"face_detection_adas;draw_detections_one_frame"
"age_gender_recognition_retail;draw_age_gender_recognition_one_frame"
"mobilenet_ssd;draw_detections_one_frame"
"openpose;draw_keypoints_one_frame"
"pedestrian_detection_adas;draw_detections_one_frame"
"palm_detection;draw_landmarks_one_frame"
"dbface;draw_landmarks_one_frame"
"person_detection_retail;draw_detections_one_frame"
"vehicle_license_plate_detection_barrier;draw_detections_one_frame"
"vehicle_detection_adas;draw_detections_one_frame"
"east_text_detector;draw_text_detections_one_frame"
"facial_landmarks_35_adas;draw_landmarks_one_frame"
"face_detection_retail;draw_detections_one_frame"
"lightweight_openpose;draw_keypoints_one_frame"
"textboxes_plus_plus;draw_text_detections_one_frame"
"person_vehicle_bike_detection_crossroad;draw_detections_one_frame"
"tiny_yolo_v3;draw_detections_one_frame"
"yolov4_tiny;draw_detections_one_frame"
"hand_pose_estimation;draw_keypoints_one_frame"
)
for i in "${MODELS[@]}"
do
  IFS=\; read model vis <<< $i
  pushd $PWD/$model
  echo "Installing $model"
  python3 setup.py bdist_wheel && rm -R build/ *.egg-info && pip3 install dist/*.whl && rm -R dist/
  python3 ../create_gif.py $model $vis
  echo "Uninstalling $model"
  pip3 uninstall -y $model
  popd
done
