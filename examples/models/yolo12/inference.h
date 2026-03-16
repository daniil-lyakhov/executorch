#ifndef INFERENCE_H
#define INFERENCE_H

#include <iostream>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <opencv2/opencv.hpp>

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::Result;

struct Detection {
  int class_id{0};
  std::string className{};
  float confidence{0.0};
  cv::Rect box{};
};

struct DetectionConfig {
  std::vector<std::string> classes;
  float modelScoreThreshold;
  float modelNMSThreshold;
};

cv::Mat scale_with_padding(
    cv::Mat& source,
    int* pad_x,
    int* pad_y,
    float* scale,
    cv::Size img_dims) {
  int col = source.cols;
  int row = source.rows;
  int m_inputWidth = img_dims.width;
  int m_inputHeight = img_dims.height;
  if (col == m_inputWidth and row == m_inputHeight) {
    return source;
  }

  *scale = std::min(m_inputWidth / (float)col, m_inputHeight / (float)row);
  int resized_w = col * *scale;
  int resized_h = row * *scale;
  *pad_x = (m_inputWidth - resized_w) / 2;
  *pad_y = (m_inputHeight - resized_h) / 2;

  cv::Mat resized;
  cv::resize(source, resized, cv::Size(resized_w, resized_h));
  cv::Mat result = cv::Mat::zeros(m_inputHeight, m_inputWidth, source.type());
  resized.copyTo(result(cv::Rect(*pad_x, *pad_y, resized_w, resized_h)));
  resized.release();
  return result;
}

// Post-NMS (end2end) output format: [1, N, 6] where each row is
// [x1, y1, x2, y2, confidence, class_id]
std::vector<Detection> parse_end2end_output(
    const float* data,
    int num_detections,
    int pad_x,
    int pad_y,
    float scale,
    const DetectionConfig& yolo_config) {
  std::vector<Detection> detections;
  const int num_classes = static_cast<int>(yolo_config.classes.size());

  for (int i = 0; i < num_detections; ++i) {
    const float* det = data + i * 6;
    const float x1 = det[0];
    const float y1 = det[1];
    const float x2 = det[2];
    const float y2 = det[3];
    const float confidence = det[4];
    const int class_id = static_cast<int>(det[5]);

    if (confidence <= yolo_config.modelScoreThreshold)
      continue;

    if (class_id < 0 || class_id >= num_classes)
      continue;

    // Map coordinates back to original image space
    const int left = static_cast<int>((x1 - pad_x) / scale);
    const int top = static_cast<int>((y1 - pad_y) / scale);
    const int width = static_cast<int>((x2 - x1) / scale);
    const int height = static_cast<int>((y2 - y1) / scale);

    Detection result;
    result.class_id = class_id;
    result.confidence = confidence;
    result.className = yolo_config.classes[class_id];
    result.box = cv::Rect(left, top, width, height);
    detections.push_back(result);
  }

  return detections;
}

// Pre-NMS (classic) output format: [1, num_classes+4, num_anchors]
// e.g. [1, 84, 8400] for 80 COCO classes
std::vector<Detection> parse_classic_output(
    const executorch::aten::Tensor& t,
    int pad_x,
    int pad_y,
    float scale,
    const DetectionConfig& yolo_config) {
  cv::Mat mat_output(
      t.dim() - 1, t.sizes().data() + 1, CV_32FC1, t.data_ptr());

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int i = 0; i < mat_output.cols; ++i) {
    const cv::Mat classes_scores =
        mat_output.col(i).rowRange(4, mat_output.rows);

    cv::Point class_id;
    double score;
    cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id);

    if (score <= yolo_config.modelScoreThreshold)
      continue;

    class_ids.push_back(class_id.y);
    confidences.push_back(score);

    const float x = mat_output.at<float>(0, i);
    const float y = mat_output.at<float>(1, i);
    const float w = mat_output.at<float>(2, i);
    const float h = mat_output.at<float>(3, i);

    const int left = int((x - 0.5 * w - pad_x) / scale);
    const int top = int((y - 0.5 * h - pad_y) / scale);
    const int width = int(w / scale);
    const int height = int(h / scale);

    boxes.push_back(cv::Rect(left, top, width, height));
  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(
      boxes,
      confidences,
      yolo_config.modelScoreThreshold,
      yolo_config.modelNMSThreshold,
      nms_result);

  std::vector<Detection> detections;
  const int num_classes = static_cast<int>(yolo_config.classes.size());
  for (auto& idx : nms_result) {
    if (class_ids[idx] < 0 || class_ids[idx] >= num_classes)
      continue;
    Detection result;
    result.class_id = class_ids[idx];
    result.confidence = confidences[idx];
    result.className = yolo_config.classes[result.class_id];
    result.box = boxes[idx];
    detections.push_back(result);
  }

  return detections;
}

std::vector<Detection> infer_yolo_once(
    Module& module,
    cv::Mat input,
    cv::Size img_dims,
    const DetectionConfig yolo_config) {
  int pad_x, pad_y;
  float scale;
  input = scale_with_padding(input, &pad_x, &pad_y, &scale, img_dims);

  cv::Mat blob;
  cv::dnn::blobFromImage(
      input, blob, 1.0 / 255.0, img_dims, cv::Scalar(), true, false);
  const auto t_input = from_blob(
      (void*)blob.data,
      std::vector<int>(blob.size.p, blob.size.p + blob.dims),
      ScalarType::Float);
  const auto result = module.forward(t_input);

  ET_CHECK_MSG(
      result.ok(),
      "Execution of method forward failed with status 0x%" PRIx32,
      (uint32_t)result.error());

  const auto t = result->at(0).toTensor();

  // Detect output format based on tensor shape:
  // End-to-end (post-NMS): [1, N, 6] where last dim is
  //   [x1, y1, x2, y2, confidence, class_id]
  // Classic (pre-NMS): [1, num_classes+4, num_anchors] where dim 1 >> 6
  if (t.dim() == 3 && t.sizes()[2] == 6) {
    // End-to-end post-NMS format: [batch, num_detections, 6]
    const int num_detections = t.sizes()[1];
    return parse_end2end_output(
        static_cast<const float*>(t.const_data_ptr()),
        num_detections,
        pad_x,
        pad_y,
        scale,
        yolo_config);
  } else {
    // Classic pre-NMS format: [batch, 84, 8400] etc.
    return parse_classic_output(t, pad_x, pad_y, scale, yolo_config);
  }
}
#endif // INFERENCE_H
