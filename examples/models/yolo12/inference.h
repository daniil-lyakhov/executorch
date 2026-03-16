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

  const auto t = result->at(0).toTensor(); // Using only the 0 output
  // yolo26 has an end-to-end (post-NMS) output of shape (batchSize, N, 6)
  // Each detection row: [x1, y1, x2, y2, confidence, class_id]
  const int num_detections = t.sizes()[1];
  const int num_classes = static_cast<int>(yolo_config.classes.size());
  const float* data = static_cast<const float*>(t.const_data_ptr());

  std::vector<Detection> detections{};
  for (int i = 0; i < num_detections; ++i) {
    const float* det = data + i * 6;
    const float confidence = det[4];
    const int class_id = static_cast<int>(det[5]);

    // Check if the detection meets the confidence threshold
    if (confidence <= yolo_config.modelScoreThreshold)
      continue;

    if (class_id < 0 || class_id >= num_classes)
      continue;

    // Map coordinates back to original image space
    const int left = static_cast<int>((det[0] - pad_x) / scale);
    const int top = static_cast<int>((det[1] - pad_y) / scale);
    const int width = static_cast<int>((det[2] - det[0]) / scale);
    const int height = static_cast<int>((det[3] - det[1]) / scale);

    Detection result;
    result.class_id = class_id;
    result.confidence = confidence;
    result.className = yolo_config.classes[class_id];
    result.box = cv::Rect(left, top, width, height);
    detections.push_back(result);
  }

  return detections;
}
#endif // INFERENCE_H
