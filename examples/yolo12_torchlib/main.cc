#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <random>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

using torch::indexing::Slice;
using torch::indexing::None;

using executorch::extension::FileDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::EventTracer;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;

float generate_scale(cv::Mat& image, const std::vector<int>& target_size) {
    int origin_w = image.cols;
    int origin_h = image.rows;

    int target_h = target_size[0];
    int target_w = target_size[1];

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);
    return resize_scale;
}


float letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size) {
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
        if (input_image.data == output_image.data) {
            return 1.;
        } else {
            output_image = input_image.clone();
            return 1.;
        }
    }

    float resize_scale = generate_scale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.;
    float padh = (target_size[0] - new_shape_h) / 2.;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::resize(input_image, output_image,
               cv::Size(new_shape_w, new_shape_h),
               0, 0, cv::INTER_AREA);

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114., 114., 114));
    return resize_scale;
}


torch::Tensor xyxy2xywh(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    y.index_put_({"...", 0}, (x.index({"...", 0}) + x.index({"...", 2})).div(2));
    y.index_put_({"...", 1}, (x.index({"...", 1}) + x.index({"...", 3})).div(2));
    y.index_put_({"...", 2}, x.index({"...", 2}) - x.index({"...", 0}));
    y.index_put_({"...", 3}, x.index({"...", 3}) - x.index({"...", 1}));
    return y;
}


torch::Tensor xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    auto dw = x.index({"...", 2}).div(2);
    auto dh = x.index({"...", 3}).div(2);
    y.index_put_({"...", 0}, x.index({"...", 0}) - dw);
    y.index_put_({"...", 1}, x.index({"...", 1}) - dh);
    y.index_put_({"...", 2}, x.index({"...", 0}) + dw);
    y.index_put_({"...", 3}, x.index({"...", 1}) + dh);
    return y;
}


// Reference: https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold) {
    if (bboxes.numel() == 0)
        return torch::empty({0}, bboxes.options().dtype(torch::kLong));

    auto x1_t = bboxes.select(1, 0).contiguous();
    auto y1_t = bboxes.select(1, 1).contiguous();
    auto x2_t = bboxes.select(1, 2).contiguous();
    auto y2_t = bboxes.select(1, 3).contiguous();

    torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

    auto order_t = std::get<1>(
        scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));

    auto ndets = bboxes.size(0);
    torch::Tensor suppressed_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kByte));
    torch::Tensor keep_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kLong));

    auto suppressed = suppressed_t.data_ptr<uint8_t>();
    auto keep = keep_t.data_ptr<int64_t>();
    auto order = order_t.data_ptr<int64_t>();
    auto x1 = x1_t.data_ptr<float>();
    auto y1 = y1_t.data_ptr<float>();
    auto x2 = x2_t.data_ptr<float>();
    auto y2 = y2_t.data_ptr<float>();
    auto areas = areas_t.data_ptr<float>();

    int64_t num_to_keep = 0;

    for (int64_t _i = 0; _i < ndets; _i++) {
        auto i = order[_i];
        if (suppressed[i] == 1)
            continue;
        keep[num_to_keep++] = i;
        auto ix1 = x1[i];
        auto iy1 = y1[i];
        auto ix2 = x2[i];
        auto iy2 = y2[i];
        auto iarea = areas[i];

        for (int64_t _j = _i + 1; _j < ndets; _j++) {
        auto j = order[_j];
        if (suppressed[j] == 1)
            continue;
        auto xx1 = std::max(ix1, x1[j]);
        auto yy1 = std::max(iy1, y1[j]);
        auto xx2 = std::min(ix2, x2[j]);
        auto yy2 = std::min(iy2, y2[j]);

        auto w = std::max(static_cast<float>(0), xx2 - xx1);
        auto h = std::max(static_cast<float>(0), yy2 - yy1);
        auto inter = w * h;
        auto ovr = inter / (iarea + areas[j] - inter);
        if (ovr > iou_threshold)
            suppressed[j] = 1;
        }
    }
    return keep_t.narrow(0, 0, num_to_keep);
}


torch::Tensor non_max_suppression(torch::Tensor& prediction, float conf_thres = 0.25, float iou_thres = 0.45, int max_det = 300) {
    auto bs = prediction.size(0);
    auto nc = prediction.size(1) - 4;
    auto nm = prediction.size(1) - nc - 4;
    auto mi = 4 + nc;
    auto xc = prediction.index({Slice(), Slice(4, mi)}).amax(1) > conf_thres;

    prediction = prediction.transpose(-1, -2);
    prediction.index_put_({"...", Slice({None, 4})}, xywh2xyxy(prediction.index({"...", Slice(None, 4)})));

    std::vector<torch::Tensor> output;
    for (int i = 0; i < bs; i++) {
        output.push_back(torch::zeros({0, 6 + nm}));
    }

    for (int xi = 0; xi < prediction.size(0); xi++) {
        auto x = prediction[xi];
        x = x.index({xc[xi]});
        auto x_split = x.split({4, nc, nm}, 1);
        auto box = x_split[0], cls = x_split[1], mask = x_split[2];
        auto [conf, j] = cls.max(1, true);
        x = torch::cat({box, conf, j.toType(torch::kFloat), mask}, 1);
        x = x.index({conf.view(-1) > conf_thres});
        int n = x.size(0);
        if (!n) { continue; }

        // NMS
        auto c = x.index({Slice(), Slice{5, 6}}) * 7680;
        auto boxes = x.index({Slice(), Slice(None, 4)}) + c;
        auto scores = x.index({Slice(), 4});
        auto i = nms(boxes, scores, iou_thres);
        i = i.index({Slice(None, max_det)});
        output[xi] = x.index({i});
    }

    return torch::stack(output);
}



torch::Tensor scale_boxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape) {
    auto gain = (std::min)((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    auto pad0 = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad1 = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);

    boxes.index_put_({"...", 0}, boxes.index({"...", 0}) - pad0);
    boxes.index_put_({"...", 2}, boxes.index({"...", 2}) - pad0);
    boxes.index_put_({"...", 1}, boxes.index({"...", 1}) - pad1);
    boxes.index_put_({"...", 3}, boxes.index({"...", 3}) - pad1);
    boxes.index_put_({"...", Slice(None, 4)}, boxes.index({"...", Slice(None, 4)}).div(gain));
    return boxes;
}


void draw_detected_object(cv::Mat &frame, const int class_id, const std::string class_name, const cv::Rect box, const float confidence) {
	// Generate a random color for the bounding box
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(120, 255);
	const cv::Scalar &color = cv::Scalar(dis(gen), dis(gen), dis(gen));

	// Draw the bounding box around the detected object
	cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), color, 3);

	// Prepare the class label and confidence text
	const auto classString = class_name + std::to_string(confidence).substr(0, 4);

	// Get the size of the text box
	cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 0.75, 2, 0);
	cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

	// Draw the text box
	cv::rectangle(frame, textBox, color, cv::FILLED);

	// Put the class label and confidence text above the bounding box
	cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 2, 0);

}

void set_method_input(
    Result<Method> &method, std::vector<EValue> &model_inputs,
    const  torch::Tensor input_tensor) {
   const MethodMeta method_meta = method->method_meta();

    ET_CHECK_MSG(
        method->inputs_size() == 1,
        "The given method has too many inputs: %ld",
        method->inputs_size()
    );

    const int input_index = 0;
    Result<TensorInfo> tensor_meta =
        method_meta.input_tensor_meta(input_index);
    auto input_data_ptr = model_inputs[input_index].toTensor().data_ptr<char>();
    memcpy(static_cast<char *>(input_data_ptr), static_cast<char *>(input_tensor.data_ptr()), tensor_meta->nbytes());
}




class EventTraceManager {
 public:
  EventTraceManager() : event_tracer_ptr_(nullptr) {
  }

  EventTracer* get_event_tracer() const {
    return event_tracer_ptr_.get();
  };

  Error write_etdump_to_file() const {
    EventTracer* const event_tracer_ptr = get_event_tracer();
    if (!event_tracer_ptr) {
      return Error::NotSupported;
    }

    return Error::Ok;
  }

 private:
  std::shared_ptr<EventTracer> event_tracer_ptr_;
};

//int main(){
//    std::cout << "AAAAA";
//    return 0;
//}
int main() {
    // Device
    std::cout << "AAAAA";
    std::cout.flush();

    //torch::Device device(torch::cuda::is_available() ? torch::kCUDA :torch::kCPU);

    // Note that in this example the classes are hard-coded
    std::vector<std::string> classes {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                                      "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                                      "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                                      "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                                      "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                                      "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

    try {
        const std::string model_path = "yolo12s.pte";
        executorch::runtime::runtime_init();
        Result<FileDataLoader> loader = FileDataLoader::from(model_path.c_str());
        ET_CHECK_MSG(
            loader.ok(),
            "FileDataLoader::from() failed: 0x%" PRIx32,
            (uint32_t)loader.error());

        // Parse the program file. This is immutable, and can also be reused between
        // multiple execution invocations across multiple threads.
        Result<Program> program = Program::load(&loader.get());
        if (!program.ok()) {
          ET_LOG(Error, "Failed to parse model file %s", model_path.c_str());
          return 1;
        }

        // Use the first method in the program.
        const char* method_name = nullptr;
        {
          const auto method_name_result = program->get_method_name(0);
          ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
          method_name = *method_name_result;
        }
        ET_LOG(Info, "Using method %s", method_name);

        // MethodMeta describes the memory requirements of the method.
        Result<MethodMeta> method_meta = program->method_meta(method_name);
        ET_CHECK_MSG(
            method_meta.ok(),
            "Failed to get method_meta for %s: 0x%" PRIx32,
            method_name,
            (uint32_t)method_meta.error());

        //
        // The runtime does not use malloc/new; it allocates all memory using the
        // MemoryManger provided by the client. Clients are responsible for allocating
        // the memory ahead of time, or providing MemoryAllocator subclasses that can
        // do it dynamically.
        //

        // The method allocator is used to allocate all dynamic C++ metadata/objects
        // used to represent the loaded method. This allocator is only used during
        // loading a method of the program, which will return an error if there was
        // not enough memory.
        //
        // The amount of memory required depends on the loaded method and the runtime
        // code itself. The amount of memory here is usually determined by running the
        // method and seeing how much memory is actually used, though it's possible to
        // subclass MemoryAllocator so that it calls malloc() under the hood (see
        // MallocMemoryAllocator).
        //
        // In this example we use a statically allocated memory pool.
        static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB
        static uint8_t temp_allocator_pool[1024U * 1024U];
        MemoryAllocator method_allocator{
            MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};

        // Temporary memory required by kernels
        MemoryAllocator temp_allocator{
            MemoryAllocator(sizeof(temp_allocator_pool), temp_allocator_pool)};

        // The memory-planned buffers will back the mutable tensors used by the
        // method. The sizes of these buffers were determined ahead of time during the
        // memory-planning pasees.
        //
        // Each buffer typically corresponds to a different hardware memory bank. Most
        // mobile environments will only have a single buffer. Some embedded
        // environments may have more than one for, e.g., slow/large DRAM and
        // fast/small SRAM, or for memory associated with particular cores.
        std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the memory
        std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
        size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
        for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
          // .get() will always succeed because id < num_memory_planned_buffers.
          size_t buffer_size =
              static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
          ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
          planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
          planned_spans.push_back({planned_buffers.back().get(), buffer_size});
        }
        HierarchicalAllocator planned_memory(
            {planned_spans.data(), planned_spans.size()});

        // Assemble all of the allocators into the MemoryManager that the Executor
        // will use.
        MemoryManager memory_manager(
            &method_allocator, &planned_memory, &temp_allocator);

        //
        // Load the method from the program, using the provided allocators. Running
        // the method can mutate the memory-planned buffers, so the method should only
        // be used by a single thread at at time, but it can be reused.
        //
        std::cout << "BBBBB";
        std::cout.flush();
        EventTraceManager tracer;
        Result<Method> method = program->load_method(
            method_name, &memory_manager, tracer.get_event_tracer());
        std::cout << "DDDD" << std::endl << (uint32_t)method.error();
        std::cout.flush();
        ET_CHECK_MSG(
            method.ok(),
            "Loading of method %s failed with status 0x%" PRIx32,
            method_name,
            (uint32_t)method.error());
        ET_LOG(Info, "Method loaded.");

        et_timestamp_t time_spent_executing = 0;
        // Run the model.
        ET_LOG(Debug, "Preparing inputs.");
        auto method_inputs = executorch::extension::prepare_input_tensors(*method);
        ET_CHECK_MSG(
            method_inputs.ok(),
            "Could not prepare inputs: 0x%" PRIx32,
            (uint32_t)method_inputs.error());
        ET_LOG(Debug, "Inputs prepared.");
        ET_CHECK_MSG(
            method->inputs_size() == 1,
            "The given method has too many inputs: %ld, 1 expected.",
            method->inputs_size()
        );
        ET_CHECK_MSG(
            method->outputs_size() == 1,
            "The given method has too many outputs: %ld, 1 expected.",
            method->outputs_size()
        );
        std::vector<EValue> inputs(method->inputs_size());
        ET_LOG(Info, "Number of input layers: %zu", inputs.size());

        Error status = method->get_inputs(inputs.data(), inputs.size());
        ET_CHECK(status == Error::Ok);

        // Load image and preprocess
        cv::Mat image = cv::imread("cat.jpg");
        cv::Mat input_image;
        letterbox(image, input_image, {640, 640});
        cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

        //torch::Tensor image_tensor = torch::from_blob(input_image.data, {input_image.rows, input_image.cols, 3}, torch::kByte).to(device);
        torch::Tensor image_tensor = torch::from_blob(input_image.data, {input_image.rows, input_image.cols, 3}, torch::kByte);
        image_tensor = image_tensor.toType(torch::kFloat32).div(255);
        image_tensor = image_tensor.permute({2, 0, 1});
        image_tensor = image_tensor.unsqueeze(0);
        set_method_input(method, inputs, image_tensor);

        status = method->execute();

        ET_CHECK_MSG(
            status == Error::Ok,
            "Execution of method %s failed with status 0x%" PRIx32,
            method_name,
            (uint32_t)status);

        std::vector<EValue> outputs(method->outputs_size());
        ET_LOG(Info, "%zu outputs: ", outputs.size());
        status = method->get_outputs(outputs.data(), outputs.size());
        ET_CHECK(status == Error::Ok);

        // Inference
        torch::Tensor output = output[0];

        // NMS
        auto keep = non_max_suppression(output)[0];
        auto boxes = keep.index({Slice(), Slice(None, 4)});
        keep.index_put_({Slice(), Slice(None, 4)}, scale_boxes({input_image.rows, input_image.cols}, boxes, {image.rows, image.cols}));

        // Show the results
        for (int i = 0; i < keep.size(0); i++) {
            int x1 = keep[i][0].item().toFloat();
            int y1 = keep[i][1].item().toFloat();
            int x2 = keep[i][2].item().toFloat();
            int y2 = keep[i][3].item().toFloat();
            float conf = keep[i][4].item().toFloat();
            int cls = keep[i][5].item().toInt();
            std::cout << "Rect: [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]  Conf: " << conf << "  Class: " << classes[cls] << std::endl;

            const auto box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            draw_detected_object(image, cls, classes[cls], box, conf);
            cv::imwrite("out.jpg", image);
        }
    } catch (const c10::Error& e) {
        std::cout << e.msg() << std::endl;
    }

    return 0;
}