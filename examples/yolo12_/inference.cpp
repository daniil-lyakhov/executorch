#include "inference.h"
#include <csignal>

Inference::Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape, const std::string &classesTxtFile, const bool &runWithCuda)
{
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    classesPath = classesTxtFile;
    cudaEnabled = runWithCuda;

    // loadClassesFromFile(); The classes are hard-coded for this example
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

void set_method_input(
    Result<Method> &method, std::vector<EValue> &model_inputs,
    const cv::Mat input_tensor) {
    const MethodMeta method_meta = method->method_meta();

    ET_CHECK_MSG(
        method->inputs_size() == 1,
        "The given method has too many inputs: %ld",
        method->inputs_size()
    );

    const int input_index = 0;
    Result<TensorInfo> tensor_meta =
        method_meta.input_tensor_meta(input_index);
    auto input_data_ptr = model_inputs[input_index].toTensor().data_ptr<uchar>();
    memcpy(static_cast<uchar *>(input_data_ptr), static_cast<uchar *>(input_tensor.data), tensor_meta->nbytes());
}

std::vector<Detection> Inference::runInference(const cv::Mat &input)
{

    cv::Mat modelInput = input;
    int pad_x, pad_y;
    float scale;
    if (letterBoxForSquare && modelShape.width == modelShape.height)
        modelInput = formatToSquare(modelInput, &pad_x, &pad_y, &scale);

    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, modelShape, cv::Scalar(), true, false);


    executorch::runtime::runtime_init();
    Result<FileDataLoader> loader = FileDataLoader::from(this->modelPath.c_str());
    ET_CHECK_MSG(
        loader.ok(),
        "FileDataLoader::from() failed: 0x%" PRIx32,
        (uint32_t)loader.error());

    // Parse the program file. This is immutable, and can also be reused between
    // multiple execution invocations across multiple threads.
    Result<Program> program = Program::load(&loader.get());
    if (!program.ok()) {
      ET_LOG(Error, "Failed to parse model file %s", this->modelPath.c_str());
      return std::vector<Detection>();
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
    std::cout <<"cccc" << std::endl << std::endl;
    std::cout.flush();
    EventTraceManager tracer;
    Result<Method> method = program->load_method(
        method_name, &memory_manager, tracer.get_event_tracer());
    std::cout <<"dddd" << std::endl << (uint32_t)method.error() <<std::endl;
    std::cout.flush();
    ET_CHECK_MSG(
        method.ok(),
        "Loading of method %s failed with status 0x%" PRIx32,
        method_name,
        (uint32_t)method.error());
    ET_LOG(Info, "Method loaded.");
    std::cout <<"BBBBBBBBBB" << std::endl << modelPath <<std::endl;
    std::cout.flush();

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
    //ET_CHECK_MSG(
    //    method->outputs_size() == 1,
    //    "The given method has too many outputs: %ld, 1 expected.",
    //    method->outputs_size()
    //);
    std::vector<EValue> inputs(method->inputs_size());
    ET_LOG(Info, "Number of input layers: %zu", inputs.size());

    Error status = method->get_inputs(inputs.data(), inputs.size());
    ET_CHECK(status == Error::Ok);

    set_method_input(method, inputs, blob);

    status = method->execute();

    ET_CHECK_MSG(
        status == Error::Ok,
        "Execution of method %s failed with status 0x%" PRIx32,
        method_name,
        (uint32_t)status);

    std::vector<EValue> ex_outputs(method->outputs_size());
    ET_LOG(Info, "%zu outputs: ", ex_outputs.size());
    status = method->get_outputs(ex_outputs.data(), ex_outputs.size());
    ET_CHECK(status == Error::Ok);
    //std::raise(SIGINT);

    //const auto output_size = ex_outputs[0].toTensor().sizes();
    //const auto mat_output = cv::Mat(output_size[0], output_size[1], CV_32FC1, ex_outputs[0].toTensor().data_ptr());
    const auto t = ex_outputs[0].toTensor();
    // Copy etensor to a Mat
    const auto mat_output = cv::Mat(t.dim(), t.sizes().data(), CV_32FC1, t.data_ptr());
    auto outputs = std::vector<cv::Mat>();
    outputs.push_back(mat_output);
    //std::vector<cv::Mat> outputs;
    //net.forward(outputs, net.getUnconnectedOutLayersNames());

    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    bool yolov8 = false;
    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        yolov8 = true;
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];

        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }
    float *data = (float *)outputs[0].data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        if (yolov8)
        {
            float *classes_scores = data+4;

            cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > modelScoreThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w - pad_x) / scale);
                int top = int((y - 0.5 * h - pad_y) / scale);

                int width = int(w / scale);
                int height = int(h / scale);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        else // yolov5
        {
            float confidence = data[4];

            if (confidence >= modelConfidenceThreshold)
            {
                float *classes_scores = data+5;

                cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;

                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > modelScoreThreshold)
                {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w - pad_x) / scale);
                    int top = int((y - 0.5 * h - pad_y) / scale);

                    int width = int(w / scale);
                    int height = int(h / scale);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    std::vector<Detection> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = classes[result.class_id];
        result.box = boxes[idx];

        detections.push_back(result);
    }

    return detections;
}

void Inference::loadClassesFromFile()
{
    std::ifstream inputFile(classesPath);
    if (inputFile.is_open())
    {
        std::string classLine;
        while (std::getline(inputFile, classLine))
            classes.push_back(classLine);
        inputFile.close();
    }
}

void Inference::loadOnnxNetwork()
{
    net = cv::dnn::readNetFromONNX(modelPath);
    if (cudaEnabled)
    {
        std::cout << "\nRunning on CUDA" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "\nRunning on CPU" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

cv::Mat Inference::formatToSquare(const cv::Mat &source, int *pad_x, int *pad_y, float *scale)
{
    int col = source.cols;
    int row = source.rows;
    int m_inputWidth = modelShape.width;
    int m_inputHeight = modelShape.height;

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
