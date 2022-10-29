#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/optional_debug_tools.h"

std::vector<std::string> load_labels(std::string labels_file)
{
    std::ifstream file(labels_file.c_str());
    if (!file.is_open())
    {   
        fprintf(stderr, "unable to open label file\n");
        exit(-1);
    }   
    std::string label_str;
    std::vector<std::string> labels;

    while (std::getline(file, label_str))
    {   
        if (label_str.size() > 0)
            labels.push_back(label_str);
    }   
    file.close();
    return labels;
}

int main(int argc, char **argv)
{

    // Get Model label and input image
    if (argc != 3)
    {   
        fprintf(stderr, "tflite_gpu modelfile image\n");
        exit(-1);
    }   
    const char *modelFileName = argv[1];
    const char *imageFile = argv[2];

    // Load Model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelFileName);
    if (model == nullptr)
    {
        fprintf(stderr, "failed to load model\n");
        exit(-1);
    }
    // Initiate Interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    if (interpreter == nullptr)
    {
        fprintf(stderr, "Failed to initiate the interpreter\n");
        exit(-1);
    }

    auto *delegate = TfLiteGpuDelegateV2Create(nullptr);
    interpreter->ModifyGraphWithDelegate(delegate);

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf(stderr, "Failed to allocate tensor\n");
        exit(-1);
    }

    //tflite::PrintInterpreterState(interpreter.get());
    // Configure the interpreter
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(1);
    // Get Input Tensor Dimensions
    int input = interpreter->inputs()[0];
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];
    auto channels = interpreter->tensor(input)->dims->data[3];
    // Load Input Image
    cv::Mat image;
    auto frame = cv::imread(imageFile);
    if (frame.empty())
    {
        fprintf(stderr, "Failed to load iamge\n");
        exit(-1);
    }

    // Copy image to input tensor
    cv::cvtColor(frame, image, cv::COLOR_BGR2RGB);
    cv::resize(frame, image, cv::Size(width, height), cv::INTER_LINEAR);
    image.convertTo(image, CV_32FC3);
    image = image / 255;
    memcpy(interpreter->typed_input_tensor<float>(0), image.data, image.total() * image.elemSize());

    // Inference
    std::chrono::steady_clock::time_point start, end;
    start = std::chrono::steady_clock::now();
    interpreter->Invoke();
    end = std::chrono::steady_clock::now();
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Get Output
    const auto& output_indices = interpreter->outputs();
    const int num_outputs = output_indices.size();

   for (int i = 0; i < num_outputs; ++i) {
        const auto* out_tensor = interpreter->tensor(output_indices[i]);
        if (out_tensor->type == kTfLiteFloat32) {
            const int num_values = out_tensor->bytes/ sizeof(float);
            const float* output = interpreter->typed_output_tensor<float>(i);
            printf("out[%d]: num_values = %d\n", i, num_values);

            for (int j = 0; j < 10; j++)
                printf("%2f ", output[j]);
            printf("\n");
        }
    }
    int output = interpreter->outputs()[0];
    TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
    int output_2 = interpreter->outputs()[1];
    TfLiteIntArray *output_dims_2 = interpreter->tensor(output_2)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];
    std::vector<std::pair<float, int>> top_results;
    float threshold = 0.01f;
    printf("out 1 size = %d, out 2 size = %d\n", output_dims->size, output_dims_2->size);
    printf("tpye = %d\n", interpreter->tensor(output)->type);
    // Print inference ms in input image
    cv::putText(frame, "Infernce Time in ms: " + std::to_string(inference_time), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

    // Display image
    cv::imshow("Output", frame);
    cv::waitKey(0);

    TfLiteGpuDelegateV2Delete(delegate);
    return 0;
}
