#include <exception>
#include <iostream>
#include <chrono>
#include <functional>
#include <algorithm>
#include <deque>
#include <thread>
#include <mutex>
#include <iomanip>
#include <condition_variable>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>

#include "cpu.h"
#include "gpu.h"
#include "platform.h"
#include "benchmark.h"

#include "rife.h"
#include "realesrgan.h"

#include "filesystem_utils.h"

namespace fs = std::filesystem;

class TaskQueue {
public:
    TaskQueue() {}

    void push_back(ncnn::Mat v) {
        lock.lock();
        while (tasks.size() >= 8) {
            condition.wait(lock);
        }
        tasks.push_back(v);
        lock.unlock();
        condition.signal();
    }

    void pop_front() {
        lock.lock();
        while (tasks.size() == 0) {
            condition.wait(lock);
        }
        tasks.pop_front();
        lock.unlock();
        condition.signal();
    }

    ncnn::Mat front() {
        lock.lock();
        while (tasks.size() == 0) {
            condition.wait(lock);
        }
        ncnn::Mat v = tasks.front();
        lock.unlock();
        condition.signal();
        return v;
    }

    ncnn::Mat back() {
        lock.lock();
        while (tasks.size() == 0) {
            condition.wait(lock);
        }
        ncnn::Mat v = tasks.back();
        lock.unlock();
        condition.signal();
        return v;
    }

    ncnn::Mat at(int idx) {
        lock.lock();
        while (tasks.size() <= idx) {
            condition.wait(lock);
        }
        ncnn::Mat v = tasks.at(idx);
        lock.unlock();
        condition.signal();
        return v;
    }

    bool empty() {
        bool v = true;
        if (tasks.size() > 0) {
            v = false;
        }
        return v;
    }

private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::deque<ncnn::Mat> tasks;
};

struct MediaInfo {
    int width;
    int height;
    float fps;
    int multiplier;
};

class ImageDecoder {
public:
    virtual ncnn::Mat get_next() = 0;
    virtual MediaInfo get_info() = 0;
};

class VideoImageDecoder: public ImageDecoder {
public:
    VideoImageDecoder(const fs::path& video_path)
            : capture_(video_path) {}

    MediaInfo get_info() {
        MediaInfo info;

        info.width      = capture_.get(cv::CAP_PROP_FRAME_WIDTH ); 
        info.height     = capture_.get(cv::CAP_PROP_FRAME_HEIGHT); 
        info.fps        = capture_.get(cv::CAP_PROP_FPS         );
        info.multiplier = 1;

        return info;
    }

    ncnn::Mat get_next() {
        cv::Mat cv_image;
        capture_ >> cv_image;

        unsigned char* raw_data = (unsigned char*)malloc(cv_image.cols * cv_image.rows * cv_image.channels());
        std::move(cv_image.data, cv_image.data + cv_image.cols * cv_image.rows * cv_image.channels(), raw_data);
        ncnn::Mat ncnn_image(cv_image.cols, cv_image.rows, (void*)raw_data, (size_t)3, 3);

        return ncnn_image;
    }

private:
    cv::VideoCapture capture_;
};

class DirImageDecoder: public ImageDecoder {
public:
    DirImageDecoder(const fs::path& dir_path, int fps)
            : dir_path_(dir_path), current_(0), fps_(fps) {}

    MediaInfo get_info() {
        MediaInfo info;

        for (const auto& entry: fs::directory_iterator(dir_path_)) {
            cv::Mat image = cv::imread(entry.path());

            info.width      = image.cols;
            info.height     = image.rows;
            info.fps        = fps_;
            info.multiplier = 1;

            break;
        }

        return info;
    }

    ncnn::Mat get_next() {
        std::ostringstream oss;
        oss << "frame_" << std::setw(8) << std::setfill('0') << current_ + 1 << ".png";
        cv::Mat cv_image = cv::imread(dir_path_ / oss.str());

        unsigned char* raw_data = (unsigned char*)malloc(cv_image.cols * cv_image.rows * cv_image.channels());
        std::move(cv_image.data, cv_image.data + cv_image.cols * cv_image.rows * cv_image.channels(), raw_data);
        ncnn::Mat image(cv_image.cols, cv_image.rows, (void*)raw_data, (size_t)3, 3);

        return image;
    }

private:
    fs::path dir_path_;
    int current_;
    int fps_;
};

class ImageEncoder {
public:
    virtual void put_next(ncnn::Mat image) = 0;
};

class VideoImageEncoder: public ImageEncoder {
public:
    VideoImageEncoder(const fs::path& path, MediaInfo info) {
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        cv::Size size = cv::Size(info.width, info.height);
        writer_ = cv::VideoWriter(path, fourcc, info.fps, size);
    }

    void put_next(ncnn::Mat image) {
        cv::Mat cv_image(image.h, image.w, CV_8UC3, image.data);
        writer_.write(cv_image);
    }

private:
    cv::VideoWriter writer_;
};

class DirImageEncoder: public ImageEncoder {
public:
    DirImageEncoder(const fs::path& dir_path)
            : dir_path_(dir_path), current_(0) {}

    void put_next(ncnn::Mat image) {
        cv::Mat cv_image(image.h, image.w, CV_8UC3, image.data);
        std::ostringstream oss;
        oss << std::setw(8) << std::setfill('0') << current_ + 1 << ".png";
        cv::imwrite(dir_path_ / oss.str(), cv_image);
        ++current_;
    }

private:
    fs::path dir_path_;
    int current_;
};

class Processor {
public:
    virtual MediaInfo get_info(MediaInfo src_info) = 0;
    virtual bool get_result(TaskQueue& load_buff, TaskQueue& save_buff) = 0;
};

class UpscaleProcessor: public Processor {
public:
    UpscaleProcessor(const fs::path& model_dir, int gpu_id) : current_(0) {
        int tilesize = 0;
        uint32_t heap_budget = ncnn::get_gpu_device(gpu_id)->get_heap_budget();

        if (heap_budget > 1900) {
            tilesize = 200;
        } else if (heap_budget > 550) {
            tilesize = 100;
        } else if (heap_budget > 190) {
            tilesize = 64;
        } else {
            tilesize = 32;
        }

        path_t param_path;
        path_t model_path;

        for (const auto& entry: fs::directory_iterator(model_dir)) {
            std::string filename = entry.path().filename().string();

            if (filename.find(".param") != std::string::npos) {
                param_path = sanitize_filepath(model_dir / filename);

            } else if (filename.find(".bin") != std::string::npos) {
                model_path = sanitize_filepath(model_dir / filename);
            }
        }

        std::string dir_name = model_dir.parent_path().filename().string();

        if (dir_name.find("x4") != std::string::npos) {
            scale_ = 4;
        } else if (dir_name.find("x3") != std::string::npos) {
            scale_ = 3;
        } else if (dir_name.find("x2") != std::string::npos) {
            scale_ = 2;
        }

        model_ = new RealESRGAN(gpu_id, false);
        model_->load(param_path, model_path);
        model_->scale = scale_;
        model_->tilesize = tilesize;
        model_->prepadding = 10;
    }

    MediaInfo get_info(MediaInfo src_info) {
        MediaInfo dst_info = src_info;
        dst_info.width *= scale_;
        dst_info.height *= scale_;
        return dst_info;
    }

    bool get_result(TaskQueue& load_buff, TaskQueue& save_buff) {
        ncnn::Mat image = load_buff.front();
        if (image.empty()) {
            return false;
        }
        load_buff.pop_front();

        ncnn::Mat result(image.w * scale_, image.h * scale_, (size_t)3, 3);
        model_->process(image, result);

        save_buff.push_back(result);

        return true;
    }

private:
    RealESRGAN* model_;
    int scale_;
    int current_;
};

class InterpolateProcessor: public Processor {
public:
    InterpolateProcessor(const fs::path& model_dir, int gpu_id, int num_threads, 
                         int multiplier) 
            : begin_(true), idx_(0), current_(0), multiplier_(multiplier) {

        float fraction = 1.0 / multiplier;
        float timestep = 0.0;

        // std::vector<float> timesteps;
        for (int i = 0; i < multiplier - 1; ++i) {
            timestep += fraction;
            timesteps_.push_back(timestep);
        }

        model_ = new RIFE(gpu_id, false, false, false, num_threads, false, true);
        model_->load(model_dir);
    }

    MediaInfo get_info(MediaInfo src_info) {
        MediaInfo dst_info = src_info;
        dst_info.multiplier = multiplier_;
        dst_info.fps *= multiplier_;
        return dst_info;
    }

    bool get_result(TaskQueue& load_buff, TaskQueue& save_buff) {
        ncnn::Mat image_1 = load_buff.at(0);
        ncnn::Mat image_2 = load_buff.at(1);
        if (image_2.empty()) {
            return false;
        }

        ncnn::Mat result(image_1.w, image_1.h, (size_t)3, 3);
        model_->process(image_1, image_2, timesteps_[idx_], result);

        if (begin_) {
            save_buff.push_back(image_1);
            begin_ = !begin_;
        }
        save_buff.push_back(result);
        if (idx_ == timesteps_.size() - 1) {
            load_buff.pop_front();
            save_buff.push_back(image_2);
        } 

        ++idx_;
        if (idx_ == timesteps_.size()) {
            idx_ = 0;
        }

        return true;
    }

private:
    RIFE* model_;
    int multiplier_;
    std::vector<float> timesteps_;
    bool begin_;
    int idx_;
    int current_;
};

TaskQueue load_buff;
TaskQueue save_buff;
int current = 0;

struct LoadThreadParams {
    ImageDecoder* decoder;
};

void* load(void* args) {
    const LoadThreadParams* ltp = (const LoadThreadParams*)args;
    ImageDecoder* decoder = ltp->decoder;

    while (true) {
        ncnn::Mat image = decoder->get_next();
        if (image.empty()) {
            break;
        }
        load_buff.push_back(image);
    }

    return 0;
}

struct ProcThreadParams {
    Processor* processor;
};

void* proc(void* args) {
    const ProcThreadParams* ptp = (const ProcThreadParams*)args;
    Processor* processor = ptp->processor;

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        if (!processor->get_result(load_buff, save_buff)) {
            break;
        }
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Current image: " << current + 1 << ", Duration: " << duration.count() << " ms\n";
        ++current;
    }

    return 0;
}

struct SaveThreadParams {
    ImageEncoder* encoder;
};

void* save(void* args) {
    const SaveThreadParams* stp = (const SaveThreadParams*)args;
    ImageEncoder* encoder = stp->encoder;

    while (true) {
        ncnn::Mat image = save_buff.front();
        if (image.empty()) {
            break;
        }
        save_buff.pop_front();
        encoder->put_next(image);
    }

    return 0;
}

std::vector<float> get_timesteps(int multiplier) {
    float fraction = 1.0 / multiplier;
    float timestep = 0.0;

    std::vector<float> timesteps;
    for (int i = 0; i < multiplier - 1; ++i) {
        timestep += fraction;
        timesteps.push_back(timestep);
    }

    return timesteps;
}

bool check_gpu_id(int gpu_id) {
    if (gpu_id < -1) {
        std::cerr << "[ERROR] Invalid GPU ID specified: '";
        std::cerr << gpu_id << "'. ";
        std::cerr << "Value must be '-1' or greater.\n";
        return false;
    }
    return true;
}

bool check_num_threads(int num_threads) {
    if (num_threads <= 0) {
        std::cerr << "[ERROR] Invalid thread count specified: '";
        std::cerr << num_threads << "'. ";
        std::cerr << "Value must be positive.\n";
        return false;
    }
    return true;
}

bool check_multiplier(int multiplier, const fs::path& model_dir) {
    std::string family = model_dir.parent_path().parent_path().filename();

    if (family == "rife" && multiplier <= 1) {
        std::cerr << "[ERROR] The RIFE model requires '--mul' value to be specified.\n";
        return false;

    } else if (multiplier < 0) {
        std::cerr << "[ERROR] The '--mul' target value must be positive.\n";
        return false;
    }

    return true;
}

bool check_fps(float fps, fs::path src_path, fs::path dst_path) {
    if (fps == 0 && fs::is_directory(src_path) && !fs::is_directory(dst_path)) {
        std::cerr << "[ERROR] Framerate cannot be detected from the input data. ";
        std::cerr << "Please specify '--fps' value.\n";
        return false;

    } else if (fps < 0) {
        std::cerr << "[ERROR] The '--fps' target value must be positive.\n";
        return false;
    }

    return true;
}

int count_dir_files(const fs::path& path) {
    int count = 0;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (fs::is_regular_file(entry)) {
            ++count;
        }
    }
    return count;
}

bool check_src_path(const fs::path& path) {
    if (!fs::exists(path)) {
        std::cerr << "[ERROR] The specified input file or directory does not exists: '";
        std::cerr << path << "'.\n";
        return false;

    } else if (fs::exists(path) && fs::is_directory(path) && 
               count_dir_files(path) == 0) {
        std::cerr << "[ERROR] The specified input directory is empty: '";
        std::cerr << path << "'.\n";
        return false;
    }

    return true;
}

bool check_model_dir(const fs::path& path) {
    if (!fs::exists(path)) {
        std::cerr << "[ERROR] The specified model directory does not exists: ";
        std::cerr << path << ".\n";
        return false;
    }

    std::string family = path.parent_path().parent_path().filename();

    if (family.empty()) {
        std::cerr << "[ERROR] The specified model directory is invalid: '";
        std::cerr << path << "'. ";
        std::cerr << "Ensure that your path ends with: ";
        std::cerr << "'/<model_family>/<model_dir>/'.\n";
        return false;
    }

    if (family != "rife" && family != "realesr") {
        std::cerr << "[ERROR] The detected model family is invalid: '";
        std::cerr << family << "'.\n";
        return false;
    }

    return true;
}

void setup_parser(argparse::ArgumentParser& parser) {
    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    parser.add_argument("-i", "--input"  ).required();
    parser.add_argument("-o", "--output" ).required();
    parser.add_argument("-m", "--model"  ).required();

    parser.add_argument("-t", "--threads").default_value(1).scan<'d', int>();
    parser.add_argument("-g", "--gpu"    ).default_value(0).scan<'d', int>();
    parser.add_argument(      "--mul"    ).default_value(0).scan<'d', int>();
    parser.add_argument(      "--fps"    ).default_value(0).scan<'d', int>();
}

bool parse_args(argparse::ArgumentParser& parser, int argc, char* argv[]) {
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << '\n';
        return false;
    }
    return true;
}

bool validate_args(argparse::ArgumentParser& parser) {
    if (!check_src_path   (parser.get     ("-i")) ||
        !check_model_dir  (parser.get     ("-m")) ||
        !check_gpu_id     (parser.get<int>("-g")) ||
        !check_num_threads(parser.get<int>("-t"))
    ) {
        return false;
    }

    if (!check_fps       (parser.get<int>("--fps"), 
                          parser.get("-i"), parser.get("-o")) ||

        !check_multiplier(parser.get<int>("--mul"),
                          parser.get("-m"))
    ) {
        return false;
    }

    return true;
}

ImageDecoder* decoder_factory(const fs::path& path, int fps) {
    if (fs::is_directory(path)) {
        return new DirImageDecoder(path, fps);
    }
    return new VideoImageDecoder(path);
}

ImageEncoder* encoder_factory(const fs::path& path, MediaInfo info) {
    if (fs::is_directory(path)) {
        return new DirImageEncoder(path);
    }
    return new VideoImageEncoder(path, info);
}

Processor* processor_factory(const fs::path& path, int gpu_id, int num_threads, 
                             int multiplier) {
    std::string family = path.parent_path().parent_path().filename();
    if (family == "rife") {
        return new InterpolateProcessor(path, gpu_id, num_threads, multiplier);
    }
    return new UpscaleProcessor(path, gpu_id);
}

void start_job(ImageDecoder* decoder, ImageEncoder* encoder, 
               Processor* processor, int gpu_id, int num_threads) {

    // load
    LoadThreadParams ltp{decoder};
    ncnn::Thread load_thread(load, (void*)&ltp);

    // proc
    ProcThreadParams ptp{processor};
    std::vector<ncnn::Thread> proc_thread;

    for (int i = 0; i < num_threads; ++i) {
        proc_thread.push_back(ncnn::Thread(proc, (void*)&ptp));
        if (gpu_id == -1) {
            break;
        }
    }

    // save
    SaveThreadParams stp{encoder};
    ncnn::Thread save_thread(save, (void*)&stp);

    // load
    load_thread.join();

    // proc
    load_buff.push_back(ncnn::Mat());
    for (int i = 0; i < num_threads; ++i) {
        proc_thread[i].join();
    }

    // save
    save_buff.push_back(ncnn::Mat());
    save_thread.join();
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser parser("vision-enhancer-ai");
    setup_parser(parser);

    if (!parse_args(parser, argc, argv)) {
        return -1;
    }
    if (!validate_args(parser)) {
        return -1;
    }

    ImageDecoder* decoder = decoder_factory(parser.get("-i"),
                                            parser.get<int>("--fps"));

    Processor* processor = processor_factory(parser.get("-m"), 
                                             parser.get<int>("-g"), 
                                             parser.get<int>("-t"), 
                                             parser.get<int>("--mul"));

    MediaInfo src_info = decoder->get_info();
    MediaInfo dst_info = processor->get_info(src_info);

    ImageEncoder* encoder = encoder_factory(parser.get("-o"), dst_info);

    start_job(decoder, encoder, processor, parser.get<int>("-g"), 
              parser.get<int>("-t"));

    return 0;
}
