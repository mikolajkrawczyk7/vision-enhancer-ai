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
        info.width = capture_.get(cv::CAP_PROP_FRAME_WIDTH); 
        info.height = capture_.get(cv::CAP_PROP_FRAME_HEIGHT); 
        info.fps = capture_.get(cv::CAP_PROP_FPS);
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
    DirImageDecoder(const fs::path& dir_path)
            : dir_path_(dir_path), current_(0) {}

    MediaInfo get_info() {
        MediaInfo info;
        for (const auto& entry: fs::directory_iterator(dir_path_)) {
            std::cout << "First file: " << entry.path() << std::endl;
            cv::Mat image = cv::imread(entry.path());
            info.width = image.cols;
            info.height = image.rows;
            info.fps = 0;
            break;
        }
        return info;
    }

    ncnn::Mat get_next() {
        std::ostringstream oss;
        oss << "frame_" << std::setw(8) << std::setfill('0') << current_ + 1 << ".png";
        cv::Mat cv_image = cv::imread(dir_path_ / oss.str());
        // cv_image = cv::imread(dir_path_ / oss.str());

        unsigned char* raw_data = (unsigned char*)malloc(cv_image.cols * cv_image.rows * cv_image.channels());
        std::move(cv_image.data, cv_image.data + cv_image.cols * cv_image.rows * cv_image.channels(), raw_data);
        ncnn::Mat image(cv_image.cols, cv_image.rows, (void*)raw_data, (size_t)3, 3);

        return image;
    }

private:
    fs::path dir_path_;
    int current_;
};

class ImageEncoder {
public:
    virtual void put_next(ncnn::Mat image) = 0;
};

class VideoImageEncoder: public ImageEncoder {
public:
    // VideoImageEncoder(const fs::path& path, int width, int height, float fps) {
    VideoImageEncoder(const fs::path& path, MediaInfo info) {
        // int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        // cv::Size size = cv::Size(width, height);
        // writer_ = cv::VideoWriter(path, fourcc, fps, size);
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
    InterpolateProcessor(const fs::path& model_dir, int gpu_id, std::vector<float> timesteps) 
            : timesteps_(timesteps), begin_(true), idx_(0), current_(0) {
        model_ = new RIFE(gpu_id, false, false, false, 1, false, true);
        model_->load(model_dir);
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

void decoder_factory() {
}

bool check_if_exists(const std::vector<fs::path>& paths) {
    for (int i = 0; i < paths.size(); ++i) {
        if (!fs::exists(paths[i])) {
            std::cerr << "[ERROR] Path does not exists: " << paths[i] << '\n';
            return false;
        }
    }
    return true;
}

bool check_gpu_id(int gpu_id) {
    if (gpu_id <= -1) {
        std::cerr << "[ERROR] GPU ID must be '-1' or greater\n";
        return false;
    }
    return true;
}

bool check_num_threads(int num_threads) {
    if (num_threads <= 0) {
        std::cerr << "[ERROR] Thread count must be positive\n";
        return false;
    }
    return true;
}

bool check_multiplier(int multiplier) {
    if (multiplier <= 1) {
        std::cerr << "[ERROR] Multiplier value must be greater than '0'\n";
        return false;
    }
    return true;
}

bool check_fps(float fps, fs::path src_path, fs::path dst_path) {
    if (fps == 0 && fs::is_directory(src_path) && !fs::is_directory(dst_path)) {
        std::cerr << "[ERROR] No '--fps' argument provided\n";
        return false;
    } else if (fps < 0) {
        std::cerr << "[ERROR] Target FPS value can't be negative\n";
        return false;
    }
    return true;
}

bool check_model_family(std::string model_family) {
    if (model_family != "rife" && model_family != "realesr") {
        std::cerr << "[ERROR] Detected model family is invalid\n";
        return false;
    }
    return true;
}

bool is_file_video(fs::path path) {
    cv::VideoCapture capture(path);
    if (!capture.isOpened()) {
        return false;
    }
    capture.release();
    return true;
}

void parse_args(int argc, char* argv[], std::string program_name) {
    argparse::ArgumentParser parser(program_name);

    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);
    parser.add_argument("-i", "--input").required();
    parser.add_argument("-o", "--output").required();
    parser.add_argument("-m", "--model").required();
    parser.add_argument("-t", "--threads").default_value(1).scan<'d', int>();
    parser.add_argument("-g", "--gpu").default_value(0).scan<'d', int>();
    parser.add_argument("--mul").default_value(1).scan<'d', int>();
    parser.add_argument("--fps").default_value(0).scan<'d', int>();

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << '\n';
        std::cerr << parser;
        std::exit(1);
    }
}

int main(int argc, char* argv[]) {
    // parse_args(argc, argv, "vision-enhancer-ai");

    argparse::ArgumentParser program("vision-enhancer-ai");

    program.add_argument("-v", "--verbose").default_value(false).implicit_value(true);
    program.add_argument("-i", "--input").required();
    program.add_argument("-o", "--output").required();
    program.add_argument("-m", "--model").required();
    program.add_argument("-t", "--threads").default_value(1).scan<'d', int>();
    program.add_argument("-g", "--gpu").default_value(0).scan<'d', int>();
    program.add_argument("--mul").default_value(1).scan<'d', int>();
    program.add_argument("--fps").default_value(0).scan<'d', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << '\n';
        std::cerr << program;
        std::exit(1);
    }

    fs::path src_path = program.get("-i");
    fs::path dst_path = program.get("-o");
    fs::path model_dir = program.get("-m");

    int gpu_id = program.get<int>("-g");
    int num_threads = program.get<int>("-t");
    int multiplier = program.get<int>("--mul");
    int fps = program.get<int>("--fps"); //float
    bool verbose = program.get<bool>("-v");

    // check model family and decide what to do

    // error handling layer 1 - parameters
    // put this in namespace (???)

    // first pass
    if (!check_if_exists({src_path, dst_path, model_dir}) ||
        !check_gpu_id(gpu_id) ||
        !check_num_threads(num_threads) || 
        !check_multiplier(multiplier))
    {
        return -1;
    }

    std::string model_family = model_dir.parent_path().parent_path().filename();

    // second pass
    if (!check_fps(fps, src_path, dst_path) ||
        !check_model_family(model_family))
    {
        return -1;
    }

    // check if file is image or vid
    if (model_family == "rife" && multiplier == 1) {
        multiplier = 2;
    }
    std::vector<float> timesteps = get_timesteps(multiplier);

    // dst empty dir

    // if (fs::is_directory(src_path)) {
    //     
    // }

    // cv::Mat image = cv::imread("video.mp4");
    // if (image.empty()) {
    //     std::cout << "error\n";
    // }
    // if (!capture.isOpened()) {
    //     std::cout << "error\n";
    // }

    // return 0;

    // error handling layer 2 - combos

    std::cout << "Source path: " << src_path << '\n';
    std::cout << "Destination path: " << dst_path << '\n';
    std::cout << "Model directory: " << model_dir << '\n';
    std::cout << "GPU ID: " << gpu_id << '\n';
    std::cout << "Processing threads: " << num_threads << '\n';
    std::cout << "Multiplier: " << multiplier << "x\n";
    std::cout << "Model family: " << model_family << '\n';

    ImageDecoder* decoder = nullptr;
    ImageEncoder* encoder = nullptr;
    Processor* processor = nullptr;

    if (fs::is_directory(src_path)) {
        decoder = new DirImageDecoder(src_path);
    } else {
        decoder = new VideoImageDecoder(src_path);
    }

    if (fs::is_directory(dst_path)) {
        encoder = new DirImageEncoder(dst_path);
    } else {
        MediaInfo info = decoder->get_info();
        if (info.fps == 0) {
            info.fps = fps;
        } else if (model_family == "rife") {
            info.fps *= multiplier;
        }
        std::cout << "Resolution: " << info.width << 'x' << info.height << '\n';
        std::cout << "Target FPS: " << info.fps << '\n';
        encoder = new VideoImageEncoder(dst_path, info);
    }

    if (model_family == "rife") {
        processor = new InterpolateProcessor(model_dir, gpu_id, timesteps);
    } else if (model_family == "realesr") {
        processor = new UpscaleProcessor(model_dir, gpu_id);
    }

    // main routine
    LoadThreadParams ltp{decoder};
    ncnn::Thread load_thread(load, (void*)&ltp);

    ProcThreadParams ptp{processor};

    std::vector<ncnn::Thread> proc_thread;
    for (int i = 0; i < num_threads; ++i) {
        proc_thread.push_back(ncnn::Thread(proc, (void*)&ptp));
    }

    SaveThreadParams stp{encoder};
    ncnn::Thread save_thread(save, (void*)&stp);

    load_thread.join();

    load_buff.push_back(ncnn::Mat());
    for (int i = 0; i < num_threads; ++i) {
        proc_thread[i].join();
    }

    save_buff.push_back(ncnn::Mat());
    save_thread.join();

    return 0;
}
