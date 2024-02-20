#include <exception>
#include <iostream>
#include <chrono>
#include <functional>
#include <algorithm>
#include <deque>
#include <memory>
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

template <typename T>
class ThreadSafeBuffer {
public:
    ThreadSafeBuffer(size_t size = 8) : size_(size) {}

    void wait_custom(std::function<bool()> condition) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, condition);
    }

    void wait_full() {
        wait_custom([this] { return data_.size() < size_; });
    }

    void wait_empty() {
        wait_custom([this] { return !data_.empty(); });
    }

    void push_back(const T& item) {
        wait_full();
        data_.push_back(item);
        condition_.notify_one();
    }

    void pop_front() {
        wait_empty();
        data_.pop_front();
        condition_.notify_one();
    }

    T& front() {
        wait_empty();
        return data_.front();
    }

    T& back() {
        wait_empty();
        return data_.back();
    }

    T& at(size_t index) {
        wait_custom([this, index](){ return data_.size() > index; });
        return data_.at(index);
    }

    bool empty() {
        return data_.empty();
    }

    void clear() {
        data_.clear();
    }

private:
    size_t size_;
    std::deque<T> data_;
    std::mutex mutex_;
    std::condition_variable condition_;
};

struct MediaInfo {
    int width;
    int height;
    int fps;
    int multiplier;
};

class ImageDecoder {
public:
    virtual cv::Mat   get_next() = 0;
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

    cv::Mat get_next() {
        cv::Mat image;
        capture_.read(image);
        return image;
    }

private:
    cv::VideoCapture capture_;
};

std::string get_filename(const std::string& prefix, int number, 
                         const std::string& extension) {
    std::ostringstream oss;

    oss << prefix;
    oss << std::setw(8) << std::setfill('0') << number;
    oss << extension;

    return oss.str();
}

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

    cv::Mat get_next() {
        ++current_;
        std::string filename = get_filename("frame_", current_, ".png");
        cv::Mat image = cv::imread(dir_path_ / filename);
        return image;
    }

private:
    fs::path dir_path_;
    int current_;
    int fps_;
};

class ImageEncoder {
public:
    virtual void put_next(const cv::Mat& image) = 0;
};

class VideoImageEncoder: public ImageEncoder {
public:
    VideoImageEncoder(const fs::path& path, MediaInfo info) {
        std::string extension = path.filename().extension().string();

        std::map<std::string, std::string> codec;
        codec[".mkv" ] = "FFV1";
        codec[".webm"] = "VP09";
        codec[".avi" ] = "MJPG";
        codec[".mp4" ] = "H264";

        char c1 = codec[extension][0];
        char c2 = codec[extension][1];
        char c3 = codec[extension][2];
        char c4 = codec[extension][3];

        int fourcc = cv::VideoWriter::fourcc(c1, c2, c3, c4);
        cv::Size size = cv::Size(info.width, info.height);
        writer_ = cv::VideoWriter(path, fourcc, info.fps, size, true);
    }

    void put_next(const cv::Mat& image) {
        writer_.write(image);
    }

private:
    cv::VideoWriter writer_;
};

class DirImageEncoder: public ImageEncoder {
public:
    DirImageEncoder(const fs::path& dir_path)
            : dir_path_(dir_path), current_(0) {}

    void put_next(const cv::Mat& image) {
        ++current_;
        std::string filename = get_filename("", current_, ".png");
        cv::imwrite(dir_path_ / filename, image);
    }

private:
    fs::path dir_path_;
    int current_;
};

class Processor {
public:
    virtual MediaInfo get_info(MediaInfo src_info) = 0;

    virtual bool get_result(ThreadSafeBuffer<cv::Mat>& load_buffer,
                            ThreadSafeBuffer<cv::Mat>& save_buffer) = 0;
};

int get_tilesize(int gpu_id) {
    size_t heap_budget = ncnn::get_gpu_device(gpu_id)->get_heap_budget();

    if (heap_budget > 1900) {
        return 200;
    } 
    if (heap_budget > 550) {
        return 100;
    } 
    if (heap_budget > 190) {
        return 64;
    }

    return 32;
}

class UpscaleProcessor: public Processor {
public:
    UpscaleProcessor(const fs::path& model_dir, int gpu_id) : current_(0) {
        int tilesize = get_tilesize(gpu_id);

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

        if (dir_name.find("x2") != std::string::npos) {
            scale_ = 2;
        } else if (dir_name.find("x3") != std::string::npos) {
            scale_ = 3;
        } else {
            scale_ = 4;
        }

        model_ = new RealESRGAN(gpu_id, false);
        model_->load(param_path, model_path);
        model_->scale = scale_;
        model_->tilesize = tilesize;
        model_->prepadding = 10;
    }

    ~UpscaleProcessor() {
        delete model_;
    }

    MediaInfo get_info(MediaInfo src_info) {
        MediaInfo dst_info = src_info;
        dst_info.width  *= scale_;
        dst_info.height *= scale_;
        return dst_info;
    }

    bool get_result(ThreadSafeBuffer<cv::Mat>& load_buffer,
                    ThreadSafeBuffer<cv::Mat>& save_buffer) {
        cv::Mat cv_image = load_buffer.front();
        if (cv_image.empty()) {
            return false;
        }
        load_buffer.pop_front();

        ncnn::Mat ncnn_image(cv_image.cols, cv_image.rows, cv_image.data, 
                             3, 3);

        ncnn::Mat ncnn_result(ncnn_image.w * scale_, ncnn_image.h * scale_, 
                              3, 3);

        model_->process(ncnn_image, ncnn_result);

        save_buffer.push_back(cv::Mat(ncnn_result.h, ncnn_result.w, CV_8UC3, 
                                      ncnn_result.data).clone());

        return true;
    }

private:
    RealESRGAN* model_;
    int scale_;
    int current_;
};

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

class InterpolateProcessor: public Processor {
public:
    InterpolateProcessor(const fs::path& model_dir, int gpu_id, 
                         int num_threads, int multiplier) 
            : begin_(true), index_(0), current_(0), multiplier_(multiplier) {

        timesteps_ = get_timesteps(multiplier);

        model_ = new RIFE(gpu_id, false, false, false, num_threads, false, 
                          true);
        model_->load(model_dir);
    }

    ~InterpolateProcessor() {
        delete model_;
    }

    MediaInfo get_info(MediaInfo src_info) {
        MediaInfo dst_info = src_info;
        dst_info.multiplier  = multiplier_;
        dst_info.fps        *= multiplier_;
        return dst_info;
    }

    bool get_result(ThreadSafeBuffer<cv::Mat>& load_buffer,
                    ThreadSafeBuffer<cv::Mat>& save_buffer) {
        if (load_buffer.at(1).empty()) {
            return false;
        }

        std::array<cv::Mat, 2> cv_images = { load_buffer.at(0), 
                                             load_buffer.at(1) };
        std::array<ncnn::Mat, 2> ncnn_images;

        for (int i = 0; i < 2; ++i) {
            ncnn_images[i] = ncnn::Mat(cv_images[i].cols, cv_images[i].rows, 
                                       cv_images[i].data, 3, 3);
        }

        ncnn::Mat ncnn_result(ncnn_images[0].w, ncnn_images[0].h, 3, 3);

        model_->process(ncnn_images[0], ncnn_images[1], timesteps_[index_], 
                        ncnn_result);

        if (begin_) {
            save_buffer.push_back(cv_images[0]);
            begin_ = !begin_;
        }

        save_buffer.push_back(cv::Mat(ncnn_result.h, ncnn_result.w, CV_8UC3, 
                                      ncnn_result.data).clone());

        if (index_ == timesteps_.size() - 1) {
            load_buffer.pop_front();
            save_buffer.push_back(cv_images[1]);
        } 

        index_ = (index_ + 1) % timesteps_.size();

        return true;
    }

private:
    RIFE* model_;
    bool begin_;
    int current_;
    int index_;
    int multiplier_;
    std::vector<float> timesteps_;
};

ThreadSafeBuffer<cv::Mat> load_buffer;
ThreadSafeBuffer<cv::Mat> save_buffer;
int current = 0;

void load(std::shared_ptr<ImageDecoder> decoder) {
    while (true) {
        cv::Mat image = decoder->get_next();
        if (image.empty()) {
            break;
        }
        load_buffer.push_back(image);
    }
}

void proc(std::shared_ptr<Processor> processor, bool verbose) {
    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        if (!processor->get_result(load_buffer, save_buffer)) {
            break;
        }
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>
            (stop - start);

        if (verbose) {
            std::cout << "Current image: " << current + 1 << ", ";
            std::cout << "Duration: " << duration.count() << " ms\n";
        }

        ++current;
    }
}

void save(std::shared_ptr<ImageEncoder> encoder) {
    while (true) {
        cv::Mat image = save_buffer.front();
        if (image.empty()) {
            break;
        }
        save_buffer.pop_front();
        encoder->put_next(image);
    }
}

bool check_gpu_id(int gpu_id) {
    if (gpu_id < -1) {
        std::cerr << "[ERROR] Invalid GPU ID specified: '" << gpu_id << "'. "
                  << "Value must be '-1' or greater.\n";
        return false;
    }
    return true;
}

bool check_num_threads(int num_threads) {
    if (num_threads <= 0) {
        std::cerr << "[ERROR] Invalid thread count specified: '"
                  << num_threads << "'. " << "Value must be positive.\n";
        return false;
    }
    return true;
}

bool check_multiplier(int multiplier, const fs::path& model_dir) {
    std::string family = model_dir.parent_path().parent_path().filename();

    if (family == "rife" && multiplier <= 1) {
        std::cerr << 
            "[ERROR] The RIFE model requires '--mul' value to be specified.\n";
        return false;

    } else if (multiplier < 0) {
        std::cerr << "[ERROR] The '--mul' target value must be positive.\n";
        return false;
    }

    return true;
}

bool check_fps(float fps, fs::path src_path, fs::path dst_path) {
    if (fps == 0 && fs::is_directory(src_path) && 
        !fs::is_directory(dst_path)) {

        std::cerr << "[ERROR] Framerate cannot be detected " 
                  << "from the input data. "
                  << "Please specify '--fps' value.\n";
        return false;

    } else if (fps < 0) {
        std::cerr << "[ERROR] The '--fps' target value must be positive.\n";
        return false;
    }

    return true;
}

int count_dir_files(const fs::path& path) {
    int count = 0;
    for (const auto& entry: fs::directory_iterator(path)) {
        if (fs::is_regular_file(entry)) {
            ++count;
        }
    }
    return count;
}

bool check_src_path(const fs::path& path) {
    if (!fs::exists(path)) {
        std::cerr << "[ERROR] The specified input file or directory " 
                  << "does not exists: '" << path << "'.\n";
        return false;

    } else if (fs::exists(path) && fs::is_directory(path) && 
               count_dir_files(path) == 0) {
        std::cerr << "[ERROR] The specified input directory is empty: '"
                  << path << "'.\n";
        return false;
    }

    return true;
}

bool check_model(const fs::path& path) {
    if (!fs::exists(path)) {
        std::cerr << "[ERROR] The specified model directory does not exists: "
                  << path << ".\n";
        return false;
    }

    std::string family = path.parent_path().parent_path().filename();

    if (family.empty()) {
        std::cerr << "[ERROR] The specified model directory is invalid: '"
                  << path << "'. Ensure that your path ends with: "
                  << "'/<model_family>/<model_dir>/'.\n";
        return false;
    }

    if (family != "rife" && family != "realesr") {
        std::cerr << "[ERROR] The detected model family is invalid: '"
                  << family << "'.\n";
        return false;
    }

    return true;
}

bool check_model_dirs(std::vector<std::string> model_dirs) {
    for (int i = 0; i < model_dirs.size(); ++i) {
        if (!check_model(model_dirs[i])) {
            return false;
        }
    }
    return true;
}

void setup_parser(argparse::ArgumentParser& parser) {
    parser.add_argument("-v", "--verbose")
        .default_value(false).implicit_value(true);

    parser.add_argument("-i", "--input"  ).required();
    parser.add_argument("-o", "--output" ).required();
    parser.add_argument("-m", "--model"  ).required().nargs(1, 2);
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
    if (!check_src_path   (parser.get                          ("-i")) ||
        !check_model_dirs (parser.get<std::vector<std::string>>("-m")) ||
        !check_gpu_id     (parser.get<int>                     ("-g")) ||
        !check_num_threads(parser.get<int>                     ("-t"))
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

std::shared_ptr<ImageDecoder> decoder_factory(const fs::path& path, int fps) {
    if (fs::is_directory(path)) {
        return std::make_shared<DirImageDecoder>(path, fps);
    }
    return std::make_shared<VideoImageDecoder>(path);
}

bool check_extension(const std::vector<std::string> extensions, 
                     const fs::path& path) {
    std::string extension = path.filename().extension().string();
    auto iter = std::find(extensions.begin(), extensions.end(), extension);

    if (iter == extensions.end()) {
        return false;
    }

    return true;
}

bool is_video_path(const fs::path& path) {
    return check_extension({".mkv", ".webm", ".avi", ".mp4"}, path);
}

bool is_image_path(const fs::path& path) {
    return check_extension({".png", ".jpg", ".webp"}, path);
}

std::shared_ptr<ImageEncoder> encoder_factory(const fs::path& path,
                                              MediaInfo info) {
    if (is_video_path(path)) {
        return std::make_shared<VideoImageEncoder>(path, info);
    }
    return std::make_shared<DirImageEncoder>(path);
}

std::shared_ptr<Processor> processor_factory(const fs::path& path, int gpu_id,
                                             int num_threads, int multiplier) {
    std::string family = path.parent_path().parent_path().filename();
    if (family == "rife") {
        return std::make_shared<InterpolateProcessor>(path, gpu_id, 
                                                      num_threads, multiplier);
    }
    return std::make_shared<UpscaleProcessor>(path, gpu_id);
}

struct Job {
    std::shared_ptr<ImageDecoder> decoder;
    std::shared_ptr<ImageEncoder> encoder;
    std::shared_ptr<Processor> processor;
};

void start_job(Job job, int gpu_id, int num_threads, bool verbose) {
    // load
    std::thread load_thread(load, job.decoder);

    // proc
    std::vector<std::thread> proc_thread;
    for (int i = 0; i < num_threads; ++i) {
        proc_thread.push_back(std::thread(proc, job.processor, verbose));
        if (gpu_id == -1) {
            break;
        }
    }

    // save
    std::thread save_thread(save, job.encoder);

    // load
    load_thread.join();

    // proc
    load_buffer.push_back(cv::Mat());
    for (int i = 0; i < num_threads; ++i) {
        proc_thread[i].join();
    }

    // save
    save_buffer.push_back(cv::Mat());
    save_thread.join();
}

Job setup_job(argparse::ArgumentParser& parser, const fs::path& src_path, 
              const fs::path& dst_path, const fs::path& model_path) {
    auto decoder = decoder_factory(src_path, parser.get<int>("--fps"));

    auto processor = 
        processor_factory(model_path, parser.get<int>("-g"),
                          parser.get<int>("-t"), parser.get<int>("--mul"));

    MediaInfo src_info = decoder->get_info();
    MediaInfo dst_info = processor->get_info(src_info);

    auto encoder = encoder_factory(dst_path, dst_info);

    return Job{decoder, encoder, processor};
}

void process_job(argparse::ArgumentParser& parser, const fs::path& src_path, 
                 const fs::path& dst_path, const fs::path& model_path) {
    Job job = setup_job(parser, src_path, dst_path, model_path);

    start_job(job, parser.get<int>("-g"), parser.get<int>("-t"),
              parser.get<bool>("-v"));

    current = 0;
    load_buffer.clear();
    save_buffer.clear();
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser parser("vision-enhancer-ai");
    setup_parser(parser);

    // add options for example to list models
    if (!parse_args(parser, argc, argv) || !validate_args(parser)) {
        return -1;
    }

    // implement single image procesing
    fs::path dst_path = parser.get("-o");

    if (!is_video_path(dst_path) && !fs::exists(dst_path)) {
        fs::create_directories(dst_path);
    }

    auto model_paths = parser.get<std::vector<std::string>>("-m");

    // refactor job related things
    if (model_paths.size() == 1) {
        process_job(parser, parser.get("-i"), parser.get("-o"),
                    parser.get("-m"));
    } else {
        process_job(parser, parser.get("-i"), "temp.avi", model_paths[0]);
        process_job(parser, "temp.avi", parser.get("-o"), model_paths[1]);
    }

    return 0;
}
