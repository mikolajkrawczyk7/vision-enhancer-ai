// rife implemented with ncnn library

#include <iostream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

int get_file_count(const std::string& path)
{
    int count = 0;
    for (const auto& entry : fs::directory_iterator(path))
    {
        if (fs::is_regular_file(entry))
        {
            count++;
        }
    }
    return count;
}

#include <stdio.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <clocale>

#if _WIN32
// image decoder and encoder with wic
#include "wic_image.h"
#else // _WIN32
// image decoder and encoder with stb
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_STDIO
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // _WIN32
#include "webp_image.h"

#if _WIN32
#include <wchar.h>
static wchar_t* optarg = NULL;
static int optind = 1;
static wchar_t getopt(int argc, wchar_t* const argv[], const wchar_t* optstring)
{
    if (optind >= argc || argv[optind][0] != L'-')
        return -1;

    wchar_t opt = argv[optind][1];
    const wchar_t* p = wcschr(optstring, opt);
    if (p == NULL)
        return L'?';

    optarg = NULL;

    if (p[1] == L':')
    {
        optind++;
        if (optind >= argc)
            return L'?';

        optarg = argv[optind];
    }

    optind++;

    return opt;
}

static std::vector<int> parse_optarg_int_array(const wchar_t* optarg)
{
    std::vector<int> array;
    array.push_back(_wtoi(optarg));

    const wchar_t* p = wcschr(optarg, L',');
    while (p)
    {
        p++;
        array.push_back(_wtoi(p));
        p = wcschr(p, L',');
    }

    return array;
}
#else // _WIN32
#include <unistd.h> // getopt()

static std::vector<int> parse_optarg_int_array(const char* optarg)
{
    std::vector<int> array;
    array.push_back(atoi(optarg));

    const char* p = strchr(optarg, ',');
    while (p)
    {
        p++;
        array.push_back(atoi(p));
        p = strchr(p, ',');
    }

    return array;
}
#endif // _WIN32

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"
#include "benchmark.h"

#include "rife.h"
#include "realesrgan.h"

#include "filesystem_utils.h"

static void print_usage()
{
    fprintf(stderr, "Usage: rife-ncnn-vulkan -0 infile -1 infile1 -o outfile [options]...\n");
    fprintf(stderr, "       rife-ncnn-vulkan -i indir -o outdir [options]...\n\n");
    fprintf(stderr, "  -h                   show this help\n");
    fprintf(stderr, "  -v                   verbose output\n");
    fprintf(stderr, "  -0 input0-path       input image0 path (jpg/png/webp)\n");
    fprintf(stderr, "  -1 input1-path       input image1 path (jpg/png/webp)\n");
    fprintf(stderr, "  -i input-path        input image directory (jpg/png/webp)\n");
    fprintf(stderr, "  -o output-path       output image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -n num-frame         target frame count (default=N*2)\n");
    fprintf(stderr, "  -s time-step         time step (0~1, default=0.5)\n");
    fprintf(stderr, "  -m model-path        rife model path (default=rife-v2.3)\n");
    fprintf(stderr, "  -g gpu-id            gpu device to use (-1=cpu, default=auto) can be 0,1,2 for multi-gpu\n");
    fprintf(stderr, "  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n");
    fprintf(stdout, "  -x                   enable spatial tta mode\n");
    fprintf(stdout, "  -z                   enable temporal tta mode\n");
    fprintf(stdout, "  -u                   enable UHD mode\n");
    fprintf(stderr, "  -f pattern-format    output image filename pattern format (%%08d.jpg/png/webp, default=ext/%%08d.png)\n");
}

static int decode_image(const path_t& imagepath, ncnn::Mat& image, int* webp)
{
    *webp = 0;

    unsigned char* pixeldata = 0;
    int w;
    int h;
    int c;

#if _WIN32
    FILE* fp = _wfopen(imagepath.c_str(), L"rb");
#else
    FILE* fp = fopen(imagepath.c_str(), "rb");
#endif
    if (fp)
    {
        // read whole file
        unsigned char* filedata = 0;
        int length = 0;
        {
            fseek(fp, 0, SEEK_END);
            length = ftell(fp);
            rewind(fp);
            filedata = (unsigned char*)malloc(length);
            if (filedata)
            {
                fread(filedata, 1, length, fp);
            }
            fclose(fp);
        }

        if (filedata)
        {
            pixeldata = webp_load(filedata, length, &w, &h, &c);
            if (pixeldata)
            {
                *webp = 1;
            }
            else
            {
                // not webp, try jpg png etc.
#if _WIN32
                pixeldata = wic_decode_image(imagepath.c_str(), &w, &h, &c);
#else // _WIN32
                pixeldata = stbi_load_from_memory(filedata, length, &w, &h, &c, 3);
                c = 3;
#endif // _WIN32
            }

            free(filedata);
        }
    }

    if (!pixeldata)
    {
#if _WIN32
        fwprintf(stderr, L"decode image %ls failed\n", imagepath.c_str());
#else // _WIN32
        fprintf(stderr, "decode image %s failed\n", imagepath.c_str());
#endif // _WIN32

        return -1;
    }

    image = ncnn::Mat(w, h, (void*)pixeldata, (size_t)3, 3);

    return 0;
}

static int encode_image(const path_t& imagepath, const ncnn::Mat& image)
{
    int success = 0;

    cv::Mat cv_image(image.h, image.w, CV_8UC3, image.data);
    cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2RGB);
    success = cv::imwrite(imagepath, cv_image);

    // path_t ext = get_file_extension(imagepath);
// 
    // if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
    // {
        // success = webp_save(imagepath.c_str(), image.w, image.h, image.elempack, (const unsigned char*)image.data);
    // }
    // else if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
    // {
// #if _WIN32
        // success = wic_encode_image(imagepath.c_str(), image.w, image.h, image.elempack, image.data);
// #else
        // success = stbi_write_png(imagepath.c_str(), image.w, image.h, image.elempack, image.data, 0);
// #endif
    // }
    // else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
    // {
// #if _WIN32
        // success = wic_encode_jpeg_image(imagepath.c_str(), image.w, image.h, image.elempack, image.data);
// #else
        // success = stbi_write_jpg(imagepath.c_str(), image.w, image.h, image.elempack, image.data, 100);
// #endif
    // }

    if (!success)
    {
#if _WIN32
        fwprintf(stderr, L"encode image %ls failed\n", imagepath.c_str());
#else
        fprintf(stderr, "encode image %s failed\n", imagepath.c_str());
#endif
    }

    return success ? 0 : -1;
}

class Task
{
public:
    int id;
    int webp0;
    int webp1;

    path_t in0path;
    path_t in1path;
    path_t outpath;
    float timestep;

    ncnn::Mat in0image;
    ncnn::Mat in1image;
    ncnn::Mat outimage;

    int load_duration;
    int proc_duration;
    int save_duration;
};

class TaskQueue
{
public:
    TaskQueue()
    {
    }

    void put(const Task& v)
    {
        lock.lock();

        while (tasks.size() >= 8) // FIXME hardcode queue length
        {
            condition.wait(lock);
        }

        tasks.push(v);

        lock.unlock();

        condition.signal();
    }

    void get(Task& v)
    {
        lock.lock();

        while (tasks.size() == 0)
        {
            condition.wait(lock);
        }

        v = tasks.front();
        tasks.pop();

        lock.unlock();

        condition.signal();
    }

private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::queue<Task> tasks;
};

path_t get_frame_path(path_t source_dir, path_t prefix, int number)
{
    std::ostringstream oss;
    oss << prefix << std::setw(8) << std::setfill('0') << number << ".png";
    path_t path = source_dir + PATHSTR('/') + oss.str();
    return path;
}

TaskQueue toproc;
TaskQueue tosave;

class LoadThreadParams
{
public:
    int jobs_load;

    // session data
    std::vector<path_t> input0_files;
    std::vector<path_t> input1_files;
    std::vector<path_t> output_files;
    std::vector<float> timesteps;

    path_t input_dir;
    path_t output_dir;

    int frame_count;
    bool realesr;
    int scale;
};

void* load(void* args)
{
    const LoadThreadParams* ltp = (const LoadThreadParams*)args;
    // const int count = ltp->output_files.size();
    const int count = ltp->frame_count - 1;
    const path_t input_dir = ltp->input_dir;
    const path_t output_dir = ltp->output_dir;
    const bool realesr = ltp->realesr;
    const int scale = ltp->scale;

    // const path_t& image0path = ltp->input0_files[0];
    // const path_t& image1path = ltp->input1_files[0];
    // const path_t& outputpath = ltp->output_files[0];

    if (ltp->output_files.size() > 0)
    {
        Task v;
        v.id = -1;
        v.in0path = ltp->input0_files[0];
        v.in1path = ltp->input1_files[0];
        v.outpath = ltp->output_files[0];
        v.timestep = 0.5;

        auto start = std::chrono::high_resolution_clock::now();

        int ret0 = decode_image(v.in0path, v.in0image, &v.webp0);
        // int ret1 = decode_image(v.in1path, v.in1image, &v.webp1);

        if (realesr)
        {
            v.outimage = ncnn::Mat(v.in0image.w * scale, v.in0image.h * scale, (size_t)3, 3);
        }
        else
        {
            int ret1 = decode_image(v.in1path, v.in1image, &v.webp1);
            v.outimage = ncnn::Mat(v.in0image.w, v.in0image.h, (size_t)3, 3);
        }

        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        v.load_duration = duration.count();

        toproc.put(v);
    }
    // todo
    else
    {
        #pragma omp parallel for schedule(static,1) num_threads(ltp->jobs_load)
        for (int i=0; i<count; i++)
        {
            Task v;
            v.id = i;
            v.in0path = get_frame_path(input_dir, "frame_", i + 1);
            v.in1path = get_frame_path(input_dir, "frame_", i + 2);
            if (realesr)
            {
                v.outpath = get_frame_path(output_dir, "", i + 1);
            }
            else 
            {
                v.outpath = get_frame_path(output_dir, "", i * 2 + 2);
            }
            v.timestep = 0.5;

            auto start = std::chrono::high_resolution_clock::now();

            int ret0 = decode_image(
                // get_frame_path(input_dir, "frame_", i + 1),
                v.in0path, v.in0image, &v.webp0);

            // int ret1 = decode_image(
            //     // get_frame_path(input_dir, "frame_", i + 2),
            //     v.in1path, v.in1image, &v.webp1);

            if (realesr)
            {
                v.outimage = ncnn::Mat(v.in0image.w * scale, v.in0image.h * scale, (size_t)3, 3);
            }
            else
            {
                int ret1 = decode_image(
                    // get_frame_path(input_dir, "frame_", i + 2),
                    v.in1path, v.in1image, &v.webp1);

                v.outimage = ncnn::Mat(v.in0image.w, v.in0image.h, (size_t)3, 3);
            }

            auto stop = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            v.load_duration = duration.count();

            toproc.put(v);

            // if (ret0 != 0 || ret1 != 1)
            // {
            //     v.outimage = ncnn::Mat(v.in0image.w, v.in0image.h, (size_t)3, 3);
            //     toproc.put(v);
            // }

            // const path_t& image0path = ltp->input0_files[i];
            // const path_t& image1path = ltp->input1_files[i];

            // Task v;
            // v.id = i;
            // v.in0path = image0path;
            // v.in1path = image1path;
            // v.outpath = ltp->output_files[i];
            // v.timestep = ltp->timesteps[i];

            // int ret0 = decode_image(image0path, v.in0image, &v.webp0);
            // int ret1 = decode_image(image1path, v.in1image, &v.webp1);

            // if (ret0 != 0 || ret1 != 1)
            // {
            //     v.outimage = ncnn::Mat(v.in0image.w, v.in0image.h, (size_t)3, 3);
            //     toproc.put(v);
            // }
        }
    }

    return 0;
}

class ProcThreadParams
{
public:
    const RIFE* rife;
    const RealESRGAN* realesrgan;
};

void* proc(void* args)
{
    const ProcThreadParams* ptp = (const ProcThreadParams*)args;
    const RIFE* rife = ptp->rife;
    const RealESRGAN* realesrgan = ptp->realesrgan;

    for (;;)
    {
        Task v;

        toproc.get(v);

        if (v.id == -233)
            break;

        auto start = std::chrono::high_resolution_clock::now();

        if (realesrgan)
        {
            realesrgan->process(v.in0image, v.outimage);
        }
        else 
        {
            rife->process(v.in0image, v.in1image, v.timestep, v.outimage);
        }

        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        v.proc_duration = duration.count();

        tosave.put(v);
    }

    return 0;
}

class SaveThreadParams
{
public:
    int verbose;
    path_t input_dir;
    path_t output_dir;
    bool realesr;
};

void* save(void* args)
{
    const SaveThreadParams* stp = (const SaveThreadParams*)args;
    const int verbose = stp->verbose;
    const path_t input_dir = stp->input_dir;
    const path_t output_dir = stp->output_dir;
    const bool realesr = stp->realesr;

    for (;;)
    {
        Task v;

        tosave.get(v);

        if (v.id == -233)
            break;

        auto start = std::chrono::high_resolution_clock::now();

        // int ret = encode_image(
        //     // get_frame_path(output_dir, "", v.id * 2 + 2),
        //     v.outpath, v.outimage);

        // cv::Mat cv_img(v.outimage.h, v.outimage.w, CV_8UC3, v.outimage.data);
        // cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB);
        // cv::imwrite(v.outpath, cv_img);

        // encode_image(v.outimage, v.outpath);
        encode_image(v.outpath, v.outimage);

        if (v.id != -1 && !realesr)
        {
            if (v.id == 0)
            {
                // int ret0 = encode_image(
                //     get_frame_path(output_dir, "", v.id * 2 + 1),
                //     v.in0image);
                encode_image(get_frame_path(output_dir, "", v.id * 2 + 1),
                             v.in0image);
            }
            // int ret1 = encode_image(
            //     get_frame_path(output_dir, "", v.id * 2 + 3),
            //     v.in1image);
            encode_image(get_frame_path(output_dir, "", v.id * 2 + 3),
                         v.in0image);
        }

        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        v.save_duration = duration.count();
        
        // free input pixel data
        // {
            // unsigned char* pixeldata = (unsigned char*)v.in0image.data;
            // if (v.webp0 == 1)
            // {
                // free(pixeldata);
            // }
            // else
            // {
// #if _WIN32
                // free(pixeldata);
// #else
                // stbi_image_free(pixeldata);
// #endif
            // }
        // }
        // {
            // unsigned char* pixeldata = (unsigned char*)v.in1image.data;
            // if (v.webp1 == 1)
            // {
                // free(pixeldata);
            // }
            // else
            // {
// #if _WIN32
                // free(pixeldata);
// #else
                // stbi_image_free(pixeldata);
// #endif
            // }
        // }

        // if (ret == 0)
        // {
            if (verbose && v.proc_duration > 0)
            {
// #if _WIN32
                // fwprintf(stderr, L"%ls %ls %f -> %ls done\n", v.in0path.c_str(), v.in1path.c_str(), v.timestep, v.outpath.c_str());
// #else
                if (realesr)
                {
                    fprintf(stderr, "[Load: %i ms, Proc: %i ms, Save: %i ms] %s -> %s done\n", v.load_duration, v.proc_duration, v.save_duration, v.in0path.c_str(), v.outpath.c_str());
                }
                else
                {
                    fprintf(stderr, "[Load: %i ms, Proc: %i ms, Save: %i ms] %s %s %f -> %s done\n", v.load_duration, v.proc_duration, v.save_duration, v.in0path.c_str(), v.in1path.c_str(), v.timestep, v.outpath.c_str());
                }
// #endif
            }
        // }
    }

    return 0;
}


#if _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    path_t input0path;
    path_t input1path;
    path_t inputpath;
    path_t outputpath;
    int numframe = 0;
    float timestep = 0.5f;
    path_t model = PATHSTR("rife-v2.3");
    std::vector<int> gpuid;
    int jobs_load = 1;
    std::vector<int> jobs_proc;
    int jobs_save = 2;
    int verbose = 0;
    int tta_mode = 0;
    int tta_temporal_mode = 0;
    int uhd_mode = 0;
    path_t pattern_format = PATHSTR("%08d.png");
    int scale = 4;

#if _WIN32
    setlocale(LC_ALL, "");
    wchar_t opt;
    while ((opt = getopt(argc, argv, L"0:1:i:o:n:s:m:g:j:f:vxzuh")) != (wchar_t)-1)
    {
        switch (opt)
        {
        case L'0':
            input0path = optarg;
            break;
        case L'1':
            input1path = optarg;
            break;
        case L'i':
            inputpath = optarg;
            break;
        case L'o':
            outputpath = optarg;
            break;
        case L'n':
            numframe = _wtoi(optarg);
            break;
        case L's':
            timestep = _wtof(optarg);
            break;
        case L'm':
            model = optarg;
            break;
        case L'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case L'j':
            swscanf(optarg, L"%d:%*[^:]:%d", &jobs_load, &jobs_save);
            jobs_proc = parse_optarg_int_array(wcschr(optarg, L':') + 1);
            break;
        case L'f':
            pattern_format = optarg;
            break;
        case L'v':
            verbose = 1;
            break;
        case L'x':
            tta_mode = 1;
            break;
        case L'z':
            tta_temporal_mode = 1;
            break;
        case L'u':
            uhd_mode = 1;
            break;
        case L'h':
        default:
            print_usage();
            return -1;
        }
    }
#else // _WIN32
    int opt;
    while ((opt = getopt(argc, argv, "0:1:i:o:n:s:m:g:j:f:vxzuh")) != -1)
    {
        switch (opt)
        {
        case '0':
            input0path = optarg;
            break;
        case '1':
            input1path = optarg;
            break;
        case 'i':
            inputpath = optarg;
            break;
        case 'o':
            outputpath = optarg;
            break;
        case 'n':
            numframe = atoi(optarg);
            break;
        // case 's':
        //     timestep = atof(optarg);
        //     break;
        case 's':
            scale = atoi(optarg);
            break;
        case 'm':
            model = optarg;
            break;
        case 'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case 'j':
            sscanf(optarg, "%d:%*[^:]:%d", &jobs_load, &jobs_save);
            jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
            break;
        case 'f':
            pattern_format = optarg;
            break;
        case 'v':
            verbose = 1;
            break;
        case 'x':
            tta_mode = 1;
            break;
        case 'z':
            tta_temporal_mode = 1;
            break;
        case 'u':
            uhd_mode = 1;
            break;
        case 'h':
        default:
            print_usage();
            return -1;
        }
    }
#endif // _WIN32

    if (((input0path.empty() || input1path.empty()) && inputpath.empty()) || outputpath.empty())
    {
        print_usage();
        return -1;
    }

    if (inputpath.empty() && (timestep <= 0.f || timestep >= 1.f))
    {
        fprintf(stderr, "invalid timestep argument, must be 0~1\n");
        return -1;
    }

    if (!inputpath.empty() && numframe < 0)
    {
        fprintf(stderr, "invalid numframe argument, must not be negative\n");
        return -1;
    }

    if (jobs_load < 1 || jobs_save < 1)
    {
        fprintf(stderr, "invalid thread count argument\n");
        return -1;
    }

    if (jobs_proc.size() != (gpuid.empty() ? 1 : gpuid.size()) && !jobs_proc.empty())
    {
        fprintf(stderr, "invalid jobs_proc thread count argument\n");
        return -1;
    }

    for (int i=0; i<(int)jobs_proc.size(); i++)
    {
        if (jobs_proc[i] < 1)
        {
            fprintf(stderr, "invalid jobs_proc thread count argument\n");
            return -1;
        }
    }

    path_t pattern = get_file_name_without_extension(pattern_format);
    path_t format = get_file_extension(pattern_format);

    if (format.empty())
    {
        pattern = PATHSTR("%08d");
        format = pattern_format;
    }

    if (pattern.empty())
    {
        pattern = PATHSTR("%08d");
    }

    if (!path_is_directory(outputpath))
    {
        // guess format from outputpath no matter what format argument specified
        path_t ext = get_file_extension(outputpath);

        if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
        {
            format = PATHSTR("png");
        }
        else if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
        {
            format = PATHSTR("webp");
        }
        else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") || ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
        {
            format = PATHSTR("jpg");
        }
        else
        {
            fprintf(stderr, "invalid outputpath extension type\n");
            return -1;
        }
    }

    if (format != PATHSTR("png") && format != PATHSTR("webp") && format != PATHSTR("jpg"))
    {
        fprintf(stderr, "invalid format argument\n");
        return -1;
    }

    bool rife_v2 = false;
    bool rife_v4 = false;
    bool realesr = false;
    if (model.find(PATHSTR("rife-v2")) != path_t::npos)
    {
        // fine
        rife_v2 = true;
    }
    else if (model.find(PATHSTR("rife-v3")) != path_t::npos)
    {
        // fine
        rife_v2 = true;
    }
    else if (model.find(PATHSTR("rife-v4")) != path_t::npos)
    {
        // fine
        rife_v4 = true;
    }
    else if (model.find(PATHSTR("rife")) != path_t::npos)
    {
        // fine
    }
    else if (model.find(PATHSTR("realesr")) != path_t::npos)
    {
        realesr = true;
    }
    else
    {
        fprintf(stderr, "unknown model dir type\n");
        return -1;
    }

    if (!rife_v4 && (numframe != 0 || timestep != 0.5))
    {
        fprintf(stderr, "only rife-v4 model support custom numframe and timestep\n");
        return -1;
    }

    int frame_count = 0;

    // collect input and output filepath
    std::vector<path_t> input0_files;
    std::vector<path_t> input1_files;
    std::vector<path_t> output_files;
    std::vector<float> timesteps;
    {
        if (!inputpath.empty() && path_is_directory(inputpath) && path_is_directory(outputpath))
        {
            if (realesr) 
            {
                frame_count = get_file_count(inputpath) + 1;
            }
            else
            {
                frame_count = get_file_count(inputpath);
            }
        }
        else if (!inputpath.empty() && !path_is_directory(inputpath) && !outputpath.empty() && !path_is_directory(outputpath))
        {
            input0_files.push_back(inputpath);
            input1_files.push_back(inputpath);
            output_files.push_back(outputpath);
            timesteps.push_back(0);
        }
        // else if (inputpath.empty() && !path_is_directory(input0path) && !path_is_directory(input1path) && !path_is_directory(outputpath))
        else if (inputpath.empty() && !path_is_directory(input0path) && !path_is_directory(input1path) && !path_is_directory(outputpath))
        {
            input0_files.push_back(input0path);
            input1_files.push_back(input1path);
            output_files.push_back(outputpath);
            timesteps.push_back(timestep);
        }
        else
        {
            fprintf(stderr, "input0path, input1path and outputpath must be file at the same time\n");
            fprintf(stderr, "inputpath and outputpath must be directory at the same time\n");
            return -1;
        }
    }
    // int frame_count = get_file_count(inputpath);

    // collect input and output filepath
    // std::vector<path_t> input0_files;
    // std::vector<path_t> input1_files;
    // std::vector<path_t> output_files;
    // std::vector<float> timesteps;
    // {
        // if (!inputpath.empty() && path_is_directory(inputpath) && path_is_directory(outputpath))
        // {
            // std::vector<path_t> filenames;
            // int lr = list_directory(inputpath, filenames);
            // if (lr != 0)
                // return -1;
// 
            // const int count = filenames.size();
            // if (numframe == 0)
                // numframe = count * 2;
// 
            // input0_files.resize(numframe);
            // input1_files.resize(numframe);
            // output_files.resize(numframe);
            // timesteps.resize(numframe);
// 
            // double scale = (double)count / numframe;
            // for (int i=0; i<numframe; i++)
            // {
                // // TODO provide option to control timestep interpolate method
// //                 float fx = (float)((i + 0.5) * scale - 0.5);
                // float fx = i * scale;
                // int sx = static_cast<int>(floor(fx));
                // fx -= sx;
// 
                // if (sx < 0)
                // {
                    // sx = 0;
                    // fx = 0.f;
                // }
                // if (sx >= count - 1)
                // {
                    // sx = count - 2;
                    // fx = 1.f;
                // }
// 
// //                 fprintf(stderr, "%d %f %d\n", i, fx, sx);
// 
                // path_t filename0 = filenames[sx];
                // path_t filename1 = filenames[sx + 1];
// 
// #if _WIN32
                // wchar_t tmp[256];
                // swprintf(tmp, pattern.c_str(), i+1);
// #else
                // char tmp[256];
                // sprintf(tmp, pattern.c_str(), i+1); // ffmpeg start from 1
// #endif
                // path_t output_filename = path_t(tmp) + PATHSTR('.') + format;
// 
                // input0_files[i] = inputpath + PATHSTR('/') + filename0;
                // input1_files[i] = inputpath + PATHSTR('/') + filename1;
                // output_files[i] = outputpath + PATHSTR('/') + output_filename;
                // timesteps[i] = fx;
            // }
        // }
        // else if (inputpath.empty() && !path_is_directory(input0path) && !path_is_directory(input1path) && !path_is_directory(outputpath))
        // {
            // input0_files.push_back(input0path);
            // input1_files.push_back(input1path);
            // output_files.push_back(outputpath);
            // timesteps.push_back(timestep);
        // }
        // else
        // {
            // fprintf(stderr, "input0path, input1path and outputpath must be file at the same time\n");
            // fprintf(stderr, "inputpath and outputpath must be directory at the same time\n");
            // return -1;
        // }
    // }

    path_t modeldir = sanitize_dirpath(model);

    int prepadding = 10;
    // int prepadding = 0;
    // if (model.find(PATHSTR("models")) != path_t::npos
    //     || model.find(PATHSTR("models2")) != path_t::npos)
    // {
    //     prepadding = 10;
    // }
    // else
    // {
    //     fprintf(stderr, "unknown model dir type\n");
    //     return -1;
    // }

    fs::path model_path(model);
    std::string dir_name = model_path.parent_path().filename().string();

    if (dir_name.find("x4") != std::string::npos)
    {
        scale = 4;
    }
    else if (dir_name.find("x3") != std::string::npos)
    {
        scale = 3;
    }
    else if (dir_name.find("x2") != std::string::npos)
    {
        scale = 2;
    }

    path_t paramfullpath;
    path_t modelfullpath;
    
    for (const auto& entry : fs::directory_iterator(model))
    {
        std::string filename = entry.path().filename().string();

        if (filename.find(".param") != std::string::npos)
        {
            paramfullpath = sanitize_filepath(model + PATHSTR('/') + filename.c_str());
        }
        else if (filename.find(".bin") != std::string::npos)
        {
            modelfullpath = sanitize_filepath(model + PATHSTR('/') + filename.c_str());
        }
    }

    // path_t paramfullpath = sanitize_filepath("../models/realesr-animevideov3-x4/realesr-animevideov3-x4.param");
    // path_t modelfullpath = sanitize_filepath("../models/realesr-animevideov3-x4/realesr-animevideov3-x4.bin");

#if _WIN32
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

    ncnn::create_gpu_instance();

    if (gpuid.empty())
    {
        gpuid.push_back(ncnn::get_default_gpu_index());
    }

    const int use_gpu_count = (int)gpuid.size();

    if (jobs_proc.empty())
    {
        jobs_proc.resize(use_gpu_count, 2);
    }

    int cpu_count = std::max(1, ncnn::get_cpu_count());
    jobs_load = std::min(jobs_load, cpu_count);
    jobs_save = std::min(jobs_save, cpu_count);

    int gpu_count = ncnn::get_gpu_count();
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] < -1 || gpuid[i] >= gpu_count)
        {
            fprintf(stderr, "invalid gpu device\n");

            ncnn::destroy_gpu_instance();
            return -1;
        }
    }

    int total_jobs_proc = 0;
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] == -1)
        {
            jobs_proc[i] = std::min(jobs_proc[i], cpu_count);
            total_jobs_proc += 1;
        }
        else
        {
            total_jobs_proc += jobs_proc[i];
        }
    }

    {
        std::vector<RIFE*> rife(use_gpu_count);

        for (int i=0; i<use_gpu_count; i++)
        {
            int num_threads = gpuid[i] == -1 ? jobs_proc[i] : 1;

            if (realesr)
            {
                rife[i] = nullptr;
            }
            else
            {
                rife[i] = new RIFE(gpuid[i], tta_mode, tta_temporal_mode, uhd_mode, num_threads, rife_v2, rife_v4);

                rife[i]->load(modeldir);
            }
        }

        std::vector<RealESRGAN*> realesrgan(use_gpu_count);

        for (int i=0; i<use_gpu_count; i++)
        {
            if (realesr)
            {
                int tilesize = 0;
                uint32_t heap_budget = ncnn::get_gpu_device(gpuid[i])->get_heap_budget();

                if (heap_budget > 1900)
                    tilesize = 200;
                else if (heap_budget > 550)
                    tilesize = 100;
                else if (heap_budget > 190)
                    tilesize = 64;
                else
                    tilesize = 32;

                realesrgan[i] = new RealESRGAN(gpuid[i], tta_mode);

                realesrgan[i]->load(paramfullpath, modelfullpath);

                realesrgan[i]->scale = scale;
                realesrgan[i]->tilesize = tilesize;
                realesrgan[i]->prepadding = prepadding;
            }
            else 
            {
                realesrgan[i] = nullptr;
            }
        }

        // main routine
        {
            // load image
            LoadThreadParams ltp;
            ltp.jobs_load = jobs_load;
            ltp.input0_files = input0_files;
            ltp.input1_files = input1_files;
            ltp.output_files = output_files;
            ltp.timesteps = timesteps;
            ltp.input_dir = inputpath;
            ltp.output_dir = outputpath;
            ltp.frame_count = frame_count;
            ltp.realesr = realesr;
            ltp.scale = scale;

            ncnn::Thread load_thread(load, (void*)&ltp);

            // rife proc
            std::vector<ProcThreadParams> ptp(use_gpu_count);
            for (int i=0; i<use_gpu_count; i++)
            {
                ptp[i].rife = rife[i];
                ptp[i].realesrgan = realesrgan[i];
            }

            std::vector<ncnn::Thread*> proc_threads(total_jobs_proc);
            {
                int total_jobs_proc_id = 0;
                for (int i=0; i<use_gpu_count; i++)
                {
                    if (gpuid[i] == -1)
                    {
                        proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                    }
                    else
                    {
                        for (int j=0; j<jobs_proc[i]; j++)
                        {
                            proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                        }
                    }
                }
            }

            // save image
            SaveThreadParams stp;
            stp.verbose = verbose;
            stp.input_dir = inputpath;
            stp.output_dir = outputpath;
            stp.realesr = realesr;

            std::vector<ncnn::Thread*> save_threads(jobs_save);
            for (int i=0; i<jobs_save; i++)
            {
                save_threads[i] = new ncnn::Thread(save, (void*)&stp);
            }

            // end
            load_thread.join();

            Task end;
            end.id = -233;

            for (int i=0; i<total_jobs_proc; i++)
            {
                toproc.put(end);
            }

            for (int i=0; i<total_jobs_proc; i++)
            {
                proc_threads[i]->join();
                delete proc_threads[i];
            }

            for (int i=0; i<jobs_save; i++)
            {
                tosave.put(end);
            }

            for (int i=0; i<jobs_save; i++)
            {
                save_threads[i]->join();
                delete save_threads[i];
            }
        }

        for (int i=0; i<use_gpu_count; i++)
        {
            delete rife[i];
        }
        rife.clear();
    }

    ncnn::destroy_gpu_instance();

    return 0;
}
