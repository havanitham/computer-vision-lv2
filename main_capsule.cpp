#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <string>
#include <set>


std::string save_with_suffix(const std::string &path, const std::string &suffix = "_detected")
{

    size_t slash_pos = path.find_last_of("/\\");
    if (slash_pos == std::string::npos)
        slash_pos = 0;
    else
        slash_pos += 1;

    size_t dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos || dot_pos < slash_pos)
    {
  
        return path + suffix + ".png";
    }
    std::string base = path.substr(0, dot_pos);
    return base + suffix + ".png";
}


static double chroma_lab(float L, float a, float b)
{
  
    return std::sqrt((a - 128.0f) * (a - 128.0f) + (b - 128.0f) * (b - 128.0f));
}

cv::Mat kmeans_pill_mask(const cv::Mat &img, int K = 2, double scale = 0.25)
{
    int h = img.rows;
    int w = img.cols;

    cv::Mat lab;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);

    int small_w = std::max(1, static_cast<int>(w * scale));
    int small_h = std::max(1, static_cast<int>(h * scale));

    cv::Mat small;
    cv::resize(lab, small, cv::Size(small_w, small_h), 0, 0, cv::INTER_AREA);


    cv::Mat samples(small_h * small_w, 3, CV_32F);
    for (int y = 0; y < small_h; ++y)
    {
        for (int x = 0; x < small_w; ++x)
        {
            cv::Vec3b c = small.at<cv::Vec3b>(y, x);
            int idx = y * small_w + x;
            samples.at<float>(idx, 0) = static_cast<float>(c[0]); // L
            samples.at<float>(idx, 1) = static_cast<float>(c[1]); // a
            samples.at<float>(idx, 2) = static_cast<float>(c[2]); // b
        }
    }

    cv::Mat labels;
    cv::Mat centers; // K x 3, CV_32F
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 20, 1.0);

    cv::kmeans(samples, K, labels, criteria, 5, cv::KMEANS_PP_CENTERS, centers);


    int pill_label = 0;
    double max_chroma = -1.0;
    for (int k = 0; k < K; ++k)
    {
        float L = centers.at<float>(k, 0);
        float a = centers.at<float>(k, 1);
        float b = centers.at<float>(k, 2);
        double ch = chroma_lab(L, a, b);
        if (ch > max_chroma)
        {
            max_chroma = ch;
            pill_label = k;
        }
    }

    cv::Mat labelImg = labels.reshape(1, small_h); 
    cv::Mat mask_small(small_h, small_w, CV_8U);
    for (int y = 0; y < small_h; ++y)
    {
        for (int x = 0; x < small_w; ++x)
        {
            int lbl = labelImg.at<int>(y, x);
            mask_small.at<uchar>(y, x) = (lbl == pill_label) ? 255 : 0;
        }
    }


    cv::Mat mask;
    cv::resize(mask_small, mask, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);

    return mask; 
}


int detect_capsule_pills(const std::string &path, bool show_windows = true)
{
    cv::Mat img = cv::imread(path);
    if (img.empty())
    {
        throw std::runtime_error("Could not read image: " + path);
    }

    int h = img.rows;
    int w = img.cols;

  
    cv::Mat mask = kmeans_pill_mask(img, 2, 0.25);


    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

 
    cv::Mat dist;
    cv::distanceTransform(mask, dist, cv::DIST_L2, 5);

    double minVal, maxVal;
    cv::minMaxLoc(dist, &minVal, &maxVal);

    cv::Mat sureFgFloat;
    cv::threshold(dist, sureFgFloat, 0.30 * maxVal, 255.0, cv::THRESH_BINARY);

    cv::Mat sure_fg;
    sureFgFloat.convertTo(sure_fg, CV_8U);

    cv::Mat sure_bg;
    cv::dilate(mask, sure_bg, kernel, cv::Point(-1, -1), 2);

    cv::Mat unknown;
    cv::subtract(sure_bg, sure_fg, unknown);

    cv::Mat markers;
    int n_labels = cv::connectedComponents(sure_fg, markers);
    markers += 1;                       
    markers.setTo(0, unknown == 255);   


    cv::watershed(img, markers);        


    cv::Mat out = img.clone();
    int pill_id = 0;
    double min_area = 0.0005 * h * w;  

    std::set<int> labels_set;
    for (int y = 0; y < h; ++y)
    {
        const int *row = markers.ptr<int>(y);
        for (int x = 0; x < w; ++x)
        {
            labels_set.insert(row[x]);
        }
    }

    for (int lbl : labels_set)
    {

        if (lbl <= 1) continue;


        cv::Mat comp_mask = (markers == lbl);
        comp_mask.convertTo(comp_mask, CV_8U); 

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(comp_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty())
            continue;


        size_t best_idx = 0;
        double best_area = 0.0;
        for (size_t i = 0; i < contours.size(); ++i)
        {
            double a = cv::contourArea(contours[i]);
            if (a > best_area)
            {
                best_area = a;
                best_idx = i;
            }
        }

        if (best_area < min_area)
            continue;

        std::vector<cv::Point> c = contours[best_idx];


        double eps = 0.01 * cv::arcLength(c, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, eps, true);

        ++pill_id;
        cv::drawContours(out, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 2);

        cv::Moments M = cv::moments(approx);
        if (M.m00 != 0.0)
        {
            int cx = static_cast<int>(M.m10 / M.m00);
            int cy = static_cast<int>(M.m01 / M.m00);
            cv::putText(out, std::to_string(pill_id),
                        cv::Point(cx - 10, cy + 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        }
    }

    std::cout << "Pills detected in " << path << ": " << pill_id << std::endl;

    std::string save_path = save_with_suffix(path, "_capsule_detected");
    cv::imwrite(save_path, out);
    std::cout << "Saved: " << save_path << std::endl;

    if (show_windows)
    {
        cv::imshow("Mask", mask);
        cv::imshow("Capsule pills", out);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return pill_id;
}


int main()
{

    std::string image_path = "C:/Users/havan/Downloads/Input_Images/yellow.jpg";

    try
    {
        detect_capsule_pills(image_path, true);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

