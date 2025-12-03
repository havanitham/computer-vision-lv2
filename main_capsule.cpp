#include <opencv2/opencv.hpp>
#include <iostream>
#include <set>
#include <cmath>

static double chroma_lab(float L, float a, float b)
{
    return std::sqrt((a - 128.0f)*(a - 128.0f) + (b - 128.0f)*(b - 128.0f));
}

cv::Mat kmeans_pill_mask(const cv::Mat &img)
{
    int h = img.rows;
    int w = img.cols;

    cv::Mat lab;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);

    int sw = w * 0.25;
    int sh = h * 0.25;
    if (sw < 1) sw = 1;
    if (sh < 1) sh = 1;

    cv::Mat small;
    cv::resize(lab, small, cv::Size(sw, sh), 0, 0, cv::INTER_AREA);

    cv::Mat samples(sh * sw, 3, CV_32F);

    for (int y = 0; y < sh; y++)
        for (int x = 0; x < sw; x++)
        {
            cv::Vec3b c = small.at<cv::Vec3b>(y, x);
            int idx = y * sw + x;

            samples.at<float>(idx, 0) = c[0];
            samples.at<float>(idx, 1) = c[1];
            samples.at<float>(idx, 2) = c[2];
        }

    cv::Mat labels, centers;
    cv::TermCriteria crit(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 20, 1.0);

    cv::kmeans(samples, 2, labels, crit, 5, cv::KMEANS_PP_CENTERS, centers);

    int pill_label = 0;
    double best = -1;

    for (int k = 0; k < 2; k++)
    {
        float L = centers.at<float>(k, 0);
        float a = centers.at<float>(k, 1);
        float b = centers.at<float>(k, 2);

        double ch = chroma_lab(L, a, b);
        if (ch > best)
        {
            best = ch;
            pill_label = k;
        }
    }

    cv::Mat labelImg = labels.reshape(1, sh);
    cv::Mat mask_small(sh, sw, CV_8U);

    for (int y = 0; y < sh; y++)
        for (int x = 0; x < sw; x++)
            mask_small.at<uchar>(y, x) = (labelImg.at<int>(y, x) == pill_label ? 255 : 0);

    cv::Mat mask;
    cv::resize(mask_small, mask, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);

    return mask;
}

int detect_capsule_pills(const std::string &path)
{
    cv::Mat img = cv::imread(path);
    if (img.empty())
    {
        std::cout << "Image not found\n";
        return -1;
    }

    int h = img.rows;
    int w = img.cols;

    cv::Mat mask = kmeans_pill_mask(img);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    cv::Mat dist;
    cv::distanceTransform(mask, dist, cv::DIST_L2, 5);

    double mn, mx;
    cv::minMaxLoc(dist, &mn, &mx);

    cv::Mat sure_fg;
    cv::threshold(dist, sure_fg, 0.3 * mx, 255, cv::THRESH_BINARY);
    sure_fg.convertTo(sure_fg, CV_8U);

    cv::Mat sure_bg;
    cv::dilate(mask, sure_bg, kernel);

    cv::Mat unknown;
    cv::subtract(sure_bg, sure_fg, unknown);

    cv::Mat markers;
    cv::connectedComponents(sure_fg, markers);

    markers += 1;
    markers.setTo(0, unknown == 255);

    cv::watershed(img, markers);

    cv::Mat out = img.clone();
    int count = 0;

    std::set<int> labs;
    for (int y = 0; y < h; y++)
    {
        const int* r = markers.ptr<int>(y);
        for (int x = 0; x < w; x++)
            labs.insert(r[x]);
    }

    for (int lbl : labs)
    {
        if (lbl <= 1) continue;

        cv::Mat comp = (markers == lbl);
        comp.convertTo(comp, CV_8U);

        std::vector<std::vector<cv::Point>> cs;
        cv::findContours(comp, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (cs.empty()) continue;

        double bestArea = 0;
        int bi = 0;

        for (int i = 0; i < cs.size(); i++)
        {
            double a = cv::contourArea(cs[i]);
            if (a > bestArea)
            {
                bestArea = a;
                bi = i;
            }
        }

        count++;
        cv::drawContours(out, cs, bi, cv::Scalar(0,255,0), 2);

        cv::Moments M = cv::moments(cs[bi]);
        if (M.m00 != 0)
        {
            int cx = M.m10 / M.m00;
            int cy = M.m01 / M.m00;
            cv::putText(out, std::to_string(count), cv::Point(cx, cy),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 2);
        }
    }

    cv::imshow("Mask", mask);
    cv::imshow("Output", out);
    cv::waitKey(0);

    return count;
}

int main()
{
    detect_capsule_pills("C:/Users/havan/Downloads/Input_Images/orange_17.png");
    return 0;
}
