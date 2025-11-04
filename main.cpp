#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string imagePath = "D:/opencv_project/input_images/yellow_capsule_57.png";
    cv::Mat image = cv::imread(imagePath);

    // Convert to Lab and use 'b' channel 
    cv::Mat lab; cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    cv::Mat b = lab.clone();
    std::vector<cv::Mat> ch; cv::split(lab, ch);
    b = ch[2];

    // Threshold to isolate pills
    cv::GaussianBlur(b, b, cv::Size(5, 5), 0);
    cv::Mat mask;
    cv::threshold(b, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Clean small noise
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);

    // Remove small blobs
    cv::Mat lbl, st, cen;
    int n = cv::connectedComponentsWithStats(mask, lbl, st, cen);
    for (int i = 1; i < n; i++)
        if (st.at<int>(i, cv::CC_STAT_AREA) < 800)
            mask.setTo(0, lbl == i);

    // Split touching pills using watershed
    cv::Mat dist; cv::distanceTransform(mask, dist, cv::DIST_L2, 5);
    double maxVal; cv::minMaxLoc(dist, nullptr, &maxVal);
    cv::Mat fg; cv::threshold(dist, fg, 0.55 * maxVal, 255, cv::THRESH_BINARY);
    fg.convertTo(fg, CV_8U);
    cv::Mat bg; cv::dilate(mask, bg, k, cv::Point(-1,-1), 3);
    cv::Mat unknown; cv::subtract(bg, fg, unknown);
    cv::Mat markers; int nf = cv::connectedComponents(fg, markers);
    markers += 1; markers.setTo(0, unknown == 255);
    cv::watershed(image, markers);

    // Final mask and counting
    cv::Mat pills = (markers > 1);
    pills.convertTo(pills, CV_8U, 255);

    cv::Mat lbl2, st2, cen2;
    int total = cv::connectedComponentsWithStats(pills, lbl2, st2, cen2);
    cv::Mat result = image.clone();
    int count = 0;

    for (int i = 1; i < total; i++) {
        if (st2.at<int>(i, cv::CC_STAT_AREA) < 800) continue;
        count++;
        cv::Mat comp = (lbl2 == i); comp.convertTo(comp, CV_8U, 255);
        std::vector<std::vector<cv::Point>> c;
        cv::findContours(comp, c, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (c.empty()) continue;
        cv::drawContours(result, c, -1, cv::Scalar(0,255,0), 2);
        cv::Moments M = cv::moments(c[0]);
        if (M.m00 > 0)
            cv::putText(result, std::to_string(count),
                        {int(M.m10/M.m00)-10, int(M.m01/M.m00)-10},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,255}, 2);
    }

    std::cout << "Pills detected: " << count << "\n";
    cv::imshow("Mask", pills);
    cv::imshow("Detected Pills", result);
    cv::waitKey(0);
    return 0;
}