#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;


string save_with_suffix(const string &path, const string &suffix) {
    size_t dotPos = path.find_last_of('.');
    string base = (dotPos == string::npos) ? path : path.substr(0, dotPos);
    return base + suffix + ".png";
}


int detect_round(const string &path, const Mat &img) {
    Mat gray, gray_blur;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray_blur, 5);

    int h = gray.rows;
    int w = gray.cols;

    int min_r = static_cast<int>(min(h, w) * 0.07);   
    int max_r = static_cast<int>(min(h, w) * 0.12);   

    vector<Vec3f> circles;
    HoughCircles(
        gray_blur,
        circles,
        HOUGH_GRADIENT,
        1.2,            
        min_r,             
        100,               
        30,                
        min_r,             
        max_r              
    );

    Mat out = img.clone();
    int count = 0;

    for (size_t i = 0; i < circles.size(); ++i) {
        float x = circles[i][0];
        float y = circles[i][1];
        float r = circles[i][2];

        Point center(cvRound(x), cvRound(y));
        int radius = cvRound(r);

        circle(out, center, radius, Scalar(0, 255, 0), 2);
        putText(out, to_string(count + 1),
                Point(center.x - radius / 2, center.y),
                FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(255, 0, 0), 2);

        count++;
    }

    cout << "Round pills detected: " << count << endl;

    string save_path = save_with_suffix(path, "_round_detected");
    imwrite(save_path, out);
    cout << "Saved: " << save_path << endl;

    imshow("Round pills", out);
    waitKey(0);
    destroyAllWindows();

    return count;
}


int main() {
    string image_path = "C:/Users/swara/Downloads/imgs/img3.jpeg";

    Mat img = imread(image_path);
    if (img.empty()) {
        cerr << "Could not read image" << endl;
        return -1;
    }

    detect_round(image_path, img);
    return 0;
}
