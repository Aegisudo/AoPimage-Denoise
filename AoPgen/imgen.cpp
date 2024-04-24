#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <opencv2\opencv.hpp>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include <Eigen\core>
#include <Eigen\Dense>
#include <opencv2\core\eigen.hpp>

using namespace cv;
using namespace std;

typedef std::numeric_limits<double> Info;
double const NAN_d = Info::quiet_NaN();

void Vector2Mat(vector<vector<double>>src, Mat dst);
void cv_to_eigen(const Mat& input, Eigen::Matrix3d& output);

int main()
{
    string loc1 = "E:\\sunvector\\sunvector\\0.JPG";
    string loc2 = "E:\\sunvector\\sunvector\\45.JPG";
    string loc3 = "E:\\sunvector\\sunvector\\90.JPG";
    string loc4 = "E:\\sunvector\\sunvector\\135.JPG";

    Mat src1 = imread(loc1);
    Mat src2 = imread(loc2);
    Mat src3 = imread(loc3);
    Mat src4 = imread(loc4);

    src1.convertTo(src1, CV_64FC3, 1.0, 0);
    src2.convertTo(src2, CV_64FC3, 1.0, 0);
    src3.convertTo(src3, CV_64FC3, 1.0, 0);
    src4.convertTo(src4, CV_64FC3, 1.0, 0);
    int x1, x2, y1, y2;
    int rr, oo, dd;

    Mat image0 = src1(Rect(Point(x1, y1), Point(x2, y2)));
    Mat image45 = src2(Rect(Point(x1, y1), Point(x2, y2)));
    Mat image90 = src3(Rect(Point(x1, y1), Point(x2, y2)));
    Mat image135 = src4(Rect(Point(x1, y1), Point(x2, y2)));

    vector<Mat> planes0;
    split(image0, planes0);
    Mat I0 = (planes0[0] + planes0[1] + planes0[2]) / 3;

    vector<Mat> planes45;
    split(image45, planes45);
    Mat I45 = (planes45[0] + planes45[1] + planes45[2]) / 3;

    vector<Mat> planes90;
    split(image90, planes90);
    Mat I90 = (planes90[0] + planes90[1] + planes90[2]) / 3;

    vector<Mat> planes135;
    split(image135, planes135);
    Mat I135 = (planes135[0] + planes135[1] + planes135[2]) / 3;

    Mat I = (I0 + I45 + I90 + I135) / 2;
    Mat Q = I0 - I90;
    Mat U = I45 - I135;

    int rows = I.rows; int cols = I.cols;
    double PI = 3.1415926;
    Mat aop(rows, cols, CV_64FC1);

    for (unsigned int i = 0; i < rows; i++)
    {
        const double* Qptr = Q.ptr<double>(i);
        const double* Uptr = U.ptr<double>(i);
        double* aopPtr = aop.ptr<double>(i);
        for (unsigned int j = 0; j < cols; j++)
        {
            *aopPtr = (0.5 * atan2(*Uptr, *Qptr)) * (180 / PI);
            aopPtr++;
            Uptr++; Qptr++;
        }
    }

    Mat aop_last = Mat::zeros(rows, cols, CV_64FC1);

    for (unsigned int i = 0; i < rows; i++)
    {
        const double* aopPtr = aop.ptr<double>(i);
        double* aop_lastPtr = aop_last.ptr<double>(i);
        for (unsigned int j = 0; j < cols; j++)
        {
            double x = i; double y = j;
            if (j == oo)
            {
                *aop_lastPtr = *aopPtr + (PI / 2) * (180 / PI);
            }
            else
            {
                *aop_lastPtr = *aopPtr - (atan((x - oo) / (oo - y))) * (180 / PI);
                if (*aop_lastPtr < -90)
                {
                    *aop_lastPtr = 180 + *aop_lastPtr;
                }
                else if (*aop_lastPtr > 90)
                {
                    *aop_lastPtr = *aop_lastPtr - 180;
                }
            }
            aopPtr++; aop_lastPtr++;
        }
    }

    Mat aop_circle = Mat::zeros(rows, cols, CV_64FC1);

    for (unsigned int i = 0; i < rows; i++)
    {
        const double* aop_lastPtr = aop_last.ptr<double>(i);
        double* aop_circlePtr = aop_circle.ptr<double>(i);
        for (unsigned int j = 0; j < cols; j++)
        {
            if ((i - oo) * (i - oo) + (j - oo) * (j - oo) < rr * rr)
            {
                *aop_circlePtr = *aop_lastPtr;
            }
            else
            {
                *aop_circlePtr = NAN_d;
            }
            aop_lastPtr++; aop_circlePtr++;
        }
    }

    Mat aop_color; Mat aop_show = Mat::zeros(rows, cols, CV_64FC3);
    double min_aop, max_aop, alpha_aop;

    minMaxLoc(aop_circle, &min_aop, &max_aop);
    Mat aop_circle1 = aop_circle;
    alpha_aop = 255.0 / (max_aop - min_aop);
    aop_circle1.convertTo(aop_circle1, CV_8U, alpha_aop, -min_aop * alpha_aop);
    applyColorMap(aop_circle1, aop_color, COLORMAP_JET);

    imwrite("aop.bmp", aop_color);
    string aop_map1 = "E:\\sunvector\\sunvector\\aop.bmp";

    Mat aop_map = imread(aop_map1);
    aop_map.convertTo(aop_map, CV_64FC3, 1 / 255.0, 0);
    //imshow("aop",aop_map);

    vector<Mat> channels(3);
    split(aop_map, channels);

    for (unsigned int i = 0; i < rows; i++)
    {
        const double* channel0Ptr = channels[0].ptr<double>(i);
        const double* channel1Ptr = channels[1].ptr<double>(i);
        const double* channel2Ptr = channels[2].ptr<double>(i);
        double* aop_showPtr = aop_show.ptr<double>(i);
        for (unsigned int j = 0; j < cols; j++)
        {
            if ((i - oo)*(i - oo) + (j - oo)*(j - oo) < rr * rr)
            {
                *aop_showPtr++ = *channel0Ptr; *aop_showPtr++ = *channel1Ptr; *aop_showPtr++ = *channel2Ptr;
            }
            else
            {
                *aop_showPtr++ = 1; *aop_showPtr++ = 1; *aop_showPtr++ = 1;
            }
            channel0Ptr++; channel1Ptr++; channel2Ptr++;
        }
    }

    imshow("aop_show",aop_show);
    aop_show.convertTo(aop_show, CV_8UC3, 255.0, 0);
    imwrite("aop_save.bmp", aop_show);

    int amount = 10000; int q = 0;
    vector<vector<double>> Su;

    for (unsigned int inter = 0; inter < amount; inter++)
    {
        int i1 = rand() % (rows - 1);
        int j1 = rand() % (cols - 1);
        int i2 = rand() % (rows - 1);
        int j2 = rand() % (cols - 1);
        if (((i1 - oo) * (i1 - oo) + (j1 - oo) * (j1 - oo) < rr * rr) && ((i2 - oo) * (i2 - oo) + (j2 - oo) * (j2 - oo) < rr * rr))
        {
            q++;
            double x1 = i1 - oo; double y1 = j1 - oo;
            double x2 = i2 - oo; double y2 = j2 - oo;

            double k1 = (aop_circle.at<double>(i1, j1)) * (PI / 180);
            double k2 = (aop_circle.at<double>(i2, j2)) * (PI / 180);

            double b1 = atan(y1 / x1);
            double b2 = atan(y2 / x2);

            double a1 = atan(sqrt(x1 * x1 + y1 * y1) * (0.0087 / 8));
            double a2 = atan(sqrt(x2 * x2 + y2 * y2) * (0.0087 / 8));

            Eigen::Matrix<double, 3, 3> C11;
            Eigen::Matrix<double, 3, 3> C12;
            C11 << cos(a1), 0, -sin(a1), 0, 1, 0, sin(a1), 0, cos(a1);
            C12 << cos(b1), sin(b1), 0, -sin(b1), cos(b1), 0, 0, 0, 1;
            Eigen::Matrix<double, 3, 3> Cli1 = C11 * C12;

            Eigen::Matrix<double, 3, 3> C21;
            Eigen::Matrix<double, 3, 3> C22;
            C21 << cos(a2), 0, -sin(a2), 0, 1, 0, sin(a2), 0, cos(a2);
            C22 << cos(b2), sin(b2), 0, -sin(b2), cos(b2), 0, 0, 0, 1;
            Eigen::Matrix<double, 3, 3> Cli2 = C21 * C22;

            Eigen::Matrix<double, 1, 3> PEi1;
            Eigen::Matrix<double, 1, 3> PEi2;
            PEi1 << cos(k1), sin(k1), 0;
            PEi2 << cos(k2), sin(k2), 0;

            Eigen::Matrix<double, 1, 3> e1 = PEi1 * Cli1;
            Eigen::Matrix<double, 1, 3> e2 = PEi2 * Cli2;

            Eigen::Vector3d Evec1 = Eigen::Vector3d(e1(0, 0), e1(0, 1), e1(0, 2));
            Eigen::Vector3d Evec2 = Eigen::Vector3d(e2(0, 0), e2(0, 1), e2(0, 2));
            Eigen::Vector3d S0 = Evec1.cross(Evec2);
            double normS0 = sqrt(S0[0] * S0[0] + S0[1] * S0[1] + S0[2] * S0[2]);
            Eigen::Vector3d S = S0 / normS0;

            vector<double> s(3);
            s[0] = S[0]; s[1] = S[1]; s[2] = S[2];

            Su.push_back(s);
            if (q > 999)
            {
                break;
            }
        }
    }

    Mat Smat(Su.size(), 3, CV_64FC1);
    Vector2Mat(Su, Smat);
    Mat S1;
    S1 = (Smat.t()) * Smat;

    Eigen::Matrix3d SunMat;
    cv_to_eigen(S1, SunMat);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(SunMat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d SigularVal = svd.singularValues();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Vector3d SunVector = Eigen::Vector3d(V(0, 0), V(1, 0), V(2, 0));
    double posAngle = atan2(SunVector(1), SunVector(0)) * (180 / PI);

    cout << "The orientation angle is:" << posAngle << " degree" << endl;

    waitKey(0);
    return 0;
}

void Vector2Mat(vector<vector<double>>src, Mat dst)
{
    assert(dst.rows == src.size());
    MatIterator_<double> it = dst.begin<double>();
    for (int i = 0; i < src.size(); i++)
    {
        for (int j = 0; j < src[0].size(); j++)
        {
            *it = src[i][j];
            it++;
        }
    }
}

void cv_to_eigen(const Mat& input, Eigen::Matrix3d& output)
{
    cv2eigen(input, output);
}
