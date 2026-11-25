#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>


bool isRightAngle(cv::Point a, cv::Point b, cv::Point c) {
    cv::Point ab = a - b;
    cv::Point cb = c - b;
    double dot = ab.dot(cb);
    double len1 = cv::norm(ab);
    double len2 = cv::norm(cb);
    if (len1 == 0 || len2 == 0) return false;
    double cosine = dot / (len1 * len2);
    return std::abs(cosine) < 0.3;
}

bool isRectangle(const std::vector<cv::Point> &contour) {
    double perimeter = cv::arcLength(contour, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);
    if (approx.size() != 4) return false;
    if (!cv::isContourConvex(approx)) return false;
    for (int i = 0; i < 4; i++) {
        if (!isRightAngle(approx[(i + 3) % 4], approx[i], approx[(i + 1) % 4])) {
            return false;
        }
    }

    return true;
}

int main() {
    cv::VideoCapture cap(0);
    cv::Mat frame, hsv;
    cv::Mat blueMask, colorMask, mask;
    cv::CascadeClassifier faceCascade;
    cv::CascadeClassifier smileCascade;
    faceCascade.load("/home/veunder/CLionProjects/CLab2/haarcascades/haarcascade_frontalface_default.xml");
    smileCascade.load("/home/veunder/CLionProjects/CLab2/haarcascades/haarcascade_smile.xml");
    while (cap.read(frame)) {
        cv::flip(frame, frame, 1);
        // делаем серую маску для определения лица и улыбки
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::Scalar blueMin(90, 50, 50);
        cv::Scalar blueMax(130, 255, 255);
        // создание цветовой маски (на синий цвет в нашем случае)
        cv::inRange(hsv, blueMin, blueMax, blueMask);
        cv::inRange(hsv, cv::Scalar(0, 50, 40), cv::Scalar(179, 255, 255), colorMask);
        // инверсия цветовой маски (убираем синий цвет)
        cv::bitwise_not(blueMask, blueMask);
        cv::bitwise_and(colorMask, blueMask, mask);
        // убираем мелкий шум
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        // проверяем каждый контур
        for (const auto &contour: contours) {
            double area = cv::contourArea(contour);
            // фильтр по площади
            if (area < 700 || area > 150200)
                continue;
            if (!isRectangle(contour))
                continue;
            // аппроксимация контура
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contour, approx, 0.02 * cv::arcLength(contour, true), true);
            // рисуем найденный прямоугольник
            cv::polylines(frame, approx, true, cv::Scalar(0, 255, 0), 3);
            cv::putText(frame, "Rectangle", approx[0], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        // поиск лиц
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(50, 50));
        // проверяем каждое найденное лицо
        for (const auto &face: faces) {
            cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);
            cv::Mat faceROI = gray(face);
            // Берём нижнюю половину лица
            cv::Rect lowerFace(0, face.height / 2, face.width, face.height / 2);
            cv::Mat smileROI = faceROI(lowerFace);
            std::vector<cv::Rect> smiles;
            smileCascade.detectMultiScale(smileROI, smiles, 1.7, 20, 0, cv::Size(25, 15));
            for (const auto &smile: smiles) {
                cv::Rect smileRect(face.x + smile.x, face.y + face.height / 2 + smile.y, smile.width, smile.height);
                cv::rectangle(frame, smileRect, cv::Scalar(0, 255, 255), 2);
                cv::putText(frame, "Smile", cv::Point(smileRect.x, smileRect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0, 255, 255), 2);
            }
        }
        cv::imshow("Original video", frame);
        cv::imshow("Color mask", mask);
        cv::imshow("Blue mask", blueMask);
        // ESC для выхода
        if (cv::waitKey(30) == 27)
            break;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
