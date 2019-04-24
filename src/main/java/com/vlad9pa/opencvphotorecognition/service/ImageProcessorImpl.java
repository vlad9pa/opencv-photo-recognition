package com.vlad9pa.opencvphotorecognition.service;

import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class ImageProcessorImpl implements ImageProcessor{

    private final static String FACE_LIB_PATH = "/haarcascades/haarcascade_frontalface_alt2.xml";
    private final static String EYE_LIB_PATH = "/haarcascades/haarcascade_eye_the_best.xml";

    @Override
    @SneakyThrows
    public byte[] detectFace(MultipartFile file){
        return detectMultiScale(file, FACE_LIB_PATH);
    }

    private byte[] detectMultiScale(MultipartFile file, String faceLibPath) throws IOException {
        MatOfRect faceDetections = new MatOfRect();
        CascadeClassifier faceDetector = new CascadeClassifier(this.getClass().getResource(faceLibPath).getPath());

        Mat image = Imgcodecs.imdecode(new MatOfByte(file.getBytes()), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
        faceDetector.detectMultiScale(image, faceDetections);

        log.info("Face detected. Count = {}", faceDetections.size());

        drawRect(faceDetections, image);

        return mat2Image(image);
    }

    @Override
    @SneakyThrows
    public byte[] detectEye(MultipartFile file) {
        return detectMultiScale(file, EYE_LIB_PATH);
    }

    @Override
    @SneakyThrows
    public byte[] backgroundColor(MultipartFile file) {
        Mat image = Imgcodecs.imdecode(new MatOfByte(file.getBytes()), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);

        return detectObjectOnWhiteBackground(file);
    }

    private byte[] backgroundRemoval(Mat image) {
        Mat hsvImg = new Mat();
        List<Mat> hsvPlanes = new ArrayList<>();
        Mat thresholdImg = new Mat();

        int thresh_type = Imgproc.THRESH_BINARY;

        // threshold the image with the average hue value
        hsvImg.create(image.size(), CvType.CV_32F);
        Imgproc.cvtColor(image, hsvImg, Imgproc.COLOR_BGR2HSV);
        Core.split(hsvImg, hsvPlanes);

        // get the average hue value of the image
        double threshValue = this.getHistAverage(hsvImg, hsvPlanes.get(0));

        Imgproc.threshold(hsvPlanes.get(0), thresholdImg, threshValue, 179.0, thresh_type);

        Imgproc.blur(thresholdImg, thresholdImg, new Size(5, 5));

        // dilate to fill gaps, erode to smooth edges
        Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 1);
        Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 3);

        Imgproc.threshold(thresholdImg, thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);

        // create the new image
        Mat foreground = new Mat(image.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        image.copyTo(foreground, thresholdImg);

        return mat2Image(foreground);
    }

    private double getHistAverage(Mat hsvImg, Mat hueValues)
    {
        // init
        double average = 0.0;
        Mat hist_hue = new Mat();
        // 0-180: range of Hue values
        MatOfInt histSize = new MatOfInt(180);
        List<Mat> hue = new ArrayList<>();
        hue.add(hueValues);

        // compute the histogram
        Imgproc.calcHist(hue, new MatOfInt(0), new Mat(), hist_hue, histSize, new MatOfFloat(0, 179));

        // get the average Hue value of the image
        // (sum(bin(h)*h))/(image-height*image-width)
        // -----------------
        // equivalent to get the hue of each pixel in the image, add them, and
        // divide for the image size (height and width)
        for (int h = 0; h < 180; h++)
        {
            // for each bin, get its value and multiply it for the corresponding
            // hue
            average += (hist_hue.get(h, 0)[0] * h);
        }

        // return the average hue of the image
        return average / hsvImg.size().height / hsvImg.size().width;
    }

    private byte[] detectObjectOnWhiteBackground(MultipartFile file) throws IOException {
        Mat image = Imgcodecs.imdecode(new MatOfByte(file.getBytes()), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
        Mat lab = new Mat();
        Imgproc.cvtColor(image, lab, Imgproc.COLOR_BGR2Lab);
        List<Mat> labList = new ArrayList<>();
        Core.split(lab, labList);

        Mat bin = new Mat();
        Imgproc.adaptiveThreshold(labList.get(2), bin, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 3, 3);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_DILATE, kernel);
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(bin, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Point> points = new ArrayList<>();
        contours.forEach(m -> {
            List<Point> collection = m.toList();
            points.addAll(collection);
        });

        Point[] pointsArray = new Point[points.size()];
        points.toArray(pointsArray);

        MatOfPoint matOfPoint = new MatOfPoint(pointsArray);
        Rect rect = Imgproc.boundingRect(matOfPoint);

        Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 0, 255));

        return mat2Image(image);
    }

    private void drawRect(MatOfRect faceDetections, Mat image) {
        for(Rect rect: faceDetections.toArray()){
            Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 0, 255));
        }
    }

    private byte[] mat2Image(Mat frame) {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".jpg", frame, buffer);
        return buffer.toArray();
    }
}
