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

@Slf4j
@Service
@RequiredArgsConstructor
public class ImageProcessorImpl implements ImageProcessor{

    private final static String FACE_LIB_PATH = "/haarcascades/haarcascade_frontalface_alt.xml";

    @Override
    @SneakyThrows
    public byte[] detectFace(MultipartFile file){
        MatOfRect faceDetections = new MatOfRect();
        CascadeClassifier faceDetector = new CascadeClassifier(this.getClass().getResource(FACE_LIB_PATH).getPath());

        Mat image = Imgcodecs.imdecode(new MatOfByte(file.getBytes()), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
        faceDetector.detectMultiScale(image, faceDetections);

        log.info("Face detected. Count = {}", faceDetections.size());

        drawRect(faceDetections, image);

        return mat2Image(image);
    }

    private void drawRect(MatOfRect faceDetections, Mat image) {
        for(Rect rect: faceDetections.toArray()){
            Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
        }
    }

    private byte[] mat2Image(Mat frame) {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".jpg", frame, buffer);
        return buffer.toArray();
    }
}
