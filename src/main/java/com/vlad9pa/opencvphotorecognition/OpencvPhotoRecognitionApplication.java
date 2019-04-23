package com.vlad9pa.opencvphotorecognition;

import lombok.RequiredArgsConstructor;
import nu.pattern.OpenCV;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@RequiredArgsConstructor
public class OpencvPhotoRecognitionApplication {

    public static void main(String[] args) {
        OpenCV.loadShared();
        SpringApplication.run(OpencvPhotoRecognitionApplication.class, args);
    }

}
